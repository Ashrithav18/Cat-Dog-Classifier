import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Do not allow partially corrupted images to load silently.
ImageFile.LOAD_TRUNCATED_IMAGES = False

print("=" * 50)
print("       CAT VS DOG IMAGE CLASSIFICATION")
print("=" * 50)


# ==========================================
# 2. VALIDATE IMAGE FILES
# ==========================================

print("\nChecking dataset images...")

image_paths = []
labels = []

class_folders = [
    ("cats", 0),
    ("dogs", 1)
]

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

corrupted_count = 0
unsupported_count = 0

for folder_name, label in class_folders:
    folder_path = DATA_DIR / folder_name

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    files = [
        path for path in folder_path.iterdir()
        if path.is_file()
    ]

    print(f"\nChecking {folder_name} images...")

    for file_path in files:
        if file_path.suffix.lower() not in valid_extensions:
            unsupported_count += 1
            continue

        try:
            # Verify file structure.
            with Image.open(file_path) as image:
                image.verify()

            # Reopen and fully decode the image.
            with Image.open(file_path) as image:
                image.convert("RGB").load()

            image_paths.append(str(file_path))
            labels.append(label)

        except Exception:
            corrupted_count += 1

    print(f"Finished checking {folder_name}.")

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

if len(image_paths) == 0:
    raise ValueError("No valid images found in data/cats and data/dogs.")

if len(np.unique(labels)) != 2:
    raise ValueError("Both cat and dog images are required.")

print("\nDataset checking completed.")
print("Valid images:", len(image_paths))
print("Corrupted images skipped:", corrupted_count)
print("Unsupported files skipped:", unsupported_count)
print("Valid cat images:", int(np.sum(labels == 0)))
print("Valid dog images:", int(np.sum(labels == 1)))


# ==========================================
# 3. SPLIT DATASET: 70 / 15 / 15
# ==========================================

print("\nSplitting dataset...")

X_train, X_remaining, y_train, y_remaining = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=y_remaining
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))


# ==========================================
# 4. IMAGE PREPROCESSING WITH PILLOW
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image_with_pillow(path):
    """Decode an image, convert it to RGB, resize and preprocess it."""

    if isinstance(path, np.ndarray):
        path = path.item()

    if isinstance(path, bytes):
        path = path.decode("utf-8")

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMG_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    return preprocess_input(image_array)


def load_and_preprocess_image(path, label):
    image = tf.numpy_function(
        load_image_with_pillow,
        [path],
        tf.float32
    )

    image.set_shape([224, 224, 3])

    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [1])

    return image, label


def create_dataset(paths, dataset_labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, dataset_labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_dataset = create_dataset(
    X_train, y_train, training=True
)

validation_dataset = create_dataset(
    X_val, y_val
)

test_dataset = create_dataset(
    X_test, y_test
)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ==========================================
# 6. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Keep pretrained feature layers frozen.
base_model.trainable = False


# ==========================================
# 7. BUILD CLASSIFICATION MODEL
# ==========================================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])


# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ==========================================
# 10. EVALUATE ON UNSEEN TEST IMAGES
# ==========================================

print("\nEvaluating test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# 11. GENERATE TEST PREDICTIONS
# ==========================================

y_true = []
y_pred = []

for images, labels_batch in test_dataset:
    predictions = model.predict(images, verbose=0)

    predicted_labels = (
        predictions >= 0.5
    ).astype(int).flatten()

    y_true.extend(
        labels_batch.numpy().flatten().astype(int)
    )

    y_pred.extend(predicted_labels)


# ==========================================
# 12. CONFUSION MATRIX AND REPORT
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix (actual rows, predicted columns):")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Cat", "Dog"],
        zero_division=0
    )
)


# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(7, 5))
plt.imshow(cm, cmap="Blues")

plt.title("Cat vs Dog - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

for row in range(2):
    for col in range(2):
        plt.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=150
)

plt.close()


# ==========================================
# 14. SAVE TRAINING ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_accuracy.png",
    dpi=150
)

plt.close()


# ==========================================
# 15. SAVE TRAINED MODEL
# ==========================================

model_path = MODEL_DIR / "cat_dog_classifier.keras"
model.save(model_path)

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Model saved at: {model_path}")
print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")
print(f"Training graph: {OUTPUT_DIR / 'training_accuracy.png'}")
