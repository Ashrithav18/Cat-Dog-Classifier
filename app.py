
import re

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Cat vs Dog AI",
    page_icon="🐾",
    layout="wide"
)

IMG_SIZE = (224, 224)
MODEL_PATH = "models/cat_dog_classifier.keras"

SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Minimum ViT confidence for accepting a cat/dog label.
VIT_MIN_CONFIDENCE = 0.15


# -------------------------------
# Custom styling
# -------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f5f8ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    .hero {
        background: linear-gradient(120deg, #173b73, #3976d2);
        padding: 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e6efff;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        color: #173b73;
        font-size: 23px;
        font-weight: 700;
        margin-top: 14px;
        margin-bottom: 12px;
    }

    .stFileUploader {
        background: white;
        border: 1px dashed #8caee5;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 16px;
        border-color: #dce6f5;
    }

    div[data-testid="stMetric"] {
        background: #f0f5ff;
        border: 1px solid #dce7fb;
        border-radius: 12px;
        padding: 12px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Header
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>🐱 Cat vs Dog AI Classifier 🐶</h1>
    <p>
        Intelligent image classification powered by deep learning.
        Upload your images and let AI predict the result.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_cat_dog_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_image_checker():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )


try:
    model = load_cat_dog_model()
    image_checker = load_image_checker()

except Exception as error:
    st.error("Could not load the required AI model.")
    st.info(
        "Check that your trained model exists and that the "
        "required libraries are installed. Internet access may "
        "be needed the first time the ViT model is downloaded."
    )
    st.code(str(error))
    st.stop()


# -------------------------------
# Image preprocessing
# -------------------------------
def prepare_image(uploaded_file):
    uploaded_file.seek(0)

    with Image.open(uploaded_file) as original:
        original.load()
        image = original.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# -------------------------------
# Identify cat/dog labels from ViT
# -------------------------------
def get_animal_type(label):
    label = label.lower().replace("_", " ").strip()

    # Cat labels and common ImageNet cat classes
    cat_patterns = [
        r"\bcat\b",
        r"\bkitten\b",
        r"\btabby\b",
        r"\btiger cat\b",
        r"\bpersian cat\b",
        r"\bsiamese cat\b",
        r"\begyptian cat\b",
    ]

    # Dog labels and common dog-breed names
    dog_patterns = [
        r"\bdog\b",
        r"\bpuppy\b",
        r"\bretriever\b",
        r"\bterrier\b",
        r"\bhound\b",
        r"\bspaniel\b",
        r"\bshepherd\b",
        r"\bpoodle\b",
        r"\bchihuahua\b",
        r"\bhusky\b",
        r"\bbeagle\b",
        r"\bbulldog\b",
        r"\bmastiff\b",
        r"\bgreyhound\b",
        r"\bwhippet\b",
        r"\bcollie\b",
        r"\bpinscher\b",
        r"\bmalamute\b",
        r"\bpomeranian\b",
        r"\bdalmatian\b",
        r"\brottweiler\b",
        r"\bdoberman\b",
        r"\bboxer\b",
        r"\bshih[- ]?tzu\b",
        r"\bpug\b",
        r"\bbasenji\b",
        r"\bchow\b",
        r"\bkomondor\b",
        r"\bpekinese\b",
        r"\bborzoi\b",
        r"\bbrabancon\b",
        r"\beskimo dog\b",
    ]

    if any(re.search(pattern, label) for pattern in cat_patterns):
        return "cat"

    if any(re.search(pattern, label) for pattern in dog_patterns):
        return "dog"

    return None


# -------------------------------
# ViT content check
# -------------------------------
def check_cat_or_dog_image(image):
    results = image_checker(image, top_k=5)

    animal_candidates = []

    for result in results:
        animal_type = get_animal_type(result["label"])

        if animal_type is not None:
            animal_candidates.append({
                "type": animal_type,
                "label": result["label"],
                "score": float(result["score"])
            })

    if not animal_candidates:
        return None, 0.0, results

    # Use the highest-confidence cat/dog label found.
    best_candidate = max(
        animal_candidates,
        key=lambda item: item["score"]
    )

    # Do not accept weak cat/dog matches.
    if best_candidate["score"] < VIT_MIN_CONFIDENCE:
        return None, best_candidate["score"], results

    return (
        best_candidate["type"],
        best_candidate["score"],
        results
    )


# -------------------------------
# Upload section
# -------------------------------
st.markdown(
    '<div class="section-title">📤 Upload your images</div>',
    unsafe_allow_html=True
)

st.write(
    "Select one or more cat or dog photographs. "
    "Each image will be checked and classified separately."
)

uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=None,
    accept_multiple_files=True,
    help="Supported formats: JPG, JPEG, PNG, BMP and WebP."
)

st.caption("Supported formats: JPG, JPEG, PNG, BMP and WebP.")


# -------------------------------
# Prediction results
# -------------------------------
if uploaded_files:
    st.markdown(
        '<div class="section-title">🔍 Classification results</div>',
        unsafe_allow_html=True
    )

    for uploaded_file in uploaded_files:
        with st.container(border=True):
            st.markdown(f"**📄 {uploaded_file.name}**")

            extension = (
                uploaded_file.name.rsplit(".", 1)[-1].lower()
                if "." in uploaded_file.name
                else ""
            )

            if extension not in SUPPORTED_FORMATS:
                st.error(
                    "Unsupported file format. Please upload "
                    "a JPG, JPEG, PNG, BMP or WebP image."
                )
                continue

            try:
                image, processed_image = prepare_image(uploaded_file)

                image_col, result_col = st.columns(
                    [1, 1.5],
                    gap="large"
                )

                with image_col:
                    st.image(
                        image,
                        caption="Uploaded image",
                        use_container_width=True
                    )

                with result_col:
                    with st.spinner("Checking image content..."):
                        vit_type, vit_confidence, vit_results = (
                            check_cat_or_dog_image(image)
                        )

                    # Show the check results for debugging/transparency.
                    with st.expander("View image-check details"):
                        st.write("ViT top predictions:")

                        for item in vit_results[:5]:
                            st.write(
                                f"- {item['label']}: "
                                f"{item['score'] * 100:.2f}%"
                            )

                    # Stop before the main model if ViT rejects the image.
                    if vit_type is None:
                        st.warning(
                            "The image checker could not confidently "
                            "identify this as a cat or dog. No final "
                            "prediction was made. Try a clear cat or "
                            "dog photograph."
                        )
                        continue

                    st.caption(
                        f"Image checker: {vit_type.title()} "
                        f"({vit_confidence * 100:.2f}%)"
                    )

                    with st.spinner("Running the trained classifier..."):
                        score = float(
                            model.predict(
                                processed_image,
                                verbose=0
                            )[0][0]
                        )

                    # This model was trained with cat = 0 and dog = 1.
                    model_type = "dog" if score >= 0.5 else "cat"
                    model_confidence = (
                        score if score >= 0.5 else 1.0 - score
                    )

                    # Do not give a final label if the two models disagree.
                    if model_type != vit_type:
                        st.warning(
                            "The two models disagree about this image. "
                            "To avoid showing a potentially incorrect "
                            "answer, no final cat/dog prediction is "
                            "displayed. Try a clearer image."
                        )

                        st.caption(
                            f"Trained classifier: {model_type.title()} "
                            f"({model_confidence * 100:.2f}%)"
                        )
                        continue

                    # Show prediction only when both models agree.
                    label = (
                        "Dog 🐶" if model_type == "dog"
                        else "Cat 🐱"
                    )

                    st.success(f"### Prediction: {label}")

                    st.metric(
                        "Model confidence",
                        f"{model_confidence * 100:.2f}%"
                    )

                    st.progress(
                        max(0.0, min(1.0, model_confidence))
                    )

                    if model_confidence < 0.70:
                        st.warning(
                            "The trained classifier is uncertain. "
                            "Try a clearer photograph."
                        )
                    else:
                        st.caption(
                           " Prediction completed successfully."
                        )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):
                st.error(
                    "This file could not be read as a valid image. "
                    "Please upload a valid image."
                )

            except Exception as error:
                st.error("Could not process this image.")
                st.caption(str(error))

else:
    with st.container(border=True):
        st.markdown("### 🐾 Ready to classify?")
        st.write(
            "Upload a cat or dog image above to see its prediction "
            "and model confidence here."
        )


# -------------------------------
# Footer
# -------------------------------
st.divider()

st.markdown("""
<div class="footer">
    Built with Streamlit, TensorFlow, MobileNetV2 and Vision Transformer (ViT)
    <br>
    Cat vs Dog Image Classification Project
</div>
""", unsafe_allow_html=True)
