# Cat vs Dog Image Classification

## 1. Project Overview

Cat vs Dog Image Classification is a deep-learning application that predicts whether an uploaded image contains a cat or a dog. It uses MobileNetV2 with transfer learning as the primary classifier and a pretrained Vision Transformer (ViT) model for additional image-content checking.

The application is built using Python and Streamlit. It supports multiple image uploads, image validation, and prediction confidence display.

## 2. Objectives

- Classify images into Cat or Dog.
- Resize images to a consistent size of 224 × 224 pixels.
- Preprocess images before prediction.
- Use transfer learning to build the classification model.
- Evaluate performance using test accuracy, a confusion matrix, and a classification report.
- Handle corrupted images and unsupported file formats.
- Support predictions on multiple uploaded images.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- Vision Transformer (ViT)
- Hugging Face Transformers
- PyTorch
- Streamlit
- NumPy
- Pillow
- Scikit-learn
- Matplotlib

## 4. Models Used

### MobileNetV2

MobileNetV2 is the primary classification model. It uses ImageNet-pretrained weights and transfer learning. Its pretrained feature layers are frozen, and a binary classification head is added to distinguish cats from dogs.

### Vision Transformer (ViT)

A pretrained ViT image-classification model is used as an additional image-content checker in the upload workflow. It is not trained on this project's cat-and-dog dataset. Its output is used by the application to help assess whether an uploaded image appears suitable for cat-or-dog classification.

## 5. Dataset

The training script expects labelled images in the following structure:

```text
data/
├── cats/
└── dogs/
```

The script validates the image files and skips corrupted images and unsupported files.

The valid dataset is split into:

- 70% training data
- 15% validation data
- 15% testing data

The dataset used in the reported training run contained 24,998 valid images: 12,499 cat images and 12,499 dog images. Two corrupted images and two unsupported files were skipped.

**Note:** The dataset is not included in this GitHub repository. To retrain the model, obtain a suitable labelled cat-and-dog dataset and arrange it in the folder structure shown above.

## 6. Project Structure

```text
Cat-Dog-Classifier/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── App_Screenshots/
├── models/
│   └── cat_dog_classifier.keras
└── outputs/
    ├── confusion_matrix.png
    └── training_accuracy.png
```

The `data/` directory and Python virtual environment are kept locally and excluded from GitHub.

## 7. Installation and Execution

### Create and activate a virtual environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Train the model

Ensure the dataset is available in `data/cats/` and `data/dogs/`, then run:

```bash
python train.py
```

The trained model is saved to `models/cat_dog_classifier.keras`. Evaluation graphs are saved in the `outputs/` directory.

### Run the application

Ensure the trained model exists, then run:

```bash
streamlit run app.py
```

Upload one or more supported images to view the predictions. The application also validates uploaded files and handles unreadable images.

## 8. Evaluation

The primary classifier was evaluated on a held-out test dataset containing 3,750 images.

- **Test Accuracy:** 98.88%
- **Test Loss:** 0.0347
- **Correct Predictions:** 3,708
- **Incorrect Predictions:** 42

### Confusion Matrix

| Actual Class | Predicted Cat | Predicted Dog |
|---|---:|---:|
| Cat | 1,851 | 24 |
| Dog | 18 | 1,857 |

The classification report showed precision, recall, and F1-score of 0.99 for both classes when rounded to two decimal places.

Evaluation graphs are available in the `outputs/` directory:

- `confusion_matrix.png`
- `training_accuracy.png`

## 9. Test Cases

| Test Case | Input | Expected Behaviour |
|---|---|---|
| TC1 | Valid cat image | Predict CAT if recognized correctly |
| TC2 | Valid dog image | Predict DOG if recognized correctly |
| TC3 | Corrupted image | Display an error without crashing |
| TC4 | Unsupported file format | Reject it clearly |
| TC5 | Multiple images | Return one prediction for each image |
| TC6 | Previously unseen cat or dog image | Run inference without retraining |

Actual test results should be recorded based on the tests performed. Application screenshots are provided in the `App_Screenshots/` directory.

## 10. Limitations

The primary model returns one overall CAT or DOG classification for each image. It is an image-classification system, not an object-detection model that locates and counts individual animals within a photograph.

Predictions can be incorrect for ambiguous, low-quality, or out-of-scope images. The additional pretrained ViT checker does not guarantee perfect rejection of every image that contains neither a cat nor a dog.

## 11. Conclusion

This project demonstrates image preprocessing, transfer learning, binary image classification, model evaluation, and deployment through a Streamlit application. It includes a held-out test evaluation, a confusion matrix, support for multiple uploaded images, and validation for corrupted or unsupported files.