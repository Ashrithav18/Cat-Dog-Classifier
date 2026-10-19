# Cat vs Dog Image Classification

## 1. Project Overview

The Cat vs Dog Image Classification project is a deep-learning application that predicts whether an uploaded image contains a cat or a dog. It uses a trained MobileNetV2 model for classification and a pretrained Vision Transformer (ViT) model for additional image-content checking.

The application is built using Python and Streamlit and supports multiple image uploads, image validation, and prediction confidence display.

## 2. Objectives

- Classify images into Cat or Dog.
- Resize images to a consistent size of 224 × 224 pixels.
- Preprocess image pixels before prediction.
- Use transfer learning to build the classification model.
- Evaluate model performance using test accuracy, a confusion matrix, and a classification report.
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

MobileNetV2 is the primary classification model. It uses ImageNet-pretrained weights and transfer learning. Its pretrained feature layers are frozen, and a binary classification layer is added to distinguish cats from dogs.

### Vision Transformer (ViT)

A pretrained ViT image-classification model is used as an additional image-content checker. It checks whether the uploaded image appears to contain a cat or dog before the main classifier makes its prediction.

The ViT model is not trained on the project's cat-and-dog dataset.

## 5. Dataset Structure

The training script expects the dataset in this structure:

```text
data/
├── cats/
└── dogs/
```

The training script validates the image files and skips corrupted images and unsupported file types.

The valid dataset is split into:
- 70% training data
- 15% validation data
- 15% testing data

## 6. Project Structure

```text
Cat-Dog-Classifier/
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── cats/
│   └── dogs/
├── models/
│   └── cat_dog_classifier.keras
└── outputs/
    ├── confusion_matrix.png
    └── training_accuracy.png
```

## 7. Installation and Execution

### Install dependencies

Create and activate a Python virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

### Train the model

Ensure the dataset is available in `data/cats` and `data/dogs`, then run:

```bash
python train.py
```

The trained model is saved in the `models` directory. Evaluation graphs are saved in the `outputs` directory.

### Run the application

After the trained model has been saved, run:

```bash
streamlit run app.py
```

Upload one or more images to view the predictions.

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

The following test cases are used to verify the application:

| Test Case | Description | Expected Result |
|---|---|---|
| TC1 | Upload a cat image | Cat prediction |
| TC2 | Upload a dog image | Dog prediction |
| TC3 | Upload a corrupted image | Clear error message without crashing |
| TC4 | Upload an unsupported file | Unsupported-format message |
| TC5 | Upload multiple images | Each image is processed individually |
| TC6 | Upload an unseen cat or dog image | Prediction without retraining |


## 10. Conclusion

This project demonstrates image preprocessing, transfer learning, binary image classification, model evaluation, and deployment through a Streamlit application. It also includes image validation and an additional pretrained image-recognition step to improve the upload workflow.