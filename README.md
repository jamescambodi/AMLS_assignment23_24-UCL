# AMLS_assignment23_24-UCL

# Project Organisation

This project uses pre-trained models to analyse medical image datasets for binary and multi-class classification.

## Role of Each File

- `binary_classifier_model.pkl`: A pre-trained CNN model for binary classification on the PneumoniaMNIST dataset.
- `multi_classifier_model.pkl`: A pre-trained CNN model for multi-class classification on the PathMNIST dataset.
- `Datasets folder: Paste in PneumoniaMNIST and PathMNIST datasets
- `main.py`: This is where datasets are loaded and pre-processed, and predictions are made using the pre-trained models. This also includes classification report generation for each model and a confusion matrix for the multi-classification.

## Packages Required

To run the code, the following packages are required:
- pandas: For data manipulation and analysis.
- tensorflow: For utilising deep learning models.
- numpy: For numerical computing and working with arrays.
- medmnist: For accessing the PneumoniaMNIST and PathMNIST datasets.
- platform: To retrieve as much platform-identifying data as possible.
- pickle: For loading the pre-trained model files.
- seaborn: For visualising the confusion matrix.
- matplotlib: For plotting purposes.
- scikit-learn: For data preprocessing, model evaluation, and generating classification reports.

These are libraries required before running the code. 
