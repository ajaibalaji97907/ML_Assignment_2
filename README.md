# ML_Assignment_2

# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence of heart disease using multiple classification algorithms and provides an interactive Streamlit web application for model comparison and evaluation.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Implemented](#models-implemented)
- [Model Performance Comparison](#model-performance-comparison)
- [Key Observations](#key-observations)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)


## Problem Statement

This project predicts the presence of heart disease using machine learning classification models. The objective is to compare multiple models and evaluate their performance using standard evaluation metrics including accuracy, AUC, precision, recall, F1-score, and Matthews Correlation Coefficient (MCC).

## Dataset Description

The Heart Disease dataset from the UCI Machine Learning Repository contains clinical and diagnostic features used to predict heart disease.

**Dataset Characteristics:**
- **Records:** 900+ patient records
- **Features:** Multiple categorical and numerical attributes
- **Target Variable:**
  - `0` → No heart disease
  - `1` → Heart disease present

## Models Implemented

The following machine learning models have been implemented and compared:

1. **Logistic Regression** - Linear classification model
2. **Decision Tree** - Tree-based model for non-linear patterns
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble learning method using multiple decision trees
6. **XGBoost** - Gradient boosting ensemble method

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.8167 | 0.9286 | 0.8400 | 0.7500 | 0.7925 | 0.6325 |
| **Decision Tree** | 0.7333 | 0.7277 | 0.7500 | 0.6429 | 0.6923 | 0.4637 |
| **KNN** | 0.8333 | 0.9051 | 0.8750 | 0.7500 | 0.8077 | 0.6683 |
| **Naive Bayes** | 0.6167 | 0.9330 | 1.0000 | 0.1786 | 0.3030 | 0.3223 |
| **Random Forest** | 0.8167 | 0.9280 | 0.8696 | 0.7143 | 0.7843 | 0.6367 |
| **XGBoost** | 0.8333 | 0.8895 | 0.8750 | 0.7500 | 0.8077 | 0.6683 |

### Best Performing Models
- **Highest Accuracy:** KNN & XGBoost (83.33%)
- **Highest AUC:** Naive Bayes (93.30%)
- **Best Balanced Performance:** XGBoost (Accuracy: 83.33%, MCC: 0.6683)

## Key Observations

### Logistic Regression
- Shows strong performance with high AUC (0.9286)
- Indicates good probability discrimination
- Provides balanced classification capability
- Suitable for linear relationships in the data

### Decision Tree
- Performs moderately with 73.33% accuracy
- Shows lower MCC (0.4637), suggesting potential overfitting
- Sensitive to data variations
- May require pruning or ensemble methods for better performance

### K-Nearest Neighbors (KNN)
- Achieves high accuracy (83.33%)
- Provides balanced metrics across precision and recall
- Effective when similar patterns exist in the data
- Performance depends on optimal k value selection

### Naive Bayes
- Shows very high precision (1.0) but extremely low recall (0.1786)
- Misses many positive cases (low sensitivity)
- **Not suitable for this dataset** due to poor recall
- High AUC indicates good probability estimates despite poor classification

### Random Forest
- Provides stable performance with good MCC (0.6367)
- Demonstrates robustness and reduced overfitting
- Benefits from ensemble approach
- Good balance between bias and variance

### XGBoost
- **Recommended Model:** Achieves the best overall balance
- High accuracy (83.33%) and strong MCC (0.6683)
- Most reliable model for this dataset
- Excellent for production deployment

## Features

The Streamlit application provides:

- **Upload Test Dataset** - Upload CSV files for prediction
- **Model Selection** - Choose from 6 different ML models
- **Evaluation Metrics** - View comprehensive performance metrics
- **Confusion Matrix** - Interactive confusion matrix visualization
- **Classification Report** - Detailed classification report table
- **Performance Comparison** - Compare all models side-by-side

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

## Usage

### Training Models

Train all machine learning models on the dataset:

```bash
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 6 models
- Save trained models to disk
- Generate performance metrics

### Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. Upload your test dataset (CSV format)
2. Select a machine learning model from the dropdown
3. View the evaluation metrics and visualizations
4. Compare different models using the performance table
5. Analyze the confusion matrix and classification report

## Requirements

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost
joblib
```

For the complete list with versions, see `requirements.txt`


