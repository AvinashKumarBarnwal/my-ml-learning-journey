# Assignment 02 — Breast Cancer Classification using Logistic Regression

## Problem Statement

The objective of this assignment is to implement a complete Machine Learning workflow for solving a **binary classification** problem using **Logistic Regression**.

The assignment uses the **Breast Cancer Wisconsin Dataset**, available through the Scikit-Learn library, to classify tumors based on multiple medical features.

The workflow includes data exploration, preprocessing, model training, evaluation, and hyperparameter experimentation.

---

# Assignment Tasks

## 1. Dataset Loading and Basic Exploration

Load the Breast Cancer Wisconsin Dataset and perform an initial exploration.

### Objectives

- Load the dataset
- Display dataset shape
- Display first five records
- Examine data types
- Analyze target class distribution
- Identify missing values

---

## 2. Exploratory Data Analysis (EDA)

Perform Exploratory Data Analysis to better understand the dataset.

### Required Analysis

- Class Distribution
- Feature Distribution
- Correlation Heatmap
- Outlier Detection using Boxplots

### Expected Outcome

Summarize the key observations and patterns identified from the exploratory analysis.

---

## 3. Data Preprocessing

Prepare the dataset for model training.

### Tasks

- Split data into Training and Testing sets (80:20)
- Apply StandardScaler for feature scaling

### Expected Outcome

Explain the importance of feature scaling for Logistic Regression and report the training and testing dataset shapes.

---

## 4. Logistic Regression Model

Build a Logistic Regression model for binary classification.

### Tasks

- Train the model
- Display model coefficients
- Display model intercept

---

## 5. Model Evaluation

Evaluate the classification model using standard evaluation metrics.

### Required Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

Display the confusion matrix using a heatmap and interpret the overall model performance.

---

## 6. Hyperparameter Experimentation

Train multiple Logistic Regression models using different values of the regularization parameter:

- C = 0.01
- C = 0.1
- C = 1
- C = 10

### Expected Outcome

Compare the accuracy obtained for each model and identify the best-performing hyperparameter configuration.

---

# Learning Focus

This assignment focuses on understanding the complete workflow of solving a binary classification problem using Logistic Regression, including:

- Data Exploration
- Exploratory Data Analysis
- Data Preprocessing
- Feature Scaling
- Binary Classification
- Model Evaluation
- Hyperparameter Tuning
- Practical Machine Learning Workflow