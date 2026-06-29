# STEP 1 — Load Dataset and Basic Exploration

# Step 1.1 — Import necessary libraries
# Data Handling
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from sklearn.datasets import load_breast_cancer

# Step 1.2 — Load the dataset
# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Step 1.3 — Create a DataFrame
# Create a DataFrame from the dataset
df = pd.DataFrame(
    breast_cancer_data.data,
    columns=breast_cancer_data.feature_names
)

# Add target column [malignant (0) and benign (1)]
# Malignant: Cancerous tumor. Can grow aggressively and spread to other parts of the body (metastasis).
# Benign: Non-cancerous tumor. Generally grows slowly and does not spread to other parts of the body.
df['target'] = breast_cancer_data.target

# Step 1.4 — Dataset Shape
print("--- Shape of dataset ---:")
print(df.shape)

# Step 1.5 — First few rows of the dataset
print("\n--- First 5 rows of the dataset ---:")
print(df.head())

# Step 1.6 — Data Types
print("\n--- Data types of each column ---:")

print(df.dtypes)

# Step 1.7 — Target Class Distribution
print("\n--- Target class distribution ---:")
target_names = {
    0: 'malignant', # Cancerous tumor
    1: 'benign' # Non-cancerous tumor
}

print(df['target'].map(target_names).value_counts())

# Step 1.8 — Missing Values
print("\n--- Missing values in each column ---:")
print(df.isnull().sum())

# Step 2: Exploratory Data Analysis (EDA)
# Step 2.1 - Class Distribution Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Class Distribution of Target Variable')
plt.xlabel('Tumor Type')
plt.ylabel('Count')

plt.xticks(
    ticks=[0,1],
    labels=['Malignant', 'Benign']
)
plt.savefig("../images/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 2.2 - Feature Distributions
# Instead of plotting all 30 features together (too cluttered), I have selected
#  a few important and commonly used ones:

# Plot histograms for selected features

selected_features = [
    'mean radius',
    'mean texture',
    'mean area',
    'mean smoothness'
]

df[selected_features].hist(
    figsize=(12,8),
    bins=20
)

plt.suptitle('Feature Distributions')
plt.savefig("../images/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 2.3 - Correlation Analysis
# Correlation heatmap

plt.figure(figsize=(28, 22))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    cmap='coolwarm',
    square=True,
    cbar_kws={'shrink': 0.9}
)

plt.title('Correlation Heatmap', pad=20)

# Rotate labels for readability
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# Adjust layout to keep all feature names visible
plt.subplots_adjust(left=0.16, bottom=0.24, right=0.97, top=0.94)
plt.savefig("../images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 2.4 Outlier Detection
# Box plots for selected features to detect outliers
# Selected numerical features were visualized to understand the overall distribution,
# spread, and variability of the dataset. Since the dataset contains many numerical features, 
# a representative subset was chosen for clearer and more interpretable visual analysis.

selected_box_features = [
    'mean radius',
    'mean texture',
    'mean area'
]

# Boxplots for outlier detection

plt.figure(figsize=(12,6))

for i, feature in enumerate(selected_box_features, 1):

    plt.subplot(1,3,i)

    sns.boxplot(y=df[feature])

    plt.title(feature)

plt.tight_layout()

plt.savefig("../images/boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 3: Data Preprocessing
# Step 3.1 - Feature and Target variable Separation
# Separating the features (X) and target variable (y = malignant or benign)
X = df.drop('target', axis=1)
y = df['target']

# Step 3.2 - Train-Test Split
from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Step 3.3 — Print Training and Testing Shapes
print("\n--- Training and Testing Dataset Shapes ---")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 3.4 - Feature Scaling using StandardScaler
# To bring all features to the same scale, otherswise features with larger 
# scales can dominate the learning process and lead to suboptimal model performance.

# Standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #Fit (Learn mean and std on training data) and 
# transform() is applying the scaling to the training data

X_test_scaled = scaler.transform(X_test) # Only transform the test data using the same parameters learned from the training data. 
# No fitting on test data to prevent data leakage.

# Step 4.0 — Understanding Generic Sigmoid Function
# Visualizing Sigmoid Function

x = np.linspace(-10, 10, 100)

y = 1 / (1 + np.exp(-x))

plt.figure(figsize=(8,5))

plt.plot(x, y)

plt.title('Sigmoid Function')
plt.xlabel('Input Value (z)')
plt.ylabel('Probability')

plt.grid(True)

plt.savefig("../images/sigmoid_function.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 5: Train Logistic Regression Model
# Step 5.1 - Import Logistic Regression 
from sklearn.linear_model import LogisticRegression

# Step 5.2 - Create Logistic Regression model instance
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# Step 5.3 - Train the model on the training data
# Here scaled training data (X_train_scaled) has been used to train the model, not the original unscaled data (X_train).
logistic_model.fit(X_train_scaled, y_train)

# Step 5.4 - Print model coefficients
# Coefficients represent how strongly a feature influences prediction.
# positive coefficient → pushes prediction toward benign
# negative coefficient → pushes prediction toward malignant
print("\n--- Logistic Regression Coefficients ---:")
print(logistic_model.coef_)

# Step 5.5 - Print model intercept
# The intercept represents the baseline log-odds of the positive class (benign) when all features are zero.
print("\n--- Logistic Regression Intercept ---:")
print(logistic_model.intercept_)


# Step 6: Predicted Probability Distribution
# Predict probabilities

y_prob = logistic_model.predict_proba(X_test_scaled)

# Probability of malignant tumor (class 0)
malignant_prob = y_prob[:, 0] # 0 index corresponds to malignant class

# Plot probability distribution

plt.figure(figsize=(8,5))

plt.hist(malignant_prob, bins=20)

# Threshold line
plt.axvline(
    x=0.5,
    color='red',
    linestyle='--',
    label='Threshold = 0.5'
)

plt.legend()

plt.title('Predicted Probability Distribution')
plt.xlabel('Probability of Malignant Tumor')
plt.ylabel('Frequency')

plt.savefig("../images/probability_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 7: Model Evaluation
# Step 7.1 - Predict on the test data
# Internally probabilities are calculated using the logistic function (sigmoid) and then converted to class labels based on a threshold (default is 0.5). 
# Then Class 0 (malignant) is predicted if the probability of malignant is greater than or equal to 0.5, otherwise Class 1 (benign) is predicted.
y_pred = logistic_model.predict(X_test_scaled)

# Step 7.2 - Accuracy Score
# Accuracy is the ratio of correctly predicted observations to the total observations. It gives an overall measure of how well the model is performing in terms of correct classifications.
# However, accuracy alone can be misleading in imbalanced datasets (where one class is much more frequent than the other), as a model could achieve high accuracy by simply predicting the majority class. Therefore, it is important to consider other metrics (like precision, recall, F1-score) in addition to accuracy for a more comprehensive evaluation of the model's performance.
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Accuracy Score ---:")  
print(accuracy)

# Step 7.3 - Confusion Matrix
# A confusion matrix is a table that summarizes the performance of a classification model by showing the counts
# of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
# For our scenario, False negative (FN) is more critical because it represents cases where the model incorrectly predicts a malignant tumor as benign, which can lead to delayed diagnosis and treatment. 
# Therefore, minimizing false negatives is crucial in medical diagnosis to ensure that patients with malignant tumors receive timely and appropriate care.
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---:")
print(conf_matrix)

# Step 7.4 - Confusion Matrix Heatmap
# Plot confusion matrix heatmap

plt.figure(figsize=(6,5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

plt.xticks(
    ticks=[0.5,1.5],
    labels=['Malignant', 'Benign']
)

plt.yticks(
    ticks=[0.5,1.5],
    labels=['Malignant', 'Benign'],
    rotation=0
)

plt.savefig("../images/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 7.5 - Precision
# Precision is the ratio of true positives (TP) to the sum of true positives and false positives (FP). It measures the accuracy of positive predictions.
# In our context, precision tells us how many of the tumors predicted as malignant were actually malignant
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("\n--- Precision Score ---:")
print(precision)

# Step 7.6 - Recall
# Recall is the ratio of true positives (TP) to the sum of true positives and false negatives (FN). It measures the ability of the model to identify all relevant instances.
# In our context, recall tells us how many of the actual malignant tumors were correctly identified by the model.
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("\n--- Recall Score ---:")
print(recall)

# Step 7.7 - F1-Score
# F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, especially useful when the class distribution is imbalanced.
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("\n--- F1-Score ---:")
print(f1)

# Step 7.8 - Classification Report
# A classification report provides a comprehensive summary of the precision, recall, F1-score, and support (number of true instances for each class) for each class in the dataset. It helps to evaluate the performance of a classification model in a more detailed manner.
from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
print("\n--- Classification Report ---:")   
print(class_report)

# Step 8: Hyperparameter Experimentation

# Step 8.1 - Experiment with different C values
# The C parameter in logistic regression is the inverse of regularization strength. 
# A smaller C value means stronger regularization, which can help prevent overfitting by penalizing large coefficients. 
# Conversely, a larger C value means weaker regularization, allowing the model to fit the training data more closely, which can lead to overfitting if the model captures noise in the data.

c_values = [0.01, 0.1, 1, 10]

print("--- Hyperparameter Experimentation ---\n")

best_accuracy = 0
best_c = None

for c in c_values:

    # Create model with different C value
    model = LogisticRegression(C=c)

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"C = {c} --> Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c = c

print(f"\nBest C value: {best_c} with Best Accuracy: {best_accuracy:.4f}")

print("\nFinal Observation")
print("-------------------")
print("Logistic Regression achieved high classification performance on the Breast Cancer dataset.")
print(f"Best Hyperparameter (C): {best_c}")
print(f"Best Accuracy: {best_accuracy:.4f}")