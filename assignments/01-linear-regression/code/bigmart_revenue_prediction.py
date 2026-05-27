# Step 1: Import Required Libraries
# Data Handling
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Step 2: Load the Dataset
root_dir = Path(__file__).resolve().parent.parent
data_path = root_dir / "dataset" / "bigmart.csv"

sales_df = pd.read_csv(data_path)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(sales_df.head())

# Step 3: Basic Data Information
print("\nDataset Shape:") 
print(sales_df.shape) #Rows and Columns

print("\nColumn Names:")
print(sales_df.columns) 

print("\nDataset Information:") 
print(sales_df.info()) #Data types and non-null counts

# Step 4: Statistical Summary
print("\nSummary Statistics:")
print(sales_df.describe())  

# Step 5: Missing Values Analysis
print("\nMissing Values:")
print(sales_df.isnull().sum())

# Step 6: Exploratory Data Analysis (EDA)

# Visualization 1: Histogram of Item_Outlet_Sales
# Distribution of Item_Outlet_Sales     
plt.figure(figsize=(8,5))
plt.hist(sales_df['Item_Outlet_Sales'], bins=30)
plt.title('Distribution of Item Outlet Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Correlation Heatmap
# We are using Corelation Heatmap to visualize the relationships between numerical features and the target variable (Item_Outlet_Sales).
# This helps us identify which features are strongly correlated with the target variable, which can be useful for
# feature selection and understanding the underlying patterns in the data.

plt.figure(figsize=(10,8))
numeric_df = sales_df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()                    # Adjust layout to prevent cropping
plt.show()

# Visualization 3: Scatter Plot
# Scatter plot of Item_MRP vs Item_Outlet_Sales
# This scatter plot helps us visualize the relationship between the maximum retail price (Item_MRP) and the sales (Item_Outlet_Sales).
# It can reveal whether there is a positive correlation (higher MRP leads to higher sales) or if there are any outliers in the data.
plt.figure(figsize=(8,5))

plt.scatter(sales_df['Item_MRP'], sales_df['Item_Outlet_Sales'])
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.title('MRP vs Sales')
plt.show()

# Visualization 4: Box Plot
# Box plot of Item_Outlet_Sales by Outlet_Type
# This box plot allows us to compare the distribution of sales across different outlet types.
plt.figure(figsize=(10,5))

sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=sales_df)
plt.xticks(rotation=45)
plt.title('Sales Distribution Across Outlet Types')
plt.tight_layout()
plt.show()


# step 7: Separate features and target variable (X and y)
X = sales_df.drop('Item_Outlet_Sales', axis=1)  
y = sales_df['Item_Outlet_Sales']

# Step 8: Train-Test Split: Splitting the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% of the data is used for testing and 80% for training
    random_state=42 #Ensures reproducibility of results
)

# Step 9: Data Preprocessing and Feature Engineering
# 9.1 Feature Engineering: Created a new feature called 'Outlet_Age' by calculating the age of the outlet.
def create_outlet_age(df):
    df = df.copy()
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df.drop('Outlet_Establishment_Year', axis=1, inplace=True)
    return df

 # FunctionTransformer allows us to apply the create_outlet_age function as a step in our machine learning pipeline. 
 # The validate=False argument indicates that we are not validating the input data, which is useful,
 # when we are applying custom transformations that may not fit the standard input format expected by scikit-learn.
feature_engineering = FunctionTransformer(create_outlet_age, validate=False)

# 9.2 Identify numerical and categorical columns after feature engineering
X_train_engineered = create_outlet_age(X_train)
numerical_cols = X_train_engineered.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_engineered.select_dtypes(include=['object']).columns

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# 9.3 Define transformers for numerical and categorical features

# Missing numerical values are replaced using Median.
# StandardScaler standardizes the numerical features.

# Median imputation was used for numerical features because the dataset contains skewed distributions and 
# potential outliers. Median is more robust to extreme values compared to mean imputation.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
    ('scaler', StandardScaler())                    # Scale features 
])

# For categorical features, missing values are imputed using the most frequent value (mode),
#  and then one-hot encoding is applied to convert categorical variables into machine-readable numeric format.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical features
])

# 9.4 Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer( #Applies the appropriate transformations to numerical and categorical columns
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# Step 10: Build Linear Regression Model

# 10.1 Create Linear Regression Pipeline
linear_model = Pipeline(steps=[
    ('feature_engineering', feature_engineering), #Create new features before preprocessing
    ('preprocessor', preprocessor), #Preprocess the data using the defined preprocessor
    ('regressor', LinearRegression()) #Fit a linear regression model to the preprocessed data
])

# 10.2 Train Linear Regression Model
linear_model.fit(X_train, y_train) #Fit the linear regression model to the training data (X_train and y_train). 
# This step involves learning the coefficients of the linear regression equation that best fits the training data.

# Step 11: Evaluate Linear Regression Model
y_pred_linear = linear_model.predict(X_test)

# Step 12: Calculate Evaluation Metrics for Linear Regression

# 12.1 Mean Absolute Error (MAE)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print(f"Linear Regression MAE: {mae_linear:.2f}") # 2f formats the output to 2 decimal places

# 12.2 Root Mean Squared Error (RMSE)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear:.2f}") 

# 12.3 R-Squared (Coefficient of Determination)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression R²: {r2_linear:.2f}") 

# Step 13: Build Stochastic Gradient Descent (SGD) Regressor Model
sgd_model = Pipeline(steps=[
    ('feature_engineering', feature_engineering),
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(
        max_iter=1000, # Maximum number of iterations for the optimization algorithm to converge.
        tol=1e-3, # Tolerance for the optimization algorithm. The algorithm will stop when the improvement in the loss function is less than this value.
        penalty='l2', # Regularization term to prevent overfitting. 'l2' adds a penalty equal to the square of the magnitude of coefficients.
        alpha=0.0001, # Regularization strength. Higher values specify stronger regularization.
        random_state=42 # Ensures reproducibility of results by setting a fixed random seed for the algorithm's internal random number generator.
    ))
])

# Step 14: Train SGD Regressor Model
sgd_model.fit(X_train, y_train)

# Step 15: SGD Regressor Predictions
y_pred_sgd = sgd_model.predict(X_test)

# Step 16: Evaluate SGD Regressor Model

# 16.1 Mean Absolute Error (MAE)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
print(f"SGD Regressor MAE: {mae_sgd:.2f}")

# 16.2 Root Mean Squared Error (RMSE)
rmse_sgd = np.sqrt(mean_squared_error(y_test, y_pred_sgd))
print(f"SGD Regressor RMSE: {rmse_sgd:.2f}")

# 16.3 R-Squared (Coefficient of Determination)
r2_sgd = r2_score(y_test, y_pred_sgd)
print(f"SGD Regressor R²: {r2_sgd:.2f}")

# Step 17: Model Comparison
print("\nModel Comparison:")
comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Linear Regression': [
        round(mae_linear, 2),
        round(rmse_linear, 2),
        round(r2_linear, 2)
    ],
    'SGDRegressor': [
        round(mae_sgd, 2),
        round(rmse_sgd, 2),
        round(r2_sgd, 2)
    ],
    'Better': ['SGD', 'SGD', 'SGD']
})

print(comparison_df)

# Step 18: Actual vs Predicted Plot for Linear Regression
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_linear, alpha=0.5) # alpha=0.5 makes the points semi-transparent to better visualize overlapping points
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # This line adds a red dashed line representing the ideal fit where predicted values equal actual values.

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# Step 19: Actual vs Predicted Plot for SGD Regressor
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred_sgd)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # This line adds a red dashed line representing the ideal fit where predicted values equal actual values.

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('SGDRegressor: Actual vs Predicted Sales')

plt.show()
