from google.colab import drive
drive.mount('/content/drive')

# %%

import zipfile
import os

zip_path = '/content/archive (1).zip'
extract_path = '/content/extracted_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the extracted files
extracted_files = []
for root, dirs, files in os.walk(extract_path):
    for file in files:
        extracted_files.append(os.path.join(root, file))

print("Extracted files:")
for file in extracted_files:
    print(file)

# %%

import pandas as pd

# Load the dataset
file_path = '/content/extracted_data/starbucks.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

# Display preview
print("\nFirst 5 rows:")
df.head()

# %%

import numpy as np

# 1. Clean column names (strip whitespace)
df.columns = df.columns.str.strip()
print("Cleaned Column Names:", list(df.columns))

# 2. Clean percentage columns
pct_cols = ['Vitamin A (% DV)', 'Vitamin C (% DV)', 'Calcium (% DV)', 'Iron (% DV)']
for col in pct_cols:
    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Clean numeric columns stored as objects
# Total Fat (g)
df['Total Fat (g)'] = pd.to_numeric(df['Total Fat (g)'], errors='coerce')

# Caffeine (mg)
# Inspect unique values in Caffeine first to understand variations
print("\nUnique Caffeine values before cleaning:", df['Caffeine (mg)'].unique()[:10])
df['Caffeine (mg)'] = pd.to_numeric(df['Caffeine (mg)'], errors='coerce')

# 4. Check for any missing values after conversion
print("\nMissing values count after numeric conversion:")
print(df.isnull().sum())

# Fill NaN values in numeric columns with the median
for col in ['Total Fat (g)', 'Caffeine (mg)'] + pct_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' with median: {median_val}")

print("\nPost-Cleaning Dataset Info:")
df.info()

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Describe numerical data
df.describe()

# %%

plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Beverage_category', order=df['Beverage_category'].value_counts().index, palette='viridis')
plt.title('Distribution of Beverages by Category')
plt.xlabel('Count')
plt.ylabel('Beverage Category')
plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix of Starbucks Beverage Nutrients')
plt.tight_layout()
plt.show()

# %%

# 1. Distribution of Calories
plt.figure(figsize=(10, 4))
sns.histplot(df['Calories'], kde=True, color='forestgreen', bins=20)
plt.title('Distribution of Calories in Starbucks Beverages')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()

# 2. Average Calories by Category
plt.figure(figsize=(10, 5))
avg_calories_cat = df.groupby('Beverage_category')['Calories'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=avg_calories_cat, x='Calories', y='Beverage_category', palette='magma')
plt.title('Average Calories by Beverage Category')
plt.xlabel('Average Calories')
plt.ylabel('Beverage Category')
plt.tight_layout()
plt.show()

# %%

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Define features (X) and target (y)
# We'll use numerical nutritional features to predict Calories
feature_cols = [
    'Total Fat (g)', 'Trans Fat (g)', 'Saturated Fat (g)', 'Sodium (mg)',
    'Total Carbohydrates (g)', 'Cholesterol (mg)', 'Dietary Fibre (g)',
    'Sugars (g)', 'Protein (g)', 'Vitamin A (% DV)', 'Vitamin C (% DV)',
    'Calcium (% DV)', 'Iron (% DV)', 'Caffeine (mg)'
]

X = df[feature_cols]
y = df['Calories']

# Perform 80% training and 20% testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Initialize and train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Calories")
print(f"R-squared (R2) Score: {r2:.4f}")

# %%

importances = dt_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='crest')
plt.title('Feature Importances in Decision Tree Calorie Predictor')
plt.xlabel('Importance Score')
plt.ylabel('Nutrient Feature')
plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(14, 6))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='forestgreen', alpha=0.7, s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Calories (Test Set)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')

# Plot 2: Residuals Plot
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred, y=residuals, color='crimson', alpha=0.7, s=80)
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.title('Residuals Plot')
plt.xlabel('Predicted Calories')
plt.ylabel('Residuals (Actual - Predicted)')

plt.tight_layout()
plt.show()

# %%

category_counts = df['Beverage_category'].value_counts()

# Create a pie chart with percentage labels and an attractive color palette
plt.figure(figsize=(10, 8))
plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel', len(category_counts)),
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
plt.title('Proportion of Beverage Categories in Starbucks Menu', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%

# Select a sample of 10 indices from the test set
sample_indices = X_test.sample(10, random_state=42).index

# Extract names and original features
sample_drinks = df.loc[sample_indices, ['Beverage_category', 'Beverage', 'Beverage_prep']]
sample_features = X_test.loc[sample_indices]

# Make predictions
sample_predictions = dt_model.predict(sample_features)

# Construct comparison DataFrame
comparison_df = sample_drinks.copy()
comparison_df['Actual Calories'] = y_test.loc[sample_indices]
comparison_df['Predicted Calories'] = sample_predictions.round(1)
comparison_df['Difference (Error)'] = (comparison_df['Actual Calories'] - comparison_df['Predicted Calories']).round(1)

# Display the results
display(comparison_df)

# %%

import pickle

# Define the file path where the model will be saved
model_filename = '/content/starbucks_calorie_predictor.pkl'

# Save the model to disk
with open(model_filename, 'wb') as file:
    pickle.dump(dt_model, file)

print(f"Success! The trained model has been saved to: {model_filename}")
