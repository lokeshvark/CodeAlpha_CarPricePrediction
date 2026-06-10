import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# 1. LOAD DATASET

df = pd.read_csv("car data.csv")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print(df.head())
print(df.shape)

# 2. DATA CLEANING

print("\nMissing Values")
print(df.isnull().sum())

# Check duplicate values
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# 3. FEATURE ENGINEERING

# Calculate Car Age

df['Car_Age'] = 2025 - df['Year']

# Drop Year column
df.drop('Year', axis=1, inplace=True)

# Encode Categorical Variables

encoder = LabelEncoder()

df['Fuel_Type'] = encoder.fit_transform(df['Fuel_Type'])
df['Selling_type'] = encoder.fit_transform(df['Selling_type'])
df['Transmission'] = encoder.fit_transform(df['Transmission'])

# 4. EXPLORATORY DATA ANALYSIS

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Matrix")
plt.show()

# Selling Price Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['Selling_Price'],
             kde=True)

plt.title("Selling Price Distribution")
plt.show()

# 5. FEATURE SELECTION

X = df.drop(['Selling_Price', 'Car_Name'], axis=1)
y = df['Selling_Price']

# 6. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. MODEL TRAINING

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 8. PREDICTIONS

y_pred = model.predict(X_test)

# 9. MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R² Score:", round(r2,2))

# 10. ACTUAL VS PREDICTED

plt.figure(figsize=(8,6))

plt.scatter(y_test,
            y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Car Price")

plt.show()

# 11. FEATURE IMPORTANCE

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.show()

# 12. BUSINESS INSIGHTS

print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)

print("""
1. Present Price strongly influences selling price.

2. Newer cars generally have higher resale value.

3. Fuel type and transmission affect market value.

4. Cars with lower mileage tend to sell for higher prices.

5. Machine Learning can accurately estimate car prices,
   helping buyers and sellers make informed decisions.
""")

# 13. CONCLUSION

print("\nProject Completed Successfully!")