# 🚗 Car Price Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the selling price of used cars using Machine Learning techniques. The objective is to analyze various car-related features such as present price, fuel type, transmission type, kilometers driven, ownership history, and car age to accurately estimate a vehicle's market value.

A Random Forest Regressor model was developed to learn patterns from historical car sales data and generate price predictions with high accuracy.

---

## 🎯 Objectives

* Collect and analyze car-related features.
* Perform data cleaning and preprocessing.
* Apply feature engineering techniques.
* Train a regression model for price prediction.
* Evaluate model performance using regression metrics.
* Understand real-world applications of machine learning in automobile pricing.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Dataset Features

The dataset includes:

* Car Name
* Year of Manufacture
* Present Price
* Selling Price
* Kilometers Driven
* Fuel Type
* Selling Type
* Transmission
* Owner Count

Additional engineered feature:

* Car Age

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Missing value analysis
* Duplicate record removal
* Feature engineering (Car Age calculation)
* Label Encoding for categorical variables
* Correlation analysis
* Feature selection

---

## 🤖 Machine Learning Model

### Algorithm Used

**Random Forest Regressor**

Reasons for selection:

* Handles non-linear relationships effectively
* Robust against overfitting
* Works well with mixed feature types
* Provides feature importance scores

---

## 📊 Visualizations

### 1. Selling Price Distribution

Analyzed the distribution of car selling prices to understand market trends and detect skewness.

### 2. Correlation Matrix

Explored relationships among numerical variables to identify influential factors affecting selling price.

### 3. Actual vs Predicted Price

Compared actual car prices with model predictions to evaluate prediction accuracy.

### 4. Feature Importance Analysis

Identified the most influential variables contributing to car price prediction.

---

## 🔍 Key Findings

### Correlation Analysis

* Present Price has a strong positive correlation with Selling Price.
* Car Age negatively impacts resale value.
* Driven Kilometers show moderate influence on vehicle pricing.

### Feature Importance

The model identified the following important features:

1. Present Price
2. Car Age
3. Driven Kilometers
4. Transmission Type
5. Fuel Type

Present Price was found to be the dominant factor affecting resale value.

### Prediction Performance

The Actual vs Predicted Price plot demonstrates that the model successfully captures overall pricing trends and provides reliable predictions for most vehicles.

---

## 📈 Model Evaluation Metrics

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics help assess prediction accuracy and model reliability.

---

## 💡 Business Insights

* Newer vehicles generally retain higher resale value.
* Cars with lower mileage tend to command better prices.
* Present market value strongly influences selling price.
* Transmission and fuel type contribute to pricing differences.
* Machine Learning can support dealerships and customers in making informed pricing decisions.

---

## 🚀 Real-World Applications

This project demonstrates how machine learning can be applied in:

* Used Car Marketplaces
* Automobile Dealerships
* Vehicle Valuation Systems
* Insurance Price Assessment
* Automotive Business Intelligence

---

## 📁 Project Structure

```text
Car-Price-Prediction-ML/
│
├── car data.csv
├── car_price_prediction.py
├── correlation matrix.png
├── selling price distribution.png
├── actual vs predicted car price.png
├── feature importance.png
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the project:

```bash
python car_price_prediction.py
```

---

## 👨‍💻 Author

Lokeshvar K

Data Analyst | Python Developer | Machine Learning Enthusiast

LinkedIn:
https://www.linkedin.com/in/lokeshvar-k-290503l
