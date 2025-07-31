# Assignment 04 – Product Delivery Time Prediction (ML Modeling + EDA)

This assignment focuses on building a machine learning pipeline to predict delivery time for manufactured products using structured data. The task includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 🧠 Objective

To analyze a manufacturing dataset containing product specifications and order completion times, and create a modeling-ready dataset that predicts delivery time for new products.

---

## 📦 Dataset Overview

- Source: Manufacturing unit product order data
- Fields: Product category, specs, time-to-complete, location, shipping method, etc.
- Total Datasets Used: 9 interconnected CSVs (based on Olist ecommerce data schema)

---

## 🔍 Key Steps

### 1. **Exploratory Data Analysis**
- Cleaned missing values, removed outliers
- Visualized data distributions using Matplotlib and Seaborn
- Examined feature correlations

### 2. **Feature Engineering**
- Categorical encoding (One-Hot, Label Encoding)
- Created derived time-related and product grouping features
- Feature scaling for numerical fields

### 3. **Modeling**
- Applied various regression algorithms:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (Best performer)
- Evaluation Metrics:
  - MAE, MSE, RMSE, R² Score

### 4. **Model Insights**
- Feature importance plotted using SHAP and model APIs
- Business insights derived from delay drivers

---

## 🧪 Results

- **Best RMSE achieved:** ~0.87  
- **Most influential features:** Product weight, product category, shipping method  
- **Business Impact:** Improves OTD (Order-to-Delivery) planning and resource management

---

## 🚀 Technologies Used

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

---

## 📂 File Structure
Assignment_04_Product_Delivery_Time_Prediction/
├── README.md
├── Assignment_04.pdf
├── data/ # Cleaned and raw datasets (optional)
├── notebook.ipynb # Jupyter Notebook with full workflow
└── models/ # Saved trained model (optional)

---

## 📄 Type

**Practical + Theoretical Assignment** – Code, modeling, and documentation included.

---

## 🧑‍💼 Author

**Name:** Kushang Akshay Shukla  
**Enrollment No:** 221130107024  
**College:** SAL College of Engineering  
**Branch:** Computer Engineering (6th Sem)  
**Faculty:** Mikin Dagli Sir

---

## 📌 Note

This assignment simulates a real-world industrial use case for delivery time prediction and lays the foundation for deployment-ready AI systems in supply chain analytics.

