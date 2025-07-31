# Assignment 06 – Streamlit Deployment: Timelytics – Order Delivery Time Prediction

This assignment demonstrates how to deploy a machine learning model using **Streamlit** to predict the delivery time of an order based on input parameters such as product category, customer location, and shipping method.

---

## 🧠 Objective

To build an interactive web application that allows end-users to predict the delivery time for a new order in real-time, using a trained ML model and a simple Streamlit-based interface.

---

## 🖥️ Application Overview

**Timelytics** is a lightweight web app where users can input order details and receive a delivery time prediction instantly. The app leverages a trained machine learning model and integrates seamlessly with Pandas, Scikit-learn, and Streamlit components.

---

## 🚀 Features

- 📥 **Input fields** for product category, shipping method, customer location, etc.
- ⚙️ **Model integration** for real-time prediction
- 📈 **Visual feedback** of delivery estimates
- 🧠 **Backend powered by XGBoost / RandomForest (based on best model from Assignment 4)**

---

## 🧪 Model Integration

- Pre-trained model saved using `joblib` or `pickle`
- Custom preprocessing pipeline (scaling, encoding) embedded into the prediction flow
- Deployment-ready script in Streamlit with form input and output logic

---

## 🧰 Technologies Used

- **Python**  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Streamlit  
- **Deployment:** Localhost / Streamlit Cloud (optional)

---

## 📂 File Structure
Assignment_06_Timelytics_Streamlit_Deployment/
├── README.md
├── Assignment_06.pdf
├── app/
│ ├── app.py # Streamlit app script
│ ├── model.pkl # Trained model file
│ └── preprocessing_utils.py # (Optional) Encoding/scaling functions
└── data/
└── sample_input.csv

---

## ▶️ How to Run the App

```bash
cd Assignment_06_Timelytics_Streamlit_Deployment/app
streamlit run app.py

📄 Type
Practical Assignment – With full frontend and backend integration

🧑‍💼 Author
Name: Kushang Akshay Shukla
Enrollment No: 221130107024
College: SAL College of Engineering
Branch: Computer Engineering (6th Sem)
Faculty: Mikin Dagli Sir

📌 Note
This assignment showcases how simple tools like Streamlit can bring powerful machine learning models into user-friendly, deployable web applications for real-time business use.
