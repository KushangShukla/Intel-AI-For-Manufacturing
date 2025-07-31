# Assignment 03 – ML Canvas: Predictive Maintenance in Manufacturing

This assignment applies the Machine Learning (ML) Canvas framework to design an AI solution that predicts which machine parts in a manufacturing factory are likely to fail in the near future. The aim is to minimize downtime and optimize maintenance scheduling.

---

## 🧠 Objective

To define a structured machine learning problem using the ML Canvas and propose a real-world predictive maintenance strategy using sensor-based data from industrial machines.

---

## 🧩 Problem Statement

**Identify machine parts that are prone to failure based on real-time operational sensor data such as temperature, pressure, vibration, and run-time cycles.**

---

## 🗂️ ML Canvas Sections

### 1. **Decisions**
- Predict failures before breakdowns occur to enable preventive action.
- Improve maintenance planning and avoid unplanned shutdowns.

### 2. **ML Task**
- Binary classification: Will a part fail soon or not?
- Optional regression: Estimate time until failure.

### 3. **Data Sources**
- IoT sensors (temperature, pressure, vibration)
- Maintenance logs and historical failure data
- Operator feedback

### 4. **Features**
- Raw: Temperature, pressure, vibration, operating hours
- Derived: Rate of change, moving averages, trend indicators
- Contextual: Machine load, environment, shift schedule

### 5. **Model Strategy**
- Train a classification model to flag failure risk in real-time.
- Periodically retrain with the latest operational and maintenance data.

### 6. **Offline Evaluation**
- Precision, Recall, F1 Score, ROC-AUC
- Cross-validation using simulated failure scenarios

### 7. **Monitoring & Feedback**
- Real-time alerts and performance monitoring dashboard
- Maintenance feedback loop for retraining and improving accuracy

---

## ✅ Deliverables

- Problem-to-solution breakdown using the ML Canvas methodology
- Defined data sources, features, and model plan
- Offline and live evaluation strategies

---

## 📂 File Structure

Assignment_03_ML_Canvas_Predictive_Maintenance/
├── README.md # Assignment overview and ML Canvas
└── Assignment_03.pdf # Original report document (optional)

---

## 📄 Type

**Theoretical Assignment** – Problem design and system planning (no coding involved)

---

## 🧑‍💼 Author

**Name:** Kushang Akshay Shukla  
**Enrollment No:** 221130107024  
**College:** SAL College of Engineering  
**Branch:** Computer Engineering (6th Sem)  
**Faculty:** Mikin Dagli Sir

---

## 📌 Note

This assignment demonstrates how structured ML planning tools like the ML Canvas help teams develop clear, actionable, and scalable machine learning solutions in an industrial context.
