# Assignment 05 – Evaluation Metrics & Confusion Matrix

This assignment explores the most widely used evaluation metrics for regression and classification models. It also includes an explanation and example of the confusion matrix along with derived metrics like precision, recall, and F1 score.

---

## 🧠 Objective

To develop a clear understanding of how to evaluate machine learning models using both regression and classification metrics, and to demonstrate the interpretation of a confusion matrix using a sample prediction table.

---

## 📊 Covered Topics

### 📌 Regression Metrics
- **MAE (Mean Absolute Error):** Average magnitude of errors.
- **MSE (Mean Squared Error):** Penalizes large errors more heavily.
- **RMSE (Root Mean Squared Error):** Square root of MSE for interpretation in original units.
- **R² Score (Coefficient of Determination):** Measures variance explained by the model.
- **MAPE (Mean Absolute Percentage Error):** Error as a percentage, useful in business forecasting.

### 📌 When to Use What?
- Use **MAE** when outliers are not significant.
- Use **MSE** or **RMSE** when penalizing large errors is critical.
- Use **R² Score** to assess overall model fit.
- Use **MAPE** when percentage-based accuracy is preferred (e.g., retail or demand forecasting).

---

## 🧮 Confusion Matrix – Classification Performance

- **Definition:** A summary table showing correct vs. incorrect predictions (TP, FP, TN, FN).
- **Scenario:** Custom example of AI-based product categorization (used in assignment).
- **Derived Metrics:**
  - **Precision:** Accuracy of positive predictions.
  - **Recall:** Model’s ability to find all positives.
  - **F1 Score:** Harmonic mean of precision and recall.

### 🧾 Sample Output Table

| Actual | Predicted | Outcome |
|--------|-----------|---------|
| 1      | 1         | TP      |
| 0      | 1         | FP      |
| 0      | 0         | TN      |
| 1      | 0         | FN      |

Metrics were calculated from this fictional dataset and interpreted accordingly.

---

## ✅ Deliverables

- Detailed write-up on 5 regression metrics  
- Example-based explanation of confusion matrix  
- Sample prediction table and metric calculations

---

## 📂 File Structure
Assignment_05_Evaluation_Metrics_and_Confusion_Matrix/
├── README.md
└── Assignment_05.pdf

---

## 📄 Type

**Theoretical Assignment** – Focused on conceptual understanding and illustrative evaluation.

---

## 🧑‍💼 Author

**Name:** Kushang Akshay Shukla  
**Enrollment No:** 221130107024  
**College:** SAL College of Engineering  
**Branch:** Computer Engineering (6th Sem)  
**Faculty:** Mikin Dagli Sir

---

## 📌 Note

Evaluation metrics are critical for understanding model performance, guiding model selection, and presenting results to stakeholders in a meaningful way.
