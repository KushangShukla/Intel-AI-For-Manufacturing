# Assignment 13 – Anomaly Detection Using Teachable Machine + Streamlit Deployment

This assignment focuses on building a real-time **Anomaly Detection** model using **Google’s Teachable Machine** and deploying it as a web app using **Streamlit**. The goal is to detect defective or abnormal items from a manufacturing process visually.

---

## 🧠 Objective

To develop a machine learning model using image classification for anomaly detection and deploy it with a user-friendly interface that mimics a quality control system in smart manufacturing.

---

## 🔍 Project Workflow

### 1. **Dataset Preparation**
- Select a manufacturing product different from prior projects (e.g., circuit boards, gear shafts, etc.)
- Capture and label images of:
  - Normal items (class: `OK`)
  - Defective/abnormal items (class: `Defective`)

### 2. **Model Training – Teachable Machine**
- Train using Google Teachable Machine’s Image Project module
- Use webcam/image upload features to improve diversity
- Export the trained model as `.bin`, `.json`, and `.labels.txt` for deployment

### 3. **Streamlit Integration**
- Load Teachable Machine model in a Python app
- Create UI for image upload (or webcam feed)
- Classify uploaded images in real-time
- Display predicted label and confidence score

---

## 🧰 Tools & Technologies Used

- **Teachable Machine (Google)** – No-code model training platform
- **Streamlit** – App framework for deployment
- **TensorFlow.js / TensorFlow Lite** – (Optional) for browser/mobile use
- **Python** – For inference logic and UI

---

## 🖥️ Streamlit App Features

- File uploader for image classification
- Confidence-level display
- Live feedback with re-training option
- Interface designed similar to `InspectorsAlly`

---

## 📂 File Structure
Assignment_13_Anomaly_Detection_TeachableMachine/
├── README.md
├── Assignment_13.pdf
├── streamlit_app/
│ ├── app.py
│ ├── model/
│ │ ├── model.json
│ │ ├── weights.bin
│ │ └── labels.txt
└── images/
└── sample_input/

---

## 📄 Type

**Practical Assignment** – Model development + deployment

---

## 🧑‍💼 Author

**Name:** Kushang Akshay Shukla  
**Enrollment No:** 221130107024  
**College:** SAL College of Engineering  
**Branch:** Computer Engineering (6th Sem)  
**Faculty:** Mikin Dagli Sir

---

## 📌 Note

This assignment showcases the power of low-code AI tools and rapid deployment frameworks in building scalable industrial applications like visual inspection and automated anomaly detection.

