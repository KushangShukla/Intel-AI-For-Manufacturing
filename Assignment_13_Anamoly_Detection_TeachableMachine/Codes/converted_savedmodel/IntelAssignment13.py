#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os

# --- Set working directory dynamically ---
st.sidebar.markdown("### 🔧 Path Settings")
path_input = st.sidebar.text_input("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Intel AI Course\\Codes\\Week 13\\converted_savedmodel:", os.getcwd())

if path_input:
    try:
        os.chdir(path_input)
        current_path = os.getcwd()
        st.sidebar.success(f"Changed working dir to: {current_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to change directory: {e}")
        st.stop()
else:
    current_path = os.getcwd()

# --- Load model using TFSMLayer ---
@st.cache_resource
def load_model(model_path):
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
    ])
    return model

@st.cache_data
def load_labels(labels_file):
    with open(labels_file, "r") as f:
        return [line.strip() for line in f.readlines()]

model_path = current_path  # because saved_model.pb is directly in the working directory
labels_file = os.path.join(current_path, "labels")
 # adjust to "labels" if needed

# Load model and class names
try:
    model = load_model(model_path)
    class_names = load_labels(labels_file)
except Exception as e:
    st.error(f"❌ Error loading model or labels: {e}")
    st.stop()

# --- Preprocessing ---
def preprocess_pil(image):
    image = image.resize((224, 224))
    image = np.asarray(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_cv2(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Streamlit App UI ---
st.set_page_config(page_title="InspectorsAlly - Anomaly Detector", layout="centered")
st.title("🔍 InspectorsAlly")
st.subheader("Anomaly Detection from Uploads or Live Camera")

# --- Select Mode ---
mode = st.sidebar.radio("Choose Mode", ["📁 Upload Image", "🎥 Live Camera Detection"])

# --- Mode: Upload Image ---
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("🔎 Analyzing...")
        processed = preprocess_pil(image)
        prediction = model(processed)[0].numpy()
        class_id = np.argmax(prediction)
        confidence = prediction[class_id]

        st.success(f"🧠 Prediction: **{class_names[class_id]}**")
        st.info(f"📊 Confidence: {confidence:.2%}")

# --- Mode: Real-time Camera ---
elif mode == "🎥 Live Camera Detection":
    run_camera = st.checkbox("Start Camera")

    if run_camera:
        frame_placeholder = st.image([])

        cap = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not read from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = preprocess_cv2(frame_rgb)
            prediction = model(processed)[0].numpy()
            class_id = np.argmax(prediction)
            confidence = prediction[class_id]

            label = f"{class_names[class_id]} ({confidence:.2%})"
            frame_annotated = cv2.putText(
                frame_rgb.copy(), label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

            frame_placeholder.image(frame_annotated)

        cap.release()


# In[ ]:




