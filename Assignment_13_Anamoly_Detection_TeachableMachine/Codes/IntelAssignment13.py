#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Set the working directory
model_dir = r"C:\Users\kusha\OneDrive\Desktop\Kushang's Files\Intel AI Course\Codes\Week 13\Data"
os.chdir(model_dir)

# Load labels
@st.cache_resource
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Load saved_model using TFSMLayer for inference-only model
@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')

model_layer = load_model()
class_names = load_labels()

st.title("📦 Anomaly Detection App - Upload or Live Camera")
st.markdown("This app detects anomalies using a model exported from Teachable Machine (TensorFlow SavedModel format).")

# Image preprocessing function
def preprocess(image):
    image = image.resize((224, 224))  # Teachable Machine typically uses 224x224
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Prediction
def predict(image):
    processed = preprocess(image)

    prediction_dict = model_layer(processed)
    # Extract the tensor from the dict using the correct key
    prediction = list(prediction_dict.values())[0].numpy()

    pred_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return pred_class, confidence

    print(prediction_dict.keys())

# Upload image section
st.header("🖼️ Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    pred_class, confidence = predict(image)
    st.success(f"**Prediction:** {pred_class} ({confidence:.2f}%)")

# Live webcam section
st.header("📷 Live Camera Feed")
run_camera = st.checkbox("Start Camera")

if run_camera:
    camera = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])
    while run_camera:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img)
        pred_class, confidence = predict(image_pil)

        img = cv2.putText(img, f"{pred_class} ({confidence:.1f}%)", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(img)
else:
    st.info("Tick 'Start Camera' to enable real-time detection.")


# In[ ]:




