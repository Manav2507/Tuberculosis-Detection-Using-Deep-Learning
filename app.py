import streamlit as st
import tensorflow as tf
from tensorflow.keras import models # type: ignore
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="TB Detection from Chest X-rays", layout="centered")

IMAGE_SIZE = 256
with open("class_names.json", "r") as f:
    class_names = json.load(f)  

@st.cache_resource
def load_model():
    model = models.load_model("tb_classifier_model.keras") 
    return model

model = load_model()

# Preprocess uploaded image
def preprocess_image(img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Predict function
def predict(model, img):
    predictions = model.predict(img)
    prob = float(predictions[0][0])
    predicted_class = class_names[int(np.round(prob))]
    confidence = round(prob * 100, 2) if predicted_class == class_names[1] else round((1 - prob) * 100, 2)
    return predicted_class, confidence

# Streamlit UI
st.title("🫁 Tuberculosis Detection from Chest X-ray Images")
st.markdown("Upload a chest X-ray image to detect whether the lungs show signs of **Tuberculosis (TB)** or are **Normal**.")

uploaded_file = st.file_uploader("📁 Upload a Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Predicting..."):
        processed_image = preprocess_image(image)
        pred_class, confidence = predict(model, processed_image)

    st.success(f"🩺 **Prediction:** {pred_class}")
    st.info(f"📊 **Confidence:** {confidence}%")