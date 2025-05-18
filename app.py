# app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap

# Set full-width layout
st.set_page_config(layout="wide")

# =========================
# Load Models with Caching
# =========================
@st.cache_resource
def load_models():
    cnn_model, rf_model = None, None
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        st.success("✅ CNN model loaded!")
    except Exception as e:
        st.warning(f"⚠️ CNN model not loaded: {e}")

    try:
        rf_model = joblib.load("rf_model.pkl")
        st.success("✅ RF model loaded!")
    except Exception as e:
        st.warning(f"⚠️ RF model not loaded: {e}")

    return cnn_model, rf_model

cnn_model, rf_model = load_models()

# ======================
# Utility Functions
# ======================
def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    return np.array(img)

def predict_rf(image_np):
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    prediction = rf_model.predict(pixels)
    return prediction.reshape(128, 128)

def predict_cnn(image_np):
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array)
    return np.argmax(predictions, axis=-1)[0]

def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image_np.shape[0], image_np.shape[1])

def difference_map(mask1, mask2):
    return (mask1 != mask2).astype(np.uint8)

# ======================
# UI Setup
# ======================
st.title("🌍 Satellite Change Detection App")
st.sidebar.header("Upload Images")

before_file = st.sidebar.file_uploader("Upload BEFORE image", type=["png", "jpg", "jpeg"], key="before")
after_file = st.sidebar.file_uploader("Upload AFTER image", type=["png", "jpg", "jpeg"], key="after")

if before_file and after_file:
    before_img = load_image(before_file)
    after_img = load_image(after_file)

    st.subheader("1. Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(before_img, caption="Before Image", use_container_width=True)
    with col2:
        st.image(after_img, caption="After Image", use_container_width=True)

    if cnn_model and rf_model:
        st.subheader("2. Running Analysis...")

        # Run Predictions
        before_rf = predict_rf(before_img)
        after_rf = predict_rf(after_img)

        before_cnn = predict_cnn(before_img)
        after_cnn = predict_cnn(after_img)

        before_kmeans = kmeans_segmentation(before_img)
        after_kmeans = kmeans_segmentation(after_img)

        # Create Diff Heatmap
        diff_rf = difference_map(before_rf, after_rf)
        changed_pixels = np.sum(diff_rf)
        total_pixels = diff_rf.size
        change_percent = (changed_pixels / total_pixels) * 100

        # Display Results
        st.subheader("3. Random Forest Results")
        cmap = ListedColormap(['blue', 'green', 'orange', 'gray'])  # example colors
        col3, col4 = st.columns(2)
        with col3:
            st.image(before_rf, caption="RF Before", clamp=True)
        with col4:
            st.image(after_rf, caption="RF After", clamp=True)

        st.subheader("4. Change Detection Heatmap")
        st.metric("Total Area Changed", f"{change_percent:.2f}%", f"{changed_pixels} pixels changed")
        st.image(diff_rf * 255, caption="Change Map (Red = Change)", clamp=True)

        # Optional: Class Distribution
        st.subheader("5. Class Distribution")
        def plot_distribution(mask, title):
            values, counts = np.unique(mask, return_counts=True)
            labels = [f"Class {v}" for v in values]
            plt.figure(figsize=(4, 4))
            plt.pie(counts, labels=labels, autopct='%1.1f%%')
            st.pyplot(plt)

        col5, col6 = st.columns(2)
        with col5:
            plot_distribution(before_rf, "Before")
        with col6:
            plot_distribution(after_rf, "After")

        # Final Status
        if change_percent > 15:
            st.error("🚨 Significant Change Detected")
        elif change_percent > 5:
            st.warning("⚠️ Moderate Change Detected")
        else:
            st.success("✅ No Major Change Detected")

    else:
        st.error("❌ Models not available. Please ensure 'cnn_model.h5' and 'rf_model.pkl' are in the project folder.")
else:
    st.info("Please upload both before and after satellite images to continue.")
