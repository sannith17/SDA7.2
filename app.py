import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from datetime import datetime
from matplotlib.colors import ListedColormap

st.set_page_config(layout="wide")

# --- Load Model with Caching ---
@st.cache_resource
def load_model():
    """Loads the CNN model with caching."""
    try:
        model = tf.keras.models.load_model("cnn_model.h5")
        st.info("CNN model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")
        return None

cnn_model = load_model()

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# --- Navigation Functions ---
def next_page():
    st.session_state.page += 1

def reset():
    st.session_state.page = 1

# --- Image Preprocessing ---
def load_and_preprocess(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)

# --- K-Means Clustering ---
def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(pixels).reshape(image_np.shape[0], image_np.shape[1])

# --- CNN Prediction ---
def predict_cnn(image_np):
    """Predicts using the CNN model with proper reshaping."""
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    predictions = cnn_model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
    return np.argmax(predictions, axis=-1).reshape(128, 128)

# --- Difference Map ---
def difference_heatmap(before_mask, after_mask):
    return (after_mask != before_mask).astype(np.uint8) * 255

# --- Calamity Detection ---
def detect_calamity(date1, date2, mask1, mask2, analysis_type):
    date_diff = (date2 - date1).days
    change_pct = np.sum(mask2 != mask1) / mask1.size
    
    if analysis_type == "Water Body":
        if date_diff <= 7: return "⚠️ Flood (Flash/River)"
        elif date_diff <= 30: return "⚠️ Waterlogging/Prolonged Flood"
        elif date_diff <= 90: return "⚠️ Seasonal Variability/Drought"
        else: return "⚠️ Climate Change/Urbanization"
    else:
        if date_diff <= 7: return "⚠️ Landslide/Earthquake"
        elif date_diff <= 30: return "⚠️ Soil Erosion/Mudslide"
        elif date_diff <= 90: return "⚠️ Deforestation/Agriculture"
        else: return "⚠️ Desertification/Urban Expansion"

# --- Layout Pages ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Image Calamity Detector")
    st.session_state.analysis_type = st.radio("Analysis Type:", ["Land Body", "Water Body"])
    st.button("Next", on_click=next_page)

elif st.session_state.page == 2:
    st.title("Step 2: Upload Satellite Images")
    col1, col2 = st.columns(2)
    with col1:
        before_image = st.file_uploader("BEFORE image", type=['jpg', 'jpeg', 'png'], key="before")
        before_date = st.date_input("Before Date")
    with col2:
        after_image = st.file_uploader("AFTER image", type=['jpg', 'jpeg', 'png'], key="after")
        after_date = st.date_input("After Date")

    if before_image and after_image:
        st.session_state.before_image = before_image
        st.session_state.after_image = after_image
        st.session_state.before_date = before_date
        st.session_state.after_date = after_date
        st.button("Next", on_click=next_page)
    else:
        st.warning("Please upload both images")

elif st.session_state.page == 3:
    st.title("Image Preview")
    if 'before_image' in st.session_state:
        b_img = load_and_preprocess(st.session_state.before_image)
        a_img = load_and_preprocess(st.session_state.after_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(b_img, caption=f"Before ({st.session_state.before_date})", use_container_width=True)
        with col2:
            st.image(a_img, caption=f"After ({st.session_state.after_date})", use_container_width=True)
        
        st.session_state.b_np = b_img
        st.session_state.a_np = a_img
        st.button("Next", on_click=next_page)

elif st.session_state.page == 4:
    if 'b_np' not in st.session_state:
        st.warning("Please upload images first")
        st.stop()
        
    b_np, a_np = st.session_state.b_np, st.session_state.a_np
    before_date, after_date = st.session_state.before_date, st.session_state.after_date
    analysis_type = st.session_state.analysis_type
    
    if not cnn_model:
        st.error("Model not loaded")
        st.stop()

    with st.spinner("Analyzing images..."):
        # CNN Predictions
        b_mask = predict_cnn(b_np)
        a_mask = predict_cnn(a_np)
        diff = difference_heatmap(b_mask, a_mask)
        calamity = detect_calamity(before_date, after_date, b_mask, a_mask, analysis_type)
        
        # Class definitions
        class_colors = {
            'Water': '#1f77b4',
            'Vegetation': '#2ca02c', 
            'Urban': '#ff7f0e',
            'Barren': '#d62728',
            'Forest': '#9467bd'
        }
        classes = {
            0: "Water",
            1: "Vegetation", 
            2: "Urban",
            3: "Barren" if analysis_type == "Water Body" else "Forest"
        }

        # Main Display
        st.title("Analysis Results")
        
        # Original Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(b_np, caption="Original Before", use_container_width=True)
        with col2:
            st.image(a_np, caption="Original After", use_container_width=True)

        # CNN Segmentation
        st.subheader("CNN Segmentation")
        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(b_mask, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
            ax.set_title("Before Segmentation")
            ax.axis('off')
            st.pyplot(fig)
        with col4:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(a_mask, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
            ax.set_title("After Segmentation") 
            ax.axis('off')
            st.pyplot(fig)

        # K-Means Change Highlights
        st.subheader("Change Highlights")
        change_mask = np.zeros_like(kmeans_segmentation(a_np))
        for i in range(4):
            change_mask[(a_mask == i) & (b_mask != i)] = i + 1
            
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(change_mask, cmap=ListedColormap(['black', 'blue', 'green', 'red', 'orange']), vmin=0, vmax=4)
        ax.set_title("Detected Changes")
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor='blue', label='Water Increase'),
            Patch(facecolor='red', label='Land Change'), 
            Patch(facecolor='green', label='Vegetation Increase')
        ]
        ax.legend(handles=legend, loc='lower right')
        st.pyplot(fig)

        # Statistics
        st.subheader("Change Statistics")
        total_pixels = b_mask.size
        change_pct = np.sum(diff > 0) / total_pixels * 100
        st.metric("Area Changed", f"{change_pct:.2f}%", f"{np.sum(diff > 0):,} pixels")

        # Class Distribution
        st.subheader("Class Distribution")
        data = []
        for i in range(4):
            before = np.sum(b_mask == i) / total_pixels * 100
            after = np.sum(a_mask == i) / total_pixels * 100
            data.append({
                "Class": classes[i],
                "Before (%)": before,
                "After (%)": after,
                "Change (%)": after - before,
                "Area Change (km²)": (after - before) * 0.01 * 100  # Assuming 100km² area
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df.style.format({
            "Before (%)": "{:.2f}",
            "After (%)": "{:.2f}",
            "Change (%)": "{:+.2f}",
            "Area Change (km²)": "{:+.2f}"
        }).background_gradient(subset=["Change (%)"], cmap='RdYlGn'))

        # Pie Charts
        st.subheader("Distribution Comparison")
        col5, col6 = st.columns(2)
        with col5:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(df["Before (%)"], labels=df["Class"], autopct='%1.1f%%', 
                  colors=[class_colors[classes[i]] for i in range(4)], startangle=90)
            ax.set_title("Before Composition")
            st.pyplot(fig)
        with col6:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(df["After (%)"], labels=df["Class"], autopct='%1.1f%%',
                  colors=[class_colors[classes[i]] for i in range(4)], startangle=90)
            ax.set_title("After Composition")
            st.pyplot(fig)

        # Calamity Assessment
        st.subheader("Calamity Assessment")
        st.warning(calamity)
        
        if "Flood" in calamity:
            st.info("Recommended Actions: Activate flood warnings, prepare shelters")
        elif "Drought" in calamity:
            st.info("Recommended Actions: Water conservation, monitor reservoirs")
        elif "Deforestation" in calamity:
            st.info("Recommended Actions: Forest patrols, anti-logging measures")
        else:
            st.info("Recommended Actions: Continue monitoring, document changes")

    st.button("Restart", on_click=reset, type="primary")
