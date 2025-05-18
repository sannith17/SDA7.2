import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib import cm

st.set_page_config(layout="wide")

# --- Load Models with Caching ---
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
    image_np = np.array(image)
    return image_np

# --- K-Means Clustering ---
def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image_np.shape, image_np.shape)

# --- CNN Prediction ---
def predict_cnn(image_np):
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array)
    return np.argmax(predictions, axis=-1)

# --- Difference Map ---
def difference_heatmap(before_mask, after_mask):
    diff = after_mask != before_mask
    return diff.astype(np.uint8) * 255

# --- Updated Calamity Detection ---
def detect_calamity(date1, date2, mask1, mask2):
    duration = (date2 - date1).days
    diff_mask = mask2 != mask1
    change_percentage = np.sum(diff_mask) / diff_mask.size
    
    calamity_info = {
        "water_changes": np.sum((mask2 == 0) & (mask1 != 0)),
        "land_changes": np.sum((mask2 == 1) & (mask1 != 1)),
        "vegetation_changes": np.sum((mask2 == 2) & (mask1 != 2))
    }
    
    event_type = ""
    if duration <= 7:
        event_type = "Floods (Flash floods, River floods)"
    elif duration <= 30:
        event_type = "Waterlogging/Prolonged River Floods"
    elif duration <= 365:
        event_type = "Seasonal Variability/Drought"
    else:
        event_type = "Climate Change/Urbanization Impact"
    
    return {
        "event_type": event_type,
        "change_percentage": change_percentage,
        "calamity_info": calamity_info
    }

# --- Color Mapping ---
class_colors = {
    0: [0, 0, 0.5],    # Blue (Water)
    1: [0.5, 0, 0],    # Red (Land)
    2: [0, 0.5, 0],    # Green (Vegetation)
    3: [0.5, 0.5, 0.5] # Grey (Urban)
}

# --- Layout Pages ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Image Calamity Detector")
    st.file_uploader("Upload Before Image", type=["png", "jpg"], key="before")
    st.file_uploader("Upload After Image", type=["png", "jpg"], key="after")
    st.date_input("Before Date", key="date1")
    st.date_input("After Date", key="date2")
    
    if st.button("Analyze"):
        if st.session_state.before and st.session_state.after:
            next_page()

if st.session_state.page == 2:
    st.title("Analysis Results")
    
    before_img = load_and_preprocess(st.session_state.before)
    after_img = load_and_preprocess(st.session_state.after)
    
    # CNN Predictions
    b_mask_cnn = predict_cnn(before_img)
    a_mask_cnn = predict_cnn(after_img)
    
    # K-Means with consistent colors
    kmeans_before = kmeans_segmentation(before_img)
    kmeans_after = kmeans_segmentation(after_img)
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Before Images
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(before_img)
    ax1.set_title("Before Image")
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(kmeans_before, cmap=ListedColormap(['black', 'blue', 'red', 'green']))
    ax2.set_title("K-Means Before")
    
    # After Images
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(after_img)
    ax3.set_title("After Image")
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(kmeans_after, cmap=ListedColormap(['black', 'blue', 'red', 'green']))
    ax4.set_title("K-Means After")
    
    # CNN Results
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(b_mask_cnn, cmap=ListedColormap([tuple(class_colors[i]) for i in range(4)]))
    ax5.set_title("CNN Before")
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(a_mask_cnn, cmap=ListedColormap([tuple(class_colors[i]) for i in range(4)]))
    ax6.set_title("CNN After")
    
    st.pyplot(fig)
    
    # Pie Charts
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 6))
    
    before_counts = np.bincount(b_mask_cnn.flatten())
    ax7.pie(before_counts, labels=['Water', 'Land', 'Vegetation', 'Urban'], 
            colors=[tuple(class_colors[i]) for i in range(4)], autopct='%1.1f%%')
    ax7.set_title("Before Distribution")
    
    after_counts = np.bincount(a_mask_cnn.flatten())
    ax8.pie(after_counts, labels=['Water', 'Land', 'Vegetation', 'Urban'], 
            colors=[tuple(class_colors[i]) for i in range(4)], autopct='%1.1f%%')
    ax8.set_title("After Distribution")
    
    st.pyplot(fig2)
    
    # Calamity Detection
    analysis = detect_calamity(st.session_state.date1, st.session_state.date2, b_mask_cnn, a_mask_cnn)
    
    # Display Results
    st.subheader("Event Analysis")
    st.write(f"**Detected Event Type:** {analysis['event_type']}")
    st.write(f"**Change Percentage:** {analysis['change_percentage']:.2%}")
    
    # Changes Table
    changes_df = pd.DataFrame({
        "Feature": ["Water", "Land", "Vegetation"],
        "Change Area (pixels)": [
            analysis['calamity_info']['water_changes'],
            analysis['calamity_info']['land_changes'],
            analysis['calamity_info']['vegetation_changes']
        ]
    })
    st.dataframe(changes_df.style.highlight_max(axis=0))
    
    if st.button("Reset"):
        reset()
