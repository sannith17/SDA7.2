import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from datetime import datetime

st.set_page_config(layout="wide")

# --- Color Mapping ---
class_colors = {
    0: [0, 0, 1],    # Blue (Water)
    1: [1, 0, 0],    # Red (Land)
    2: [0, 1, 0],    # Green (Vegetation)
    3: [0.5, 0.5, 0.5] # Grey (Urban)
}

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 1

# --- Image Processing ---
def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(pixels).reshape(image_np.shape[:2])

# --- Analysis Functions ---
def calculate_changes(before_mask, after_mask):
    change_mask = before_mask != after_mask
    return {
        "total_change": np.mean(change_mask),
        "water_change": np.mean((after_mask == 0) & (before_mask != 0)),
        "land_change": np.mean((after_mask == 1) & (before_mask != 1)),
        "vegetation_change": np.mean((after_mask == 2) & (before_mask != 2))
    }

def detect_calamity(date_diff_days, changes):
    if date_diff_days <= 7 and changes['water_change'] > 0.2:
        return "High flood risk detected"
    elif date_diff_days <= 30 and changes['vegetation_change'] > 0.3:
        return "Possible deforestation"
    elif changes['land_change'] > 0.25:
        return "Urban expansion detected"
    return "No immediate disaster detected"

# --- Interface ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Image Analysis")
    col1, col2 = st.columns(2)
    with col1:
        before_img = st.file_uploader("Before Image", type=["png", "jpg"])
        before_date = st.date_input("Before Date")
    with col2:
        after_img = st.file_uploader("After Image", type=["png", "jpg"])
        after_date = st.date_input("After Date")
    
    if st.button("Analyze") and before_img and after_img:
        st.session_state.page = 2

if st.session_state.page == 2:
    st.title("Analysis Results")
    
    # Process images
    before = np.array(Image.open(before_img).convert("RGB"))
    after = np.array(Image.open(after_img).convert("RGB"))
    
    # K-means segmentation
    kmeans_before = kmeans_segmentation(before)
    kmeans_after = kmeans_segmentation(after)
    
    # Visualization
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original images
    ax[0,0].imshow(before)
    ax[0,0].set_title("Before Image")
    ax[0,1].imshow(after)
    ax[0,1].set_title("After Image")
    
    # K-means maps
    ax[1,0].imshow(kmeans_before, cmap=ListedColormap([class_colors[i] for i in range(4)]))
    ax[1,0].set_title("Before Segmentation")
    ax[1,1].imshow(kmeans_after, cmap=ListedColormap([class_colors[i] for i in range(4)]))
    ax[1,1].set_title("After Segmentation")
    
    st.pyplot(fig)
    
    # Change analysis
    changes = calculate_changes(kmeans_before, kmeans_after)
    date_diff = (after_date - before_date).days
    calamity = detect_calamity(date_diff, changes)
    
    # Pie charts
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for mask, title, ax in zip([kmeans_before, kmeans_after], ["Before", "After"], [ax1, ax2]):
        counts = np.bincount(mask.flatten(), minlength=4)
        ax.pie(counts, 
              labels=['Water', 'Land', 'Vegetation', 'Urban'],
              colors=[class_colors[i] for i in range(4)],
              autopct='%1.1f%%')
        ax.set_title(title)
    
    st.pyplot(fig2)
    
    # Results table
    st.subheader("Change Analysis")
    change_data = {
        "Metric": ["Total Changed Area", "Water Changes", 
                  "Land Changes", "Vegetation Changes"],
        "Percentage (%)": [changes['total_change']*100, changes['water_change']*100,
                          changes['land_change']*100, changes['vegetation_change']*100]
    }
    st.table(pd.DataFrame(change_data))
    
    st.subheader("Disaster Risk Assessment")
    st.write(f"**Time Difference:** {date_diff} days")
    st.write(f"**Assessment:** {calamity}")
    
    if st.button("New Analysis"):
        st.session_state.page = 1
