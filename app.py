import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from datetime import datetime

st.set_page_config(layout="wide")

# --- Color Mapping & Constants ---
ANALYSIS_TYPES = {
    "Water Bodies": {"color": [0, 0.5, 1], "index": 0},
    "Land Areas": {"color": [1, 0.4, 0.4], "index": 1},
    "Vegetation": {"color": [0.4, 1, 0.4], "index": 2}
}

# --- Image Alignment ---
def align_images(img1, img2):
    """Aligns images using ORB feature detection"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img1, M, (img2.shape, img2.shape))
    return aligned_img

# --- Spectral Index Calculation ---
def calculate_ndwi(img):
    """Water detection using modified NDWI for RGB"""
    green = img[:,:,1].astype(float)
    blue = img[:,:,0].astype(float)
    return (green - blue) / (green + blue + 1e-6)

def calculate_ndvi(img):
    """Vegetation detection using modified NDVI for RGB"""
    red = img[:,:,0].astype(float)
    green = img[:,:,1].astype(float)
    return (green - red) / (green + red + 1e-6)

# --- Change Detection ---
def detect_changes(before, after, analysis_type):
    """Quantifies changes using spectral indices"""
    if analysis_type == "Water Bodies":
        before_idx = calculate_ndwi(before)
        after_idx = calculate_ndwi(after)
    elif analysis_type == "Vegetation":
        before_idx = calculate_ndvi(before)
        after_idx = calculate_ndvi(after)
    else:  # Land
        before_idx = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        after_idx = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    
    diff = after_idx - before_idx
    return np.clip(diff * 255, 0, 255).astype(np.uint8)

# --- Session State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 1
    st.session_state.analysis_type = None
    st.session_state.georef_data = {}

# --- Page 1: Analysis Type Selection ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Analysis Setup")
    cols = st.columns(3)
    for i, (name, params) in enumerate(ANALYSIS_TYPES.items()):
        with cols[i]:
            if st.button(f"**{name}**", use_container_width=True):
                st.session_state.analysis_type = name
                st.session_state.page = 2

# --- Page 2: Data Input ---
elif st.session_state.page == 2:
    st.title("📤 Data Upload & Alignment")
    
    col1, col2 = st.columns(2)
    with col1:
        before_img = st.file_uploader("Before Image", type=["png", "jpg"])
        before_date = st.date_input("Before Date")
    with col2:
        after_img = st.file_uploader("After Image", type=["png", "jpg"])
        after_date = st.date_input("After Date")
    
    if st.button("Process Images") and before_img and after_img:
        # Load and align images
        before = np.array(Image.open(before_img).convert("RGB"))
        after = np.array(Image.open(after_img).convert("RGB"))
        aligned_before = align_images(before, after)
        
        # Store in session state
        st.session_state.georef_data = {
            "before": aligned_before,
            "after": after,
            "dates": (before_date, after_date)
        }
        st.session_state.page = 3

# --- Page 3: Georeferencing Preview ---
elif st.session_state.page == 3:
    st.title("🛰️ Georeferenced Data Preview")
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax.imshow(st.session_state.georef_data["before"])
    ax.set_title("Aligned Before Image")
    ax.imshow(st.session_state.georef_data["after"])
    ax.set_title("After Image")
    st.pyplot(fig)
    
    if st.button("Proceed to Analysis"):
        st.session_state.page = 4

# --- Page 4: Analysis & Visualization ---
elif st.session_state.page == 4:
    st.title("📊 Analysis Results")
    data = st.session_state.georef_data
    analysis_type = st.session_state.analysis_type
    
    # Change detection
    change_map = detect_changes(data["before"], data["after"], analysis_type)
    
    # Quantitative analysis
    threshold = 0.2 * 255
    change_percentage = np.mean(change_map > threshold) * 100
    
    # Calamity assessment
    date_diff = (data["dates"] - data["dates"]).days
    if date_diff <= 7 and change_percentage > 15:
        risk = "⚠️ Immediate disaster risk!"
    elif date_diff <= 30 and change_percentage > 25:
        risk = "🔍 Potential environmental change"
    else:
        risk = "✅ Stable conditions"
    
    # Visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Change Heatmap")
        fig1, ax1 = plt.subplots()
        ax1.imshow(change_map, cmap='hot')
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Change Statistics")
        plt.figure(figsize=(6, 4))
        plt.pie([change_percentage, 100-change_percentage], 
                labels=['Changed', 'Unchanged'], 
                colors=['#ff4444', '#44ff44'], autopct='%1.1f%%')
        st.pyplot(plt.gcf())
        
        st.metric("Total Changed Area", f"{change_percentage:.1f}%")
        st.write(f"**Time Difference:** {date_diff} days")
        st.write(f"**Risk Assessment:** {risk}")
    
    if st.button("New Analysis"):
        st.session_state.page = 1
