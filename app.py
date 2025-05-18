import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from datetime import datetime
from matplotlib.colors import ListedColormap

st.set_page_config(layout="wide")

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# --- Image Alignment Fix ---
def align_images(img1, img2):
    """Aligns images using ORB feature detection with fixed dimensions"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img1, M, (img2.shape, img2.shape))  # Fixed shape
    return aligned_img

# --- K-Means Fix ---
def kmeans_segmentation(image_np, n_clusters=4):
    """Performs K-Means clustering for segmentation"""
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image_np.shape, image_np.shape)

# --- Main App Flow ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Image Calamity Detector")
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
    
    # Process images with alignment
    before = np.array(Image.open(before_img).convert("RGB"))
    after = np.array(Image.open(after_img).convert("RGB"))
    aligned_before = align_images(before, after)
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax.imshow(aligned_before)
    ax.set_title("Aligned Before Image")
    ax.imshow(after)
    ax.set_title("After Image")
    st.pyplot(fig)
    
    if st.button("Show Segmentation"):
        st.session_state.page = 3

if st.session_state.page == 3:
    # Segmentation and analysis
    aligned_before = align_images(np.array(Image.open(before_img)), np.array(Image.open(after_img)))
    after = np.array(Image.open(after_img))
    
    kmeans_before = kmeans_segmentation(aligned_before)
    kmeans_after = kmeans_segmentation(after)
    
    # Change detection
    diff_mask = kmeans_after != kmeans_before
    change_percent = np.mean(diff_mask) * 100
    
    # Results display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Change Detection Map")
        plt.figure(figsize=(8, 8))
        plt.imshow(diff_mask, cmap='hot')
        st.pyplot(plt.gcf())
    
    with col2:
        st.subheader("Statistics")
        st.metric("Changed Area Percentage", f"{change_percent:.2f}%")
        st.write(f"Date Difference: {(after_date - before_date).days} days")
        
        if change_percent > 15:
            st.error("🚨 Significant changes detected - Possible calamity!")
        else:
            st.success("✅ No significant changes detected")

    if st.button("Start New Analysis"):
        st.session_state.page = 1
