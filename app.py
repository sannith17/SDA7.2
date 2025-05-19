# app.py
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from sklearn import svm
from torchvision import transforms
from datetime import datetime

# Initialize session state for page navigation and data storage
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'heatmap_overlay' not in st.session_state:
    st.session_state.heatmap_overlay = None
if 'aligned_images' not in st.session_state:
    st.session_state.aligned_images = None

import streamlit as st

# Set the page layout and browser tab title
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Custom visible title with yellow color and large font
st.markdown(
    """
    <h1 style='color: yellow; font-size: 72px; font-weight: bold;'>
        Satellite Image Analysis
    </h1>
    """,
    unsafe_allow_html=True
)


# -------- Models --------
class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(6*14*14, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6*14*14)
        x = self.fc1(x)
        return x

cnn_model = DummyCNN()
cnn_model.eval()
svm_model = svm.SVC(probability=True)

# -------- Image Processing Functions --------
def preprocess_img(img, size=(64,64)):
    img = img.convert("RGB").resize(size)
    img_arr = np.array(img)/255.0
    return img_arr

def align_images(img1, img2):
    """Align images using ECC (Enhanced Correlation Coefficient) method"""
    # Convert PIL Images to numpy arrays
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, 
                                             cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(img2_np, warp_matrix, 
                                (img1_np.shape[1], img1_np.shape[0]), 
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        diff_mask = cv2.absdiff(img1_np, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)
        
        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except Exception as e:
        st.error(f"Image alignment failed: {e}")
        return img2, img2

def get_change_mask(img1, img2, threshold=30):
    # Ensure images are the same size
    img2 = img2.resize(img1.size)
    
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    return change_mask.astype(np.uint8)

def classify_land(img):
    """Simplified land classification"""
    return {"Vegetation": 45, "Land": 35, "Water": 20}

def detect_calamity(date1, date2, change_percentage):
    """Detects potential calamities based on changes and time difference"""
    date_diff = (date2 - date1).days
    
    if change_percentage > 0.15:
        if date_diff <= 10:
            return "⚠️ Possible Flood (Rapid Change)"
        elif date_diff <= 30:
            return "🔥 Possible Deforestation (Significant Change Over Short Term)"
        else:
            return "🏗️ Possible Urbanization (Gradual, Significant Change)"
    elif change_percentage > 0.05:
        return "🌱 Seasonal Changes Detected"
    return "✅ No Significant Calamity Detected"

def get_csv_bytes(data_dict):
    df = pd.DataFrame(list(data_dict.items()), columns=["Class", "Area (%)"])
    return df.to_csv(index=False).encode()

# -------- Pages --------
def page1():
    st.header("1. Model Selection")
    st.session_state.model_choice = st.selectbox("Select Analysis Model", ["SVM", "CNN", "KMeans"])
    if st.button("Next ➡️"):
        st.session_state.page = 2

def page2():
    st.markdown(
    """
    <h2 style='font-size: 36px; color: white;'>
        2. Image Upload & Dates
    </h2>
    <p style='font-size: 18px; color: lightgray;'>
        Please upload the <b>before</b> and <b>after/current</b> satellite images along with their respective dates for analysis.
    </p>
    """,
    unsafe_allow_html=True
)
    with st.sidebar:
        st.session_state.before_date = st.date_input("BEFORE image date", value=datetime(2023, 1, 1))
        st.session_state.before_file = st.file_uploader("Upload BEFORE image", 
                                                      type=["png", "jpg", "tif"], 
                                                      key="before")
        st.session_state.after_date = st.date_input("AFTER image date", value=datetime(2023, 6, 1))
        st.session_state.after_file = st.file_uploader("Upload AFTER image", 
                                                     type=["png", "jpg", "tif"], 
                                                     key="after")
    
    if st.button("⬅️ Back"):
        st.session_state.page = 1
    if st.session_state.before_file and st.session_state.after_file:
        if st.button("Next ➡️"):
            before_img = Image.open(st.session_state.before_file).convert("RGB")
            after_img = Image.open(st.session_state.after_file).convert("RGB")
            
            # Align images using ECC method
            aligned_after, aligned_black = align_images(before_img, after_img)
            st.session_state.aligned_images = {
                "before": before_img,
                "after": aligned_after,
                "aligned_black": aligned_black
            }
            
            # Calculate change mask
            st.session_state.change_mask = get_change_mask(before_img, aligned_after)
            
            # Classify land
            st.session_state.classification = classify_land(aligned_after)
            
            st.session_state.page = 3

def page3():
    st.header("3. Aligned Images Comparison")
    
    if st.session_state.aligned_images is None:
        st.error("No aligned images found. Please upload images first.")
        st.session_state.page = 2
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.aligned_images["before"], 
                caption="BEFORE Image", use_column_width=True)
    with col2:
        st.image(st.session_state.aligned_images["after"], 
                caption="Aligned AFTER Image", use_column_width=True)
    with col3:
        st.image(st.session_state.aligned_images["aligned_black"], 
                caption="Aligned Difference", use_column_width=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = 2
    if st.button("Next ➡️"):
        st.session_state.page = 4

def page4():
    st.header("4. Change Detection Heatmap")
    
    # Ensure we have valid images
    if 'aligned_images' not in st.session_state or st.session_state.aligned_images is None:
        st.error("Please upload images first")
        st.session_state.page = 2
        return
    
    # Get dimensions from change mask
    h, w = st.session_state.change_mask.shape
    
    # Create heatmap visualization
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap[..., 2] = st.session_state.change_mask * 255  # Red channel
    
    # Convert to PIL images
    heatmap_img = Image.fromarray(heatmap)
    aligned_after = st.session_state.aligned_images["after"].resize((w, h))
    
    # Create overlay and store in session
    st.session_state.heatmap_overlay = Image.blend(aligned_after.convert("RGB"), 
                                                 heatmap_img.convert("RGB"), 
                                                 alpha=0.5)
    st.image(st.session_state.heatmap_overlay, use_column_width=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = 3
    if st.button("Next ➡️"):
        st.session_state.page = 5

def page5():
    st.header("5. Land Classification & Analysis")
    
    if 'classification' not in st.session_state:
        st.error("Classification data not found. Please start from the beginning.")
        st.session_state.page = 1
        return
    
    # Calculate change percentage
    total_pixels = np.prod(st.session_state.change_mask.shape)
    total_change = (np.sum(st.session_state.change_mask) / total_pixels)
    
    # Calamity detection
    # Calamity detection
calamity = detect_calamity(
    st.session_state.before_date,
    st.session_state.after_date,
    total_change
)

# Display calamity alert with enhanced styling and description
st.markdown(
    """
    <h2 style='font-size: 36px; color: white;'>
        🚨 Calamity Detection
    </h2>
    <p style='font-size: 18px; color: lightgray;'>
        Based on the analysis of the uploaded satellite images and change metrics, the system has identified the most likely natural calamity in the region.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<h3 style='color: orange;'>{calamity}</h3>", unsafe_allow_html=True)

    
    # Classification Table
    st.subheader("Land Classification")
    df_class = pd.DataFrame(list(st.session_state.classification.items()), 
                          columns=["Class", "Area (%)"])
    st.table(df_class)
    
    # Pie Chart
    st.subheader("Land Distribution")
    fig, ax = plt.subplots()
    colors = ['#4CAF50', '#FFEB3B', '#2196F3']  # Removed urban color
    ax.pie(df_class["Area (%)"], 
          labels=df_class["Class"], 
          autopct='%1.1f%%', 
          colors=colors)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Changed area display
    st.subheader(f"Total Changed Area: {total_change*100:.2f}%")
    
    # Download Section
    st.header("Download Reports")
    csv_bytes = get_csv_bytes(st.session_state.classification)
    st.download_button("Download Classification CSV", data=csv_bytes, 
                      file_name="classification_summary.csv", mime="text/csv")
    
    if 'heatmap_overlay' in st.session_state and st.session_state.heatmap_overlay:
        buf = io.BytesIO()
        st.session_state.heatmap_overlay.save(buf, format='PNG')
        st.download_button("Download Annotated Image", data=buf.getvalue(), 
                          file_name="annotated_after.png", mime="image/png")
    
    if st.button("⬅️ Back"):
        st.session_state.page = 4

# -------- Main App --------
def main():
    st.title("Satellite Image Analysis with Alignment & Land Classification")
    
    pages = {
        1: page1,
        2: page2,
        3: page3,
        4: page4,
        5: page5
    }
    
    if st.session_state.page in pages:
        pages[st.session_state.page]()
    else:
        st.session_state.page = 1
        pages[1]()

if __name__ == "__main__":
    main()
