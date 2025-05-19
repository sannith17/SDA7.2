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

# Initialize session state for page navigation and data storage
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'heatmap_overlay' not in st.session_state:
    st.session_state.heatmap_overlay = None

st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

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

def preprocess_img(img, size=(64,64)):
    img = img.convert("RGB").resize(size)
    img_arr = np.array(img)/255.0
    return img_arr

def align_and_crop(before_img, after_img):
    # Convert PIL Images to numpy arrays
    before_np = np.array(before_img)
    after_np = np.array(after_img)
    
    # Convert to grayscale
    before_gray = cv2.cvtColor(before_np, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_np, cv2.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(before_gray, None)
    kp2, des2 = orb.detectAndCompute(after_gray, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Find homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return before_img, after_img  # Return original if alignment fails

    h, w = before_gray.shape
    aligned_after = cv2.warpPerspective(after_np, M, (w, h))

    # Create masks and find overlapping region
    before_mask = cv2.cvtColor(before_np, cv2.COLOR_RGB2GRAY) > 0
    after_mask = cv2.cvtColor(aligned_after, cv2.COLOR_RGB2GRAY) > 0
    overlap_mask = np.logical_and(before_mask, after_mask)

    coords = np.column_stack(np.where(overlap_mask))
    if coords.size == 0:
        return before_img, Image.fromarray(aligned_after)  # Return original if no overlap
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop images
    cropped_before = before_np[y0:y1, x0:x1]
    cropped_after = aligned_after[y0:y1, x0:x1]

    return Image.fromarray(cropped_before), Image.fromarray(cropped_after)

def get_change_mask(img1, img2, threshold=30):
    # Ensure images are the same size
    img2 = img2.resize(img1.size)
    
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    return change_mask.astype(np.uint8)

def classify_land(img):
    # Dummy classification - in a real app, replace with actual model prediction
    return {"Vegetation": 35, "Land": 30, "Urban": 25, "Water": 10}

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
    st.header("2. Image Upload & Dates")
    with st.sidebar:
        st.session_state.before_date = st.date_input("BEFORE image date")
        st.session_state.before_file = st.file_uploader("Upload BEFORE image", 
                                                      type=["png", "jpg", "tif"], 
                                                      key="before")
        st.session_state.after_date = st.date_input("AFTER image date")
        st.session_state.after_file = st.file_uploader("Upload AFTER image", 
                                                     type=["png", "jpg", "tif"], 
                                                     key="after")
    
    if st.button("⬅️ Back"):
        st.session_state.page = 1
    if st.session_state.before_file and st.session_state.after_file:
        if st.button("Next ➡️"):
            before_img = Image.open(st.session_state.before_file).convert("RGB")
            after_img = Image.open(st.session_state.after_file).convert("RGB")
            st.session_state.cropped_before, st.session_state.cropped_after = align_and_crop(before_img, after_img)
            st.session_state.change_mask = get_change_mask(st.session_state.cropped_before, st.session_state.cropped_after)
            st.session_state.classification = classify_land(st.session_state.cropped_after)
            st.session_state.page = 3

def page3():
    st.header("3. Aligned & Cropped Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.cropped_before, caption="BEFORE", use_column_width=True)
    with col2:
        st.image(st.session_state.cropped_after, caption="AFTER", use_column_width=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = 2
    if st.button("Next ➡️"):
        st.session_state.page = 4

def page4():
    st.header("4. Change Detection Heatmap")
    
    # Ensure we have valid images
    if 'cropped_after' not in st.session_state or 'change_mask' not in st.session_state:
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
    cropped_after = st.session_state.cropped_after.resize((w, h))  # Ensure same size
    
    # Create overlay and store in session
    st.session_state.heatmap_overlay = Image.blend(cropped_after.convert("RGB"), heatmap_img.convert("RGB"), alpha=0.5)
    st.image(st.session_state.heatmap_overlay, use_column_width=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = 3
    if st.button("Next ➡️"):
        st.session_state.page = 5

def page5():
    st.header("5. Land Classification Analysis")
    
    if 'classification' not in st.session_state:
        st.error("Classification data not found. Please start from the beginning.")
        st.session_state.page = 1
        return
    
    # Classification Table
    st.subheader("Land Classification")
    df_class = pd.DataFrame(list(st.session_state.classification.items()), 
                          columns=["Class", "Area (%)"])
    st.table(df_class)
    
    # Pie Chart
    st.subheader("Land Distribution")
    fig, ax = plt.subplots()
    ax.pie(df_class["Area (%)"], 
          labels=df_class["Class"], 
          autopct='%1.1f%%', 
          colors=['#4CAF50', '#FFEB3B', '#2196F3', '#9E9E9E'])
    ax.axis('equal')
    st.pyplot(fig)
    
    # Changed area calculation
    if 'change_mask' in st.session_state:
        total_pixels = np.prod(st.session_state.change_mask.shape)
        total_change = (np.sum(st.session_state.change_mask) / total_pixels) * 100
        st.subheader(f"Total Changed Area: {total_change:.2f}%")
    
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
