

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import svm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# -------- Dummy CNN model (replace with your real model) --------
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

# -------- Dummy SVM model (replace with your real model) --------
svm_model = svm.SVC(probability=True)
# Normally you'd train and load model; here we mock

# -------- Image preprocessing --------
def preprocess_img(img, size=(64,64)):
    img = img.convert("RGB").resize(size)
    img_arr = np.array(img)/255.0
    return img_arr

# -------- Align and crop images to common overlap --------
def align_and_crop(before_img, after_img):
    # Convert to gray
    before_gray = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2GRAY)

    # Detect ORB features and descriptors.
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(before_gray, None)
    kp2, des2 = orb.detectAndCompute(after_gray, None)

    # Match features.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Compute homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = before_gray.shape
    aligned_after = cv2.warpPerspective(np.array(after_img), M, (w, h))

    # Crop common overlapping area (intersection of valid pixels)
    before_mask = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2GRAY) > 0
    after_mask = cv2.cvtColor(aligned_after, cv2.COLOR_RGB2GRAY) > 0
    overlap_mask = np.logical_and(before_mask, after_mask)

    # Find bounding rect of overlap
    coords = np.column_stack(np.where(overlap_mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped_before = np.array(before_img)[y0:y1, x0:x1]
    cropped_after = aligned_after[y0:y1, x0:x1]

    return Image.fromarray(cropped_before), Image.fromarray(cropped_after)

# -------- Generate change mask between two images --------
def get_change_mask(img1, img2, threshold=30):
    # Convert to grayscale
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    return change_mask

# -------- Land classification summary (mock) --------
def classify_land(img):
    # Dummy classification percentages, replace with your own model/classification
    return {"Vegetation": 35, "Land": 30, "Urban": 25, "Water": 10}

# -------- CNN visualization (mock heatmap) --------
def cnn_visualization(img):
    # Dummy heatmap for CNN (replace with real CAM/GradCAM etc.)
    heatmap = np.random.rand(img.size[1], img.size[0])
    plt.figure(figsize=(6,6))
    sns.heatmap(heatmap, cmap="viridis")
    plt.title("CNN Feature Map (Dummy)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# -------- Download CSV summary --------
def get_csv_bytes(data_dict):
    df = pd.DataFrame(list(data_dict.items()), columns=["Class", "Area (%)"])
    return df.to_csv(index=False).encode()

# -------- Main Streamlit App --------
def main():
    st.title("Satellite Image Analysis with Alignment & Land Classification")

    st.sidebar.header("Upload BEFORE and AFTER images with dates")
    before_date = st.sidebar.date_input("Upload date for BEFORE image")
    before_file = st.sidebar.file_uploader("Upload BEFORE image", type=["png", "jpg", "tif"], key="before")

    after_date = st.sidebar.date_input("Upload date for AFTER image")
    after_file = st.sidebar.file_uploader("Upload AFTER image", type=["png", "jpg", "tif"], key="after")

    model_choice = st.sidebar.selectbox("Select Model", ["SVM", "CNN", "KMeans"])

    if before_file and after_file:
        before_img = Image.open(before_file).convert("RGB")
        after_img = Image.open(after_file).convert("RGB")

        # Align and crop
        cropped_before, cropped_after = align_and_crop(before_img, after_img)

        st.header("Aligned and Cropped BEFORE and AFTER Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cropped_before, caption="BEFORE (Aligned & Cropped)", use_column_width=True)
        with col2:
            st.image(cropped_after, caption="AFTER (Aligned & Cropped)", use_column_width=True)

        # Change mask
        change_mask = get_change_mask(cropped_before, cropped_after)

        # Show change heatmap on AFTER image only
        st.header("Change Heatmap (AFTER vs BEFORE)")
        heatmap = np.zeros((change_mask.shape[0], change_mask.shape[1], 3), dtype=np.uint8)
        heatmap[..., 2] = change_mask * 255  # highlight changes in blue channel
        heatmap_img = Image.fromarray(heatmap)
        heatmap_overlay = Image.blend(cropped_after, heatmap_img, alpha=0.5)
        st.image(heatmap_overlay, caption="Heatmap Overlay on AFTER Image", use_column_width=True)

        # Land classification summary on AFTER image only
        classification = classify_land(cropped_after)
        st.header("Land Classification Summary (AFTER Image)")

        # Table
        df_class = pd.DataFrame(list(classification.items()), columns=["Class", "Area (%)"])
        st.table(df_class)

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(df_class["Area (%)"], labels=df_class["Class"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Model prediction (dummy)
        st.header("Model Prediction")

        if model_choice == "CNN":
            st.write("Running CNN model (dummy output)...")
            # preprocess image for CNN dummy
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor()
            ])
            input_tensor = transform(cropped_after).unsqueeze(0)
            with torch.no_grad():
                outputs = cnn_model(input_tensor)
                _, predicted = torch.max(outputs, 1)
            st.write(f"Predicted class (dummy): {predicted.item()}")

            # CNN visualization
            st.write("CNN Visualization:")
            vis_img = cnn_visualization(cropped_after)
            st.image(vis_img, use_column_width=True)

        elif model_choice == "SVM":
            st.write("Running SVM model (dummy output)...")
            # Flatten and preprocess image for dummy SVM
            img_np = np.array(cropped_after.resize((32,32))).flatten().reshape(1, -1)
            # Random dummy prediction (since model is not trained)
            pred = 1  # just dummy fixed output
            st.write(f"Predicted class (dummy): {pred}")

        else:
            st.write("Running KMeans clustering (dummy output)...")
            st.write("Cluster labels would appear here.")

        # Download options
        st.header("Download Reports")

        csv_bytes = get_csv_bytes(classification)
        st.download_button("Download Classification CSV", data=csv_bytes, file_name="classification_summary.csv", mime="text/csv")

        # Prepare annotated image download (heatmap overlay)
        buf = io.BytesIO()
        heatmap_overlay.save(buf, format='PNG')
        st.download_button("Download Annotated AFTER Image", data=buf.getvalue(), file_name="annotated_after.png", mime="image/png")

if __name__ == "__main__":
    main()
