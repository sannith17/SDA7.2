import streamlit as st
import numpy as np
import pandas as pd
import cv2
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import joblib
from skimage.registration import phase_cross_correlation

# Set page
st.set_page_config(page_title="🌍 Satellite Image Calamity Detector", layout="wide")
st.title("🌍 Satellite Image Calamity Detector")

# Sidebar
st.sidebar.title("Upload Images")
before_image_file = st.sidebar.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"])
after_image_file = st.sidebar.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"])

before_date = st.sidebar.date_input("Before Date", value=datetime.date(2022, 1, 1))
after_date = st.sidebar.date_input("After Date", value=datetime.date(2024, 1, 1))

# Load models
cnn_model = None
rf_model = None
try:
    cnn_model = tf.keras.models.load_model("cnn_model.h5")
except Exception as e:
    st.warning(f"CNN model not found: {e}")

try:
    rf_model = joblib.load("rf_model.pkl")
except Exception as e:
    st.warning(f"Random Forest model not found: {e}")

# Load and preprocess image
def load_and_preprocess(image_file):
    if image_file is not None:
        img_bytes = image_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return None

# Image Alignment and Cropping
def align_and_crop_images(before_img, after_img):
    if before_img is None or after_img is None:
        return None, None
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)

    try:
        # Resize after_gray to match before_gray shape for alignment
        after_gray_resized = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))
        shifted, error, diffphase = phase_cross_correlation(before_gray, after_gray_resized)
        shift_y, shift_x = -shifted

        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aligned_after = cv2.warpAffine(after_img, translation_matrix, (after_img.shape[1], after_img.shape[0]))

        # Determine cropping window
        h1, w1 = before_img.shape[:2]
        h2, w2 = aligned_after.shape[:2]
        x1 = max(0, int(shift_x))
        y1 = max(0, int(shift_y))
        x2 = min(w1, int(w2 + shift_x))
        y2 = min(h1, int(h2 + shift_y))

        cropped_before = before_img[y1:y2, x1:x2]
        cropped_after = aligned_after[y1 - int(shift_y):y2 - int(shift_y), x1 - int(shift_x):x2 - int(shift_x)]

        return cropped_before, cropped_after
    except Exception as e:
        st.warning(f"Image alignment failed: {e}. Proceeding with original images.")
        return before_img, after_img

# Function to generate colored change map
def generate_colored_change_map(before, after):
    if before is None or after is None:
        return None
    before_resized = cv2.resize(before, (128, 128))
    after_resized = cv2.resize(after, (128, 128))
    diff = cv2.absdiff(before_resized, after_resized)
    mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    colored_change = np.zeros_like(before_resized)
    colored_change[mask > 20] = [255, 0, 0]  # Red for significant change
    return colored_change

# Function to predict masks (simplified for visualization)
def predict_masks_simple(img):
    if img is None:
        return None
    img_resized = cv2.resize(img, (64, 64))
    # Simple thresholding for demonstration (replace with actual model prediction)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    water_mask = (gray < 100).astype(np.uint8) * 255
    land_mask = ((gray >= 100) & (gray < 200)).astype(np.uint8) * 255
    veg_mask = (gray >= 200).astype(np.uint8) * 255
    return water_mask, land_mask, veg_mask

# Function to calculate area percentages
def calculate_area_percentage(mask):
    if mask is None:
        return 0
    return np.sum(mask > 0) / mask.size * 100

# Main workflow
if before_image_file and after_image_file:
    before_img = load_and_preprocess(before_image_file)
    after_img = load_and_preprocess(after_image_file)

    if before_img is not None and after_img is not None:
        aligned_before, aligned_after = align_and_crop_images(before_img, after_img)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Before Image ({before_date})")
            st.image(aligned_before, use_column_width=True)
        with col2:
            st.subheader(f"After Image ({after_date})")
            st.image(aligned_after, use_column_width=True)

        change_map = generate_colored_change_map(aligned_before, aligned_after)
        if change_map is not None:
            st.subheader("Change Detection (Red = Significant Change)")
            st.image(change_map, use_column_width=True)
        else:
            st.warning("Could not generate change map.")

        before_water, before_land, before_veg = predict_masks_simple(aligned_before)
        after_water, after_land, after_veg = predict_masks_simple(aligned_after)

        before_water_percent = calculate_area_percentage(before_water)
        before_land_percent = calculate_area_percentage(before_land)
        before_veg_percent = calculate_area_percentage(before_veg)

        after_water_percent = calculate_area_percentage(after_water)
        after_land_percent = calculate_area_percentage(after_land)
        after_veg_percent = calculate_area_percentage(after_veg)

        area_data = pd.DataFrame({
            "Category": ["Water", "Land", "Vegetation"],
            f"Before ({before_date}) (%)": [before_water_percent, before_land_percent, before_veg_percent],
            f"After ({after_date}) (%)": [after_water_percent, after_land_percent, after_veg_percent],
            "Change (%)": [round(after_water_percent - before_water_percent, 2),
                           round(after_land_percent - before_land_percent, 2),
                           round(after_veg_percent - before_veg_percent, 2)]
        })
        st.subheader("Area Percentage Comparison")
        st.dataframe(area_data)

        # Pie Charts
        st.subheader("Area Distribution")
        col_pie_b, col_pie_a = st.columns(2)

        with col_pie_b:
            labels_b = [f"Water ({before_water_percent:.1f}%)",
                        f"Land ({before_land_percent:.1f}%)",
                        f"Vegetation ({before_veg_percent:.1f}%)"]
            sizes_b = [before_water_percent, before_land_percent, before_veg_percent]
            fig_pie_b, ax_pie_b = plt.subplots()
            ax_pie_b.pie(sizes_b, labels=labels_b, autopct='%1.1f%%', startangle=90)
            ax_pie_b.axis('equal')
            st.pyplot(fig_pie_b)
            st.caption(f"Area Distribution Before ({before_date})")

        with col_pie_a:
            labels_a = [f"Water ({after_water_percent:.1f}%)",
                        f"Land ({after_land_percent:.1f}%)",
                        f"Vegetation ({after_veg_percent:.1f}%)"]
            sizes_a = [after_water_percent, after_land_percent, after_veg_percent]
            fig_pie_a, ax_pie_a = plt.subplots()
            ax_pie_a.pie(sizes_a, labels=labels_a, autopct='%1.1f%%', startangle=90)
            ax_pie_a.axis('equal')
            st.pyplot(fig_pie_a)
            st.caption(f"Area Distribution After ({after_date})")

        # Calamity Alerts (simplified based on area change)
        st.subheader("🛰️ Possible Changes Detected")
        water_change = after_water_percent - before_water_percent
        veg_change = after_veg_percent - before_veg_percent
        date_diff = (after_date - before_date).days

        if water_change > 5 and date_diff <= 10:
            st.error("⚠️ Rapid Increase in Water Area Possible Flood")
        elif veg_change < -5 and date_diff <= 30:
            st.error("🔥 Significant Decrease in Vegetation Possible Deforestation")
        elif water_change < -5 and date_diff <= 10:
            st.warning("💧 Rapid Decrease in Water Area")
        elif veg_change > 5 and date_diff <= 30:
            st.info("🌳 Significant Increase in Vegetation")
        else:
            st.success("✅ No significant rapid changes detected based on area analysis.")

else:
    st.info("Please upload both BEFORE and AFTER satellite images and select dates.")
