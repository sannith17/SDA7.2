import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
import random
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

st.set_page_config(layout="wide")

# --- Load Models with Caching ---
@st.cache_resource
def load_models():
    """Loads the CNN and Random Forest models with caching."""
    cnn_model = None
    rf_model = None
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        st.info("CNN model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")

    try:
        rf_model = joblib.load("rf_model.pkl")
        st.info("Random Forest model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load RF model: {e}")

    return cnn_model, rf_model

cnn_model, rf_model = load_models()

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'aligned_b_np' not in st.session_state:
    st.session_state.aligned_b_np = None
if 'aligned_a_np' not in st.session_state:
    st.session_state.aligned_a_np = None

# --- Navigation Functions ---
def next_page():
    """Advances to the next page in the app."""
    st.session_state.page += 1

def reset():
    """Resets the app to the first page."""
    st.session_state.page = 1

# --- Image Preprocessing ---
def load_and_preprocess(image_file):
    """Loads and preprocesses the uploaded image, ensuring it's in RGB format."""
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    return image_np

# --- Image Alignment and Cropping ---
# --- Image Alignment and Cropping ---
def align_and_crop_images(before_np, after_np):
    """Aligns and crops two images to their overlapping region."""
    before_gray = cv2.cvtColor(before_np, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_np, cv2.COLOR_RGB2GRAY)

    # Resize 'after_gray' to match the shape of 'before_gray'
    after_gray_resized = cv2.resize(after_gray, (before_gray.shape[:2][::-1])) # Corrected resizing order

    # Perform phase cross-correlation for subpixel alignment
    try:
        shifted, error, diffphase = phase_cross_correlation(before_gray, after_gray_resized)
        shift_y, shift_x = -shifted

        # Apply the shift to the 'after' color image
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aligned_after = cv2.warpAffine(after_np, translation_matrix, (after_np.shape[:2][::-1])) # Corrected warping size

        # Determine the cropping window
        h1, w1 = before_np.shape[:2]
        h2, w2 = aligned_after.shape[:2]

        x1 = max(0, int(shift_x))
        y1 = max(0, int(shift_y))
        x2 = min(w1, int(w2 + shift_x))
        y2 = min(h1, int(h2 + shift_y))

        cropped_before = before_np[:y2, :x2] # Cropping based on calculated overlap
        cropped_after = aligned_after[:y2 - int(shift_y), :x2 - int(shift_x)] # Cropping aligned image

        return cropped_before, cropped_after

    except ValueError as e:
        st.error(f"Error during image alignment: {e}. Please ensure the uploaded images are somewhat similar in content.")
        return before_np, after_np # Return original images if alignment fails
    # The 'try' block is now correctly closed with an 'except' block

# --- Random Forest Prediction ---
def predict_rf(image_np):
    """Predicts segmentation mask using the Random Forest model."""
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    prediction = rf_model.predict(pixels)
    segmented_img = prediction.reshape(128, 128)
    return segmented_img

# --- Difference Heatmap with Colored Land Change ---
def difference_heatmap_colored(before_mask, after_mask, land_class=3): # Assuming land is class 3
    """Generates a colored heatmap showing changes, especially in land."""
    diff = after_mask.astype(int) - before_mask.astype(int)
    heatmap = np.zeros((*diff.shape, 3), dtype=np.uint8)

    # Red for increased land, blue for decreased land
    heatmap[np.logical_and(before_mask != land_class, after_mask == land_class)] = [255, 0, 0]  # Increased land
    heatmap[np.logical_and(before_mask == land_class, after_mask != land_class)] = [0, 0, 255]  # Decreased land
    return heatmap

# --- Calamity Detection ---
def detect_calamity(date1, date2, mask1, mask2):
    """Detects potential calamities based on changes in the segmentation masks."""
    diff_mask = mask2 != mask1
    change_percentage = np.sum(diff_mask) / diff_mask.size
    date_diff = (date2 - date1).days

    if change_percentage > 0.15:
        if date_diff <= 10:
            return "⚠️ Possible Flood (Rapid Change)"
        elif date_diff <= 30:
            return "🔥 Possible Deforestation (Significant Change Over Short Term)"
        else:
            return "🌊 Urbanization or Seasonal Change (Gradual, Significant Change)"
    return "✅ No Significant Calamity Detected"

# --- Layout Pages ---
if st.session_state.page == 1:
    st.title("🌍 Satellite Image Calamity Detector")
    st.subheader("Step 1: Choose Analysis Type")
    st.session_state.analysis_type = st.radio("Select Type of Analysis:", ["Land Body", "Water Body"])
    st.button("Next", on_click=next_page)

elif st.session_state.page == 2:
    st.title("Step 2: Upload Satellite Images")
    col1, col2 = st.columns(2)
    with col1:
        before_image = st.file_uploader("Upload BEFORE image", type=['jpg', 'jpeg', 'png'], key="before")
        before_date = st.date_input("Before Date")
    with col2:
        after_image = st.file_uploader("Upload AFTER image", type=['jpg', 'jpeg', 'png'], key="after")
        after_date = st.date_input("After Date")

    if before_image and after_image:
        st.session_state.before_image = before_image
        st.session_state.after_image = after_image
        st.session_state.before_date = before_date
        st.session_state.after_date = after_date
        st.button("Next", on_click=next_page)
    elif before_image or after_image:
        st.warning("Please upload both 'BEFORE' and 'AFTER' images.")
    else:
        st.info("Please upload satellite images for analysis.")

elif st.session_state.page == 3:
    st.title("Step 3: Align and Crop Images")
    if 'before_image' in st.session_state and 'after_image' in st.session_state:
        b_img_np = load_and_preprocess(st.session_state.before_image)
        a_img_np = load_and_preprocess(st.session_state.after_image)

        aligned_b, aligned_a = align_and_crop_images(b_img_np, a_img_np)

        st.image([aligned_b, aligned_a], caption=[f"Aligned Before ({st.session_state.before_date})", f"Aligned After ({st.session_state.after_date})"], width=300)

        st.session_state.aligned_b_np = aligned_b
        st.session_state.aligned_a_np = aligned_a
        st.button("Next", on_click=next_page)
    else:
        st.warning("Please upload images in the previous step.")

elif st.session_state.page == 4:
    st.title("Step 4: Calamity Detection and Visualization")
    if 'aligned_b_np' in st.session_state and 'aligned_a_np' in st.session_state and rf_model is not None:
        b_np = st.session_state.aligned_b_np
        a_np = st.session_state.aligned_a_np
        before_date = st.session_state.before_date
        after_date = st.session_state.after_date

        progress_bar = st.progress(0.0, "Processing Images...")

        # Predict segmentation masks
        b_mask = predict_rf(b_np)
        progress_bar.progress(0.33, "Generating Before Mask...")
        a_mask = predict_rf(a_np)
        progress_bar.progress(0.66, "Generating After Mask...")

        # Generate colored difference heatmap
        colored_diff = difference_heatmap_colored(b_mask, a_mask, land_class=3) # Assuming land is class 3
        progress_bar.progress(1.0, "Analysis Complete!")
        progress_bar.empty()

        col_masks, col_heatmap = st.columns(2)
        with col_masks:
            st.image([b_mask, a_mask], caption=[f"RF Mask Before ({before_date})", f"RF Mask After ({after_date})"])
        with col_heatmap:
            st.subheader("Land Change Heatmap")
            st.image(colored_diff, caption="Red = Increased Land, Blue = Decreased Land", use_container_width=True)

        # Calculate area percentages
        def calculate_area_percentage(mask, class_id):
            return np.sum(mask == class_id) / mask.size * 100

        water_b_percent = calculate_area_percentage(b_mask, 1) # Assuming water is class 1
        land_b_percent = calculate_area_percentage(b_mask, 3) # Assuming land is class 3
        veg_b_percent = calculate_area_percentage(b_mask, 2) # Assuming vegetation is class 2

        water_a_percent = calculate_area_percentage(a_mask, 1)
        land_a_percent = calculate_area_percentage(a_mask, 3)
        veg_a_percent = calculate_area_percentage(a_mask, 2)

        df_area = pd.DataFrame({
            "Category": ["Water", "Land (including Vegetation)"],
            f"Before ({before_date}) (%)": [water_b_percent, land_b_percent + veg_b_percent],
            f"After ({after_date}) (%)": [water_a_percent, land_a_percent + veg_a_percent],
            "Change (%)": [round(water_a_percent - water_b_percent, 2), round((land_a_percent + veg_a_percent) - (land_b_percent + veg_b_percent), 2)]
        })
        st.subheader("Area Percentage Comparison")
        st.dataframe(df_area)

        # Pie Charts
        st.subheader("Area Distribution")
        col_pie_b, col_pie_a = st.columns(2)

        with col_pie_b:
            labels_b = [f"Water ({water_b_percent:.1f}%)", f"Land+Veg ({land_b_percent + veg_b_percent:.1f}%)"]
            sizes_b = [water_b_percent, land_b_percent + veg_b_percent]
            fig_pie_b, ax_pie_b = plt.subplots()
            ax_pie_b.pie(sizes_b, labels=labels_b, autopct='%1.1f%%', startangle=90)
            ax_pie_b.axis('equal')
            st.pyplot(fig_pie_b)
            st.caption(f"Area Distribution Before ({before_date})")

        with col_pie_a:
            labels_a = [f"Water ({water_a_percent:.1f}%)", f"Land+Veg ({land_a_percent + veg_a_percent:.1f}%)"]
            sizes_a = [water_a_percent, land_a_percent + veg_a_percent]
            fig_pie_a, ax_pie_a = plt.subplots()
            ax_pie_a.pie(sizes_a, labels=labels_a, autopct='%1.1f%%', startangle=90)
            ax_pie_a.axis('equal')
            st.pyplot(fig_pie_a)
            st.caption(f"Area Distribution After ({after_date})")

        # Fake ROC Curve
        st.subheader("Fake ROC Curve (For Showoff)")
        fpr = sorted([random.random() for _ in range(20)])
        tpr = [random.random() for _ in range(20)]
        tpr.sort()
        roc_auc = random.uniform(0.7, 0.95) # Fake AUC value

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'Fake ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('Fake False Positive Rate')
        ax_roc.set_ylabel('Fake True Positive Rate')
        ax_roc.set_title('Fake Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.caption("This is a randomly generated ROC curve for demonstration purposes only. Real ROC analysis requires ground truth data.")

        # Calamity Detection (using original logic on unaligned masks for simplicity, adjust if needed)
        calamity_result = detect_calamity(before_date, after_date, predict_rf(st.session_state.b_np), predict_rf(st.session_state.a_np))
        st.subheader("Possible Calamity Analysis")
        st.success(f"Prediction: {calamity_result}")

        if hasattr(st.session_state.before_image, "size") and st.session_state.before_image.size > 5e6:
            pca_b = pca_visualization(b_np)
            st.subheader("PCA Visualization (Before Image > 5MB)")
            st.image(pca_b, caption="PCA Visualization (Before)", use_container_width=True)
        if hasattr(st.session_state.after_image, "size") and st.session_state.after_image.size > 5e6:
            pca_a = pca_visualization(a_np)
            st.subheader("PCA Visualization (After Image > 5MB)")
            st.image(pca_a, caption="PCA Visualization (After)", use_container_width=True)

        st.button("Restart", on_click=reset)

    else:
        st.warning("Please upload both BEFORE and AFTER images on the previous page, and ensure the model is loaded.")
