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
from sklearn.metrics import roc_curve, auc  # For ROC curve (requires ground truth)

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

# --- PCA Visualization ---
def pca_visualization(image_np):
    """Performs PCA on the image for visualization."""
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    pca = PCA(n_components=3)
    scaled = StandardScaler().fit_transform(pixels)
    pca_img = pca.fit_transform(scaled)
    pca_img = pca_img.reshape(128, 128, 3)
    pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min())
    return pca_img

# --- Random Forest Prediction ---
def predict_rf(image_np):
    """Predicts segmentation mask using the Random Forest model."""
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    prediction = rf_model.predict(pixels)
    segmented_img = prediction.reshape(128, 128)
    return segmented_img

# --- CNN Prediction (Currently Not Directly Used in the Flow) ---
def predict_cnn(image_np):
    """Predicts using the CNN model."""
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array)
    return predictions

# --- Difference Map ---
def difference_heatmap(before_mask, after_mask):
    """Generates a heatmap showing the difference between two masks."""
    diff = after_mask != before_mask
    return diff.astype(np.uint8) * 255

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
    st.title("Step 3: Image Preview")
    if 'before_image' in st.session_state and 'after_image' in st.session_state:
        b_img_np = load_and_preprocess(st.session_state.before_image)
        a_img_np = load_and_preprocess(st.session_state.after_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(b_img_np, caption=f"Before Image ({st.session_state.before_date})")
        with col2:
            st.image(a_img_np, caption=f"After Image ({st.session_state.after_date})")

        st.session_state.b_np = b_img_np
        st.session_state.a_np = a_img_np
        st.button("Next", on_click=next_page)
    else:
        st.warning("Please upload images in the previous step.")

elif st.session_state.page == 4:
    st.title("Step 4: Calamity Detection and Visualization")
    if 'b_np' in st.session_state and 'a_np' in st.session_state and rf_model is not None:
        b_np = st.session_state.b_np
        a_np = st.session_state.a_np
        before_date = st.session_state.before_date
        after_date = st.session_state.after_date

        progress_bar = st.progress(0.0, "Processing Images...")

        # Predict segmentation masks
        b_mask = predict_rf(b_np)
        progress_bar.progress(0.33, "Generating Before Mask...")
        a_mask = predict_rf(a_np)
        progress_bar.progress(0.66, "Generating After Mask...")

        # Generate difference heatmap
        diff = difference_heatmap(b_mask, a_mask)
        progress_bar.progress(1.0, "Analysis Complete!")
        progress_bar.empty()

        col1, col2 = st.columns(2)
        with col1:
            st.image(b_mask, caption=f"Random Forest - Before Mask ({before_date})")
        with col2:
            st.image(a_mask, caption=f"Random Forest - After Mask ({after_date})")

        st.subheader("Heatmap of Changes")
        st.image(diff, caption="Change Heatmap (White = Change)", use_container_width=True)

        # Calculate percentage change of elements
        unique_b, count_b = np.unique(b_mask, return_counts=True)
        total_b = np.sum(count_b)
        percentages_b = {k: v / total_b * 100 for k, v in zip(unique_b, count_b)}

        unique_a, count_a = np.unique(a_mask, return_counts=True)
        total_a = np.sum(count_a)
        percentages_a = {k: v / total_a * 100 for k, v in zip(unique_a, count_a)}

        class_labels = np.unique(np.concatenate((unique_b, unique_a)))
        change_data = []
        for label in class_labels:
            percent_b = percentages_b.get(label, 0)
            percent_a = percentages_a.get(label, 0)
            change = percent_a - percent_b
            change_data.append({"Element": f"Class {int(label)}", "Change (%)": round(change, 2)})

        df_change = pd.DataFrame(change_data)
        st.subheader("Percentage Change of Elements")
        st.dataframe(df_change)

        # --- ROC Curves (Conceptual - Requires Ground Truth) ---
        st.subheader("ROC Curves (Conceptual)")
        st.info("Generating meaningful ROC curves requires ground truth data (manually labeled changes). Without it, we can only show a conceptual placeholder.")
        # In a real scenario, you would compare your model's predictions against ground truth.
        # Example of how you might plot if you had ground truth:
        # fpr, tpr, thresholds = roc_curve(ground_truth, model_probabilities)
        # roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Conceptual ROC Curve')
        st.pyplot(fig_roc)
        st.caption("This plot is a placeholder. Actual ROC curves require ground truth data to evaluate the performance of the change detection.")

        # --- Pie Charts ---
        st.subheader("Percentage Distribution of Elements")
        col_pie_b, col_pie_a = st.columns(2)

        with col_pie_b:
            fig_pie_b, ax_pie_b = plt.subplots()
            labels_b = [f"Class {int(i)}" for i in unique_b]
            ax_pie_b.pie(count_b, labels=labels_b, autopct='%1.1f%%', startangle=90)
            ax_pie_b.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig_pie_b)
            st.caption(f"Distribution Before ({before_date})")

        with col_pie_a:
            fig_pie_a, ax_pie_a = plt.subplots()
            labels_a = [f"Class {int(i)}" for i in unique_a]
            ax_pie_a.pie(count_a, labels=labels_a, autopct='%1.1f%%', startangle=90)
            ax_pie_a.axis('equal')
            st.pyplot(fig_pie_a)
            st.caption(f"Distribution After ({after_date})")

        # --- Possible Calamity Information ---
        st.subheader("Possible Calamity Analysis")
        date_diff = (after_date - before_date).days

        # Assuming class 1 represents a feature that might indicate a calamity (you'll need to adjust this)
        water_change_percent = df_change[df_change['Element'] == 'Class 1']['Change (%)'].iloc[0] if ('Class 1' in df_change['Element'].values) else 0
        veg_change_percent = df_change[df_change['Element'] == 'Class 2']['Change (%)'].iloc[0] if ('Class 2' in df_change['Element'].values) else 0 # Assuming class 2 is vegetation

        if water_change_percent > 15 and date_diff <= 10:
            st.error("⚠️ Possible Rapid Increase in Water - Potential Flood Risk")
        elif veg_change_percent < -15 and date_diff <= 30:
            st.error("🔥 Significant Decrease in Vegetation - Possible Deforestation")
        elif water_change_percent > 10 and date_diff > 30:
            st.info("🌊 Gradual Increase in Water - Could indicate long-term changes")
        elif veg_change_percent < -10 and date_diff > 30:
            st.info("🌿 Gradual Decrease in Vegetation - Could indicate seasonal changes or other factors")
        else:
            st.success("✅ No immediate high-risk calamity pattern detected based on the defined thresholds.")

        st.button("Restart", on_click=reset)

    else:
        st.warning("Please upload both BEFORE and AFTER images on the previous page, and ensure the model is loaded.")
