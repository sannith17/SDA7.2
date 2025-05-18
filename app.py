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
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
from matplotlib import cm

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
    if 'b_np' in st.session_state and 'a_np' in st.session_state:
        b_np = st.session_state.b_np
        a_np = st.session_state.a_np
        before_date = st.session_state.before_date
        after_date = st.session_state.after_date

        if cnn_model and rf_model:
            progress_bar = st.progress(0.0, "Processing Images...")

            # Predict segmentation masks
            b_mask = predict_rf(b_np)
            progress_bar.progress(0.33, "Generating Before Mask...")
            a_mask = predict_rf(a_np)
            progress_bar.progress(0.66, "Generating After Mask...")

            # Generate difference heatmap and detect calamity
            diff = difference_heatmap(b_mask, a_mask)
            calamity_result = detect_calamity(before_date, after_date, b_mask, a_mask)
            progress_bar.progress(1.0, "Analysis Complete!")
            progress_bar.empty()

            # Visualization Settings
            class_colors = {
                'Water': '#1f77b4',
                'Vegetation': '#2ca02c',
                'Urban': '#ff7f0e',
                'Barren': '#d62728',
                'Forest': '#9467bd'
            }

            # Create analysis-specific labels
            if st.session_state.analysis_type == "Water Body":
                classes = {0: "Water", 1: "Vegetation", 2: "Urban", 3: "Barren"}
            else:
                classes = {0: "Forest", 1: "Vegetation", 2: "Urban", 3: "Water"}

            # Main Visualization Layout
            st.subheader("Satellite Image Analysis")

            # Original Images Row
            col1, col2 = st.columns(2)
            with col1:
                st.image(b_np, caption=f"Original Before Image ({before_date})", use_column_width=True)
            with col2:
                st.image(a_np, caption=f"Original After Image ({after_date})", use_column_width=True)

            # Segmentation Maps Row
            col3, col4 = st.columns(2)
            with col3:
                fig1, ax1 = plt.subplots(figsize=(8,6))
                ax1.imshow(b_mask, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax1.set_title(f"Before Segmentation ({before_date})")
                ax1.axis('off')
                st.pyplot(fig1)
            with col4:
                fig2, ax2 = plt.subplots(figsize=(8,6))
                ax2.imshow(a_mask, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax2.set_title(f"After Segmentation ({after_date})")
                ax2.axis('off')
                st.pyplot(fig2)

            # Change Detection Visualization
            st.subheader("Change Detection Analysis")
            
            # Calculate total area changed
            total_pixels = b_mask.size
            changed_pixels = np.sum(diff > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Display change metrics
            st.metric("Total Area Changed", f"{change_percentage:.2f}%", 
                     delta=f"{changed_pixels} pixels changed", delta_color="inverse")

            try:
                # Convert images to proper format for OpenCV
                b_np_cv = cv2.cvtColor(b_np, cv2.COLOR_RGB2BGR) if len(b_np.shape) == 3 else cv2.cvtColor(cv2.merge([b_np]*3), cv2.COLOR_RGB2BGR)
                diff_cv = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) if len(diff.shape) == 2 else diff
                
                # Create overlay with proper dimensions
                if b_np_cv.shape == diff_cv.shape:
                    overlay = cv2.addWeighted(b_np_cv, 0.7, diff_cv, 0.3, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    
                    col5, col6 = st.columns([3,1])
                    with col5:
                        fig3, ax3 = plt.subplots(figsize=(10,8))
                        ax3.imshow(overlay_rgb)
                        ax3.set_title("Change Detection Overlay (Red = Changes)")
                        ax3.axis('off')
                        st.pyplot(fig3)
                else:
                    st.warning("Image dimensions don't match for overlay. Showing difference map instead.")
                    fig3, ax3 = plt.subplots(figsize=(10,8))
                    ax3.imshow(diff, cmap='Reds')
                    ax3.set_title("Change Detection Heatmap")
                    ax3.axis('off')
                    st.pyplot(fig3)

            except Exception as e:
                st.error(f"Error creating change visualization: {str(e)}")
                st.warning("Showing basic difference map instead")
                fig3, ax3 = plt.subplots(figsize=(10,8))
                ax3.imshow(diff, cmap='Reds')
                ax3.set_title("Change Detection Heatmap")
                ax3.axis('off')
                st.pyplot(fig3)

            with col6:
                st.markdown("**Change Legend**")
                st.markdown("- Red areas: Significant changes detected")
                st.markdown("- Blue areas: Water bodies")
                st.markdown("- Green areas: Vegetation")
                st.markdown("- Orange areas: Urban regions")

            # Class Distribution Analysis
            st.subheader("Land Cover Class Distribution")
            
            # Get class statistics
            unique_b, count_b = np.unique(b_mask, return_counts=True)
            unique_a, count_a = np.unique(a_mask, return_counts=True)

            # Create percentage data
            data = []
            for class_id in range(4):
                before_pct = (np.sum(b_mask == class_id) / total_pixels) * 100
                after_pct = (np.sum(a_mask == class_id) / total_pixels) * 100
                change = after_pct - before_pct
                data.append({
                    "Class": classes[class_id],
                    "Before (%)": before_pct,
                    "After (%)": after_pct,
                    "Change (%)": change,
                    "Color": class_colors[classes[class_id]]
                })

            df = pd.DataFrame(data)
            
            # Pie Charts Visualization
            st.subheader("Class Distribution Comparison")
            col7, col8 = st.columns(2)
            with col7:
                fig4, ax4 = plt.subplots(figsize=(8,8))
                ax4.pie(df["Before (%)"], labels=df["Class"], autopct='%1.1f%%', 
                        colors=[class_colors[classes[i]] for i in range(4)],
                        startangle=90)
                ax4.set_title(f"Before {before_date}")
                ax4.axis('equal')
                st.pyplot(fig4)
                
            with col8:
                fig5, ax5 = plt.subplots(figsize=(8,8))
                ax5.pie(df["After (%)"], labels=df["Class"], autopct='%1.1f%%', 
                        colors=[class_colors[classes[i]] for i in range(4)],
                        startangle=90)
                ax5.set_title(f"After {after_date}")
                ax5.axis('equal')
                st.pyplot(fig5)

            # Intensity Chart (Change Magnitude)
            st.subheader("Change Intensity Analysis")
            fig6, ax6 = plt.subplots(figsize=(12,6))
            
            # Calculate class-wise change intensity
            class_changes = []
            for class_id in range(4):
                class_mask_before = (b_mask == class_id)
                class_mask_after = (a_mask == class_id)
                change_intensity = np.sum(class_mask_after & ~class_mask_before) / np.sum(class_mask_before) if np.sum(class_mask_before) > 0 else 0
                class_changes.append(change_intensity * 100)
            
            bars = ax6.bar(df["Class"], class_changes, color=[class_colors[classes[i]] for i in range(4)])
            ax6.set_ylabel("Change Intensity (%)")
            ax6.set_title("Percentage Change by Land Cover Class")
            ax6.set_ylim(0, max(class_changes)*1.2 if max(class_changes) > 0 else 100)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig6)

            # Calamity Analysis Section
            st.subheader("Calamity Assessment")
            if "Possible" in calamity_result:
                st.error(f"**Alert:** {calamity_result}")
                st.warning("""
                **Recommended Actions:**
                - Initiate emergency response protocols
                - Dispatch ground verification team
                - Schedule follow-up satellite imaging
                """)
            else:
                st.success(f"**Status:** {calamity_result}")
                st.info("""
                **Recommended Actions:**
                - Continue routine monitoring
                - Schedule next imaging session
                - Review historical trends
                """)

        else:
            st.error("Models not found. Please ensure models are in the correct directory.")

        st.button("Restart Analysis", on_click=reset, type="primary")
    else:
        st.warning("Please upload images in the previous steps.")
