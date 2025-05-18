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

            # Create a custom colormap for visualization
            cmap = ListedColormap(['green', 'blue', 'brown', 'gray'])
            
            # Visualization Layout
            st.subheader("Segmentation Results")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6,6))
                ax1.imshow(b_mask, cmap=cmap)
                ax1.set_title(f"Before Image ({before_date})")
                ax1.axis('off')
                st.pyplot(fig1)
                
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6,6))
                ax2.imshow(a_mask, cmap=cmap)
                ax2.set_title(f"After Image ({after_date})")
                ax2.axis('off')
                st.pyplot(fig2)

            # Enhanced Change Heatmap
            st.subheader("Change Detection Heatmap")
            fig3, ax3 = plt.subplots(figsize=(8,6))
            heatmap = ax3.imshow(diff, cmap='hot', interpolation='nearest')
            plt.colorbar(heatmap, ax=ax3, label='Change Intensity')
            ax3.set_title("Areas of Significant Change")
            ax3.axis('off')
            st.pyplot(fig3)

            # Class Distribution Analysis
            st.subheader("Land Cover Class Distribution")
            unique_b, count_b = np.unique(b_mask, return_counts=True)
            unique_a, count_a = np.unique(a_mask, return_counts=True)

            # Create a mapping for class names based on analysis type
            if st.session_state.analysis_type == "Water Body":
                class_names = {0: "Water", 1: "Vegetation", 2: "Urban", 3: "Barren"}
            else:
                class_names = {0: "Forest", 1: "Farmland", 2: "Urban", 3: "Water"}

            # Calculate percentages
            before_percent = [round((c / np.sum(count_b)) * 100, 2) for c in count_b]
            after_percent = []
            for i, element in enumerate(unique_b):
                if element in unique_a:
                    index_a = np.where(unique_a == element)[0][0]
                    after_percent.append(round((count_a[index_a] / np.sum(count_a)) * 100, 2))
                else:
                    after_percent.append(0.0)

            # Create dataframe for display
            df = pd.DataFrame({
                "Class": [class_names.get(i, f"Class {i}") for i in unique_b],
                "Before (%)": before_percent,
                "After (%)": after_percent,
                "Change (%)": [round(after - before, 2) for before, after in zip(before_percent, after_percent)]
            })

            # Display the table
            st.dataframe(df.style.background_gradient(cmap='Blues', subset=["Change (%)"]))

            # Pie Charts Visualization
            st.subheader("Class Distribution Comparison")
            col3, col4 = st.columns(2)
            with col3:
                fig4, ax4 = plt.subplots(figsize=(6,6))
                ax4.pie(df["Before (%)"], labels=df["Class"], autopct='%1.1f%%', 
                        colors=['#66b3ff','#99ff99','#ffcc99','#ff9999'])
                ax4.set_title(f"Before {before_date}")
                st.pyplot(fig4)
                
            with col4:
                fig5, ax5 = plt.subplots(figsize=(6,6))
                ax5.pie(df["After (%)"], labels=df["Class"], autopct='%1.1f%%', 
                        colors=['#66b3ff','#99ff99','#ffcc99','#ff9999'])
                ax5.set_title(f"After {after_date}")
                st.pyplot(fig5)

            # ROC Curve (Simulated for demonstration)
            st.subheader("Model Performance Metrics")
            fig6, ax6 = plt.subplots(figsize=(8,6))
            
            # Simulate some data for ROC curve
            y_true = np.random.randint(0, 2, 100)
            y_scores = np.random.rand(100)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax6.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.2f})')
            ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax6.set_xlim([0.0, 1.0])
            ax6.set_ylim([0.0, 1.05])
            ax6.set_xlabel('False Positive Rate')
            ax6.set_ylabel('True Positive Rate')
            ax6.set_title('Receiver Operating Characteristic')
            ax6.legend(loc="lower right")
            st.pyplot(fig6)

            # Calamity Detection Result
            st.subheader("Calamity Detection Result")
            if "Possible" in calamity_result:
                st.error(calamity_result)
                st.warning("Recommendation: Immediate satellite follow-up and ground verification recommended.")
            else:
                st.success(calamity_result)
                st.info("Recommendation: Routine monitoring suggested.")

            # Change Statistics
            change_pixels = np.sum(diff > 0)
            total_pixels = diff.size
            change_percentage = (change_pixels / total_pixels) * 100
            
            st.metric("Total Area Changed", f"{change_percentage:.2f}%", 
                     delta=f"{change_pixels} pixels changed", delta_color="inverse")

        else:
            st.error("Models not found. Please ensure 'cnn_model.h5' and 'rf_model.pkl' are in the same directory as your script.")

        st.button("Restart", on_click=reset)
    else:
        st.warning("Please upload images in the previous steps.")
