import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
from matplotlib import cm
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")

# --- Load Models with Caching ---
@st.cache_resource
def load_models():
    cnn_model = None
    svm_model = None
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        st.info("CNN model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")

    try:
        svm_model = joblib.load("svm_model.pkl")
        st.info("SVM model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load SVM model: {e}")

    return cnn_model, svm_model

cnn_model, svm_model = load_models()

if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

def next_page():
    st.session_state.page += 1

def reset():
    st.session_state.page = 1

def load_and_preprocess(image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    return image_np

def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image_np.shape[0], image_np.shape[1])

def predict_svm(image_np):
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    prediction = svm_model.predict(pixels)
    segmented_img = prediction.reshape(128, 128)
    return segmented_img

def predict_cnn(image_np):
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array)
    return np.argmax(predictions, axis=-1)[0]

def difference_heatmap(before_mask, after_mask):
    diff = after_mask != before_mask
    return diff.astype(np.uint8) * 255

def detect_calamity(date1, date2, mask1, mask2):
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
            st.image(b_img_np, caption=f"Before Image ({st.session_state.before_date})", use_container_width=True)
        with col2:
            st.image(a_img_np, caption=f"After Image ({st.session_state.after_date})", use_container_width=True)

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

        if cnn_model and svm_model:
            progress_bar = st.progress(0.0, "Processing Images...")

            with st.spinner("Running SVM analysis..."):
                b_mask_svm = predict_svm(b_np)
                a_mask_svm = predict_svm(a_np)
                progress_bar.progress(0.25)

            with st.spinner("Running K-Means clustering..."):
                b_mask_kmeans = kmeans_segmentation(b_np)
                a_mask_kmeans = kmeans_segmentation(a_np)
                progress_bar.progress(0.5)

            with st.spinner("Running CNN analysis..."):
                b_mask_cnn = predict_cnn(b_np)
                a_mask_cnn = predict_cnn(a_np)
                progress_bar.progress(0.75)

            diff = difference_heatmap(b_mask_svm, a_mask_svm)
            calamity_result = detect_calamity(before_date, after_date, b_mask_svm, a_mask_svm)
            progress_bar.progress(1.0, "Analysis Complete!")
            progress_bar.empty()

            class_colors = {
                'Water': '#1f77b4',
                'Vegetation': '#2ca02c',
                'Urban': '#ff7f0e',
                'Barren': '#d62728',
                'Forest': '#9467bd'
            }

            if st.session_state.analysis_type == "Water Body":
                classes = {0: "Water", 1: "Vegetation", 2: "Urban", 3: "Barren"}
            else:
                classes = {0: "Forest", 1: "Vegetation", 2: "Urban", 3: "Water"}

            st.subheader("Satellite Image Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.image(b_np, caption=f"Original Before Image ({before_date})", use_container_width=True)
            with col2:
                st.image(a_np, caption=f"Original After Image ({after_date})", use_container_width=True)

            st.markdown("**SVM Results**")
            col3, col4 = st.columns(2)
            with col3:
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                im1 = ax1.imshow(b_mask_svm, cmap=ListedColormap([class_colors[classes[i]] for i in sorted(classes)]))
                ax1.set_title(f"Before ({before_date}) - SVM Segmentation")
                ax1.axis('off')
                st.pyplot(fig1)
            with col4:
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                im2 = ax2.imshow(a_mask_svm, cmap=ListedColormap([class_colors[classes[i]] for i in sorted(classes)]))
                ax2.set_title(f"After ({after_date}) - SVM Segmentation")
                ax2.axis('off')
                st.pyplot(fig2)

            st.subheader("Change Detection Analysis")
            total_pixels = b_mask_svm.size
            changed_pixels = np.sum(diff > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            st.metric("Total Area Changed", f"{change_percentage:.2f}%", delta=f"{changed_pixels} pixels changed", delta_color="inverse")

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.imshow(diff, cmap='Reds')
            ax3.set_title("Change Detection Heatmap")
            ax3.axis('off')
            st.pyplot(fig3)

            st.subheader("Class Distribution - Pie Charts")
            def plot_pie(mask, title):
                values, counts = np.unique(mask, return_counts=True)
                labels = [classes[v] for v in values]
                plt.figure(figsize=(4, 4))
                plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=[class_colors[classes[v]] for v in values])
                plt.title(title)
                st.pyplot(plt)

            col5, col6 = st.columns(2)
            with col5:
                plot_pie(b_mask_svm, f"Before - {before_date}")
            with col6:
                plot_pie(a_mask_svm, f"After - {after_date}")

            st.subheader("Class Distribution Table")
            df_data = []
            for class_id in sorted(classes):
                before_pct = np.mean(b_mask_svm == class_id) * 100
                after_pct = np.mean(a_mask_svm == class_id) * 100
                change = after_pct - before_pct
                df_data.append({
                    "Class": classes[class_id],
                    "Before (%)": before_pct,
                    "After (%)": after_pct,
                    "Change (%)": change,
                    "Area (sq km)": change * 0.01 * 100,
                    "Color": class_colors[classes[class_id]]
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df.style.format({
                "Before (%)": "{:.2f}",
                "After (%)": "{:.2f}",
                "Change (%)": "{:+.2f}",
                "Area (sq km)": "{:.2f}"
            }).background_gradient(subset=["Change (%)"], cmap='RdYlGn'))

            st.subheader("Calamity Assessment")
            if "Possible" in calamity_result:
                st.error(f"**Alert:** {calamity_result}")
            else:
                st.success(f"**Status:** {calamity_result}")

        else:
            st.error("Models not found. Please ensure models are in the correct directory.")

        st.button("Restart Analysis", on_click=reset, type="primary")
    else:
        st.warning("Please upload images in the previous steps.")
