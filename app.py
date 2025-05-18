import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib import cm

st.set_page_config(layout="wide")

# --- Load Models with Caching ---
@st.cache_resource
def load_models():
    """Loads the CNN model with caching."""
    cnn_model = None
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        st.info("CNN model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")
    return cnn_model

cnn_model = load_models()

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# --- Navigation Functions ---
def next_page():
    st.session_state.page += 1

def reset():
    st.session_state.page = 1

# --- Image Preprocessing ---
def load_and_preprocess(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)

# --- K-Means Clustering ---
def kmeans_segmentation(image_np, n_clusters=4):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(pixels).reshape(image_np.shape[0], image_np.shape[1])

# --- CNN Prediction ---
def predict_cnn(image_np):
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    return np.argmax(cnn_model.predict(input_array), axis=-1)[0]

# --- Difference Map ---
def difference_heatmap(before_mask, after_mask):
    return (after_mask != before_mask).astype(np.uint8) * 255

# --- Calamity Detection ---
def detect_calamity(date1, date2, mask1, mask2, analysis_type):
    diff_mask = mask2 != mask1
    change_percentage = np.sum(diff_mask) / diff_mask.size
    date_diff = (date2 - date1).days
    
    if analysis_type == "Water Body":
        if date_diff <= 10:
            return "⚠️ Possible Flood (Rapid Change)", change_percentage
        elif date_diff <= 30:
            return "🌊 Possible Waterlogging/Prolonged Flood", change_percentage
        elif date_diff <= 90:
            return "🌦️ Seasonal Variability/Monsoon Shift", change_percentage
        else:
            return "🏙️ Climate Change/Urbanization Impact", change_percentage
    else:  # Land Body
        if date_diff <= 10:
            return "⛰️ Possible Landslide/Earthquake", change_percentage
        elif date_diff <= 30:
            return "🏞️ Soil Erosion/Mudslide", change_percentage
        elif date_diff <= 90:
            return "🌳 Deforestation/Agricultural Changes", change_percentage
        else:
            return "🏜️ Desertification/Urban Expansion", change_percentage

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
        analysis_type = st.session_state.analysis_type

        if cnn_model:
            progress_bar = st.progress(0.0, "Processing Images...")

            # Predict segmentation masks
            with st.spinner("Running CNN analysis..."):
                b_mask_cnn = predict_cnn(b_np)
                a_mask_cnn = predict_cnn(a_np)
                progress_bar.progress(0.5)
            
            with st.spinner("Running K-Means clustering..."):
                b_mask_kmeans = kmeans_segmentation(b_np)
                a_mask_kmeans = kmeans_segmentation(a_np)
                progress_bar.progress(1.0, "Analysis Complete!")
                progress_bar.empty()

            # Generate difference heatmap and detect calamity
            diff = difference_heatmap(b_mask_cnn, a_mask_cnn)
            calamity_result, change_percentage = detect_calamity(
                before_date, after_date, b_mask_cnn, a_mask_cnn, analysis_type
            )

            # Visualization Settings
            class_colors = {
                'Water': '#1f77b4',
                'Vegetation': '#2ca02c',
                'Urban': '#ff7f0e',
                'Barren': '#d62728',
                'Forest': '#9467bd'
            }

            classes = {
                0: "Water",
                1: "Vegetation",
                2: "Urban",
                3: "Barren" if analysis_type == "Water Body" else "Forest"
            }

            # Main Visualization Layout
            st.subheader("Satellite Image Analysis")

            # Original Images Row
            col1, col2 = st.columns(2)
            with col1:
                st.image(b_np, caption=f"Original Before Image ({before_date})", use_container_width=True)
            with col2:
                st.image(a_np, caption=f"Original After Image ({after_date})", use_container_width=True)

            # CNN Segmentation Results
            st.subheader("CNN Segmentation Results")
            col3, col4 = st.columns(2)
            with col3:
                fig1, ax1 = plt.subplots(figsize=(8,6))
                ax1.imshow(b_mask_cnn, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax1.set_title(f"Before Segmentation ({before_date})")
                ax1.axis('off')
                st.pyplot(fig1)
            with col4:
                fig2, ax2 = plt.subplots(figsize=(8,6))
                ax2.imshow(a_mask_cnn, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax2.set_title(f"After Segmentation ({after_date})")
                ax2.axis('off')
                st.pyplot(fig2)

            # K-Means Results with Change Highlighting
            st.subheader("K-Means Change Detection")
            col5, col6 = st.columns(2)
            with col5:
                fig3, ax3 = plt.subplots(figsize=(8,6))
                ax3.imshow(b_mask_kmeans, cmap='gray')
                ax3.set_title(f"Before Clusters ({before_date})")
                ax3.axis('off')
                st.pyplot(fig3)
            
            # Create custom colormap for changes
            change_colors = ['black', 'blue', 'red', 'green']
            with col6:
                fig4, ax4 = plt.subplots(figsize=(8,6))
                change_mask = np.where(a_mask_kmeans != b_mask_kmeans, a_mask_kmeans, 0)
                ax4.imshow(change_mask, cmap=ListedColormap(change_colors))
                ax4.set_title(f"Changes Detected ({after_date})")
                ax4.axis('off')
                st.pyplot(fig4)

            # Change Detection Metrics
            st.subheader("Change Detection Metrics")
            
            # Calculate total area changed
            total_pixels = b_mask_cnn.size
            changed_pixels = np.sum(diff > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Display change metrics
            col7, col8 = st.columns(2)
            with col7:
                st.metric("Total Area Changed", f"{change_percentage:.2f}%", 
                         delta=f"{changed_pixels} pixels changed", delta_color="inverse")
            
            # Class Distribution Analysis
            st.subheader("Land Cover Class Distribution")
            
            # Get class statistics
            data = []
            for class_id in range(4):
                before_pct = (np.sum(b_mask_cnn == class_id) / total_pixels) * 100
                after_pct = (np.sum(a_mask_cnn == class_id) / total_pixels) * 100
                change = after_pct - before_pct
                data.append({
                    "Class": classes[class_id],
                    "Before (%)": before_pct,
                    "After (%)": after_pct,
                    "Change (%)": change,
                    "Area Change (sq km)": change * 0.01 * 100  # Assuming 100 sq km area
                })

            df = pd.DataFrame(data)
            
            # Display detailed statistics
            st.dataframe(df.style.format({
                "Before (%)": "{:.2f}",
                "After (%)": "{:.2f}",
                "Change (%)": "{:+.2f}",
                "Area Change (sq km)": "{:+.2f}"
            }).background_gradient(subset=["Change (%)"], cmap='RdYlGn'))
            
            # Pie Charts Visualization
            st.subheader("Class Distribution Comparison")
            col9, col10 = st.columns(2)
            with col9:
                fig5, ax5 = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = ax5.pie(
                    df["Before (%)"], 
                    labels=df["Class"], 
                    autopct=lambda p: f'{p:.1f}%\n({p*0.01*100:.1f} sq km)',
                    colors=[class_colors[classes[i]] for i in range(4)],
                    startangle=90
                )
                ax5.set_title(f"Before {before_date}")
                ax5.axis('equal')
                st.pyplot(fig5)
                
            with col10:
                fig6, ax6 = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = ax6.pie(
                    df["After (%)"], 
                    labels=df["Class"], 
                    autopct=lambda p: f'{p:.1f}%\n({p*0.01*100:.1f} sq km)',
                    colors=[class_colors[classes[i]] for i in range(4)],
                    startangle=90
                )
                ax6.set_title(f"After {after_date}")
                ax6.axis('equal')
                st.pyplot(fig6)

            # Calamity Analysis Section
            st.subheader("Calamity Assessment")
            if "⚠️" in calamity_result or "⛰️" in calamity_result:
                st.error(f"**Alert:** {calamity_result}")
                st.warning(f"**Change Detected:** {change_percentage:.2f}% of area")
                
                # Detailed impact assessment
                st.markdown("**Potential Impact Assessment**")
                impacts = []
                for class_id in range(4):
                    change = df.iloc[class_id]["Change (%)"]
                    if abs(change) > 5:  # Significant change threshold
                        if change > 0:
                            impacts.append(f"- {classes[class_id]} increased by {change:.2f}%")
                        else:
                            impacts.append(f"- {classes[class_id]} decreased by {-change:.2f}%")
                
                if impacts:
                    st.markdown("\n".join(impacts))
                else:
                    st.info("No significant changes detected in any class")
                    
                st.warning("""
                **Recommended Actions:**
                - Initiate emergency response protocols
                - Dispatch ground verification team
                - Notify local authorities
                - Schedule follow-up imaging
                """)
            else:
                st.success(f"**Status:** {calamity_result}")
                st.info(f"**Change Detected:** {change_percentage:.2f}% of area")
                
                st.info("""
                **Recommended Actions:**
                - Continue routine monitoring
                - Document changes for records
                - Schedule next imaging session
                - Review historical trends
                """)

        else:
            st.error("CNN model not found. Please ensure the model is in the correct directory.")

        st.button("Restart Analysis", on_click=reset, type="primary")
    else:
        st.warning("Please upload images in the previous steps.")
