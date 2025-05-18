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
def load_model():
    """Loads the CNN model with caching."""
    try:
        model = tf.keras.models.load_model("cnn_model.h5")
        st.info("CNN model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")
        return None

cnn_model = load_model()

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

# --- K-Means Clustering ---
def kmeans_segmentation(image_np, n_clusters=4):
    """Performs K-Means clustering for segmentation."""
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image_np.shape[0], image_np.shape[1])

# --- CNN Prediction ---
def predict_cnn(image_np):
    """Predicts using the CNN model."""
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array, verbose=0)
    return np.argmax(predictions, axis=-1)[0]

# --- Difference Map ---
def difference_heatmap(before_mask, after_mask):
    """Generates a heatmap showing the difference between two masks."""
    diff = after_mask != before_mask
    return diff.astype(np.uint8) * 255

# --- Calamity Detection ---
def detect_calamity(date1, date2, mask1, mask2, analysis_type):
    """Detects potential calamities based on changes in the segmentation masks."""
    diff_mask = mask2 != mask1
    change_percentage = np.sum(diff_mask) / diff_mask.size
    date_diff = (date2 - date1).days
    
    if analysis_type == "Water Body":
        if date_diff <= 7:
            return "⚠️ Possible Flood (Flash flood, River flood)"
        elif date_diff <= 30:
            return "⚠️ Waterlogging or Prolonged River Flood"
        elif date_diff <= 90:
            return "⚠️ Seasonal Variability / Drought / Monsoon Shift"
        else:
            return "⚠️ Climate Change / Urbanization Impact"
    else:  # Land Body
        if date_diff <= 7:
            return "⚠️ Possible Landslide or Earthquake"
        elif date_diff <= 30:
            return "⚠️ Soil Erosion or Mudslide"
        elif date_diff <= 90:
            return "⚠️ Deforestation or Agricultural Changes"
        else:
            return "⚠️ Desertification or Urban Expansion"

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
                progress_bar.progress(0.5)
                a_mask_cnn = predict_cnn(a_np)
                progress_bar.progress(1.0)

            # Generate difference heatmap and detect calamity
            diff = difference_heatmap(b_mask_cnn, a_mask_cnn)
            calamity_result = detect_calamity(before_date, after_date, b_mask_cnn, a_mask_cnn, analysis_type)
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
            if analysis_type == "Water Body":
                classes = {0: "Water", 1: "Vegetation", 2: "Urban", 3: "Barren"}
            else:
                classes = {0: "Forest", 1: "Vegetation", 2: "Urban", 3: "Water"}

            # Main Visualization Layout
            st.subheader("Satellite Image Analysis")

            # Original Images Row
            col1, col2 = st.columns(2)
            with col1:
                st.image(b_np, caption=f"Original Before Image ({before_date})", use_container_width=True)
            with col2:
                st.image(a_np, caption=f"Original After Image ({after_date})", use_container_width=True)

            # CNN Results
            st.subheader("CNN Segmentation Results")
            col3, col4 = st.columns(2)
            with col3:
                fig1, ax1 = plt.subplots(figsize=(8,6))
                ax1.imshow(b_mask_cnn, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax1.set_title(f"Before ({before_date})")
                ax1.axis('off')
                st.pyplot(fig1)
            with col4:
                fig2, ax2 = plt.subplots(figsize=(8,6))
                ax2.imshow(a_mask_cnn, cmap=ListedColormap([class_colors[classes[i]] for i in range(4)]))
                ax2.set_title(f"After ({after_date})")
                ax2.axis('off')
                st.pyplot(fig2)

            # K-Means Results (After image only with change highlights)
            st.subheader("Change Highlights (K-Means)")
            a_mask_kmeans = kmeans_segmentation(a_np)
            
            # Create custom colormap for changes
            change_colors = ['black', 'blue', 'red', 'green']  # Black=no change, Blue=water, Red=land, Green=vegetation
            change_cmap = ListedColormap(change_colors)
            
            # Create change mask (simple version)
            change_mask = np.zeros_like(a_mask_kmeans)
            for i in range(4):
                class_change = (a_mask_cnn == i) & (b_mask_cnn != i)
                change_mask[class_change] = i + 1  # 0=no change, 1=water, 2=urban/barren, 3=vegetation
            
            fig3, ax3 = plt.subplots(figsize=(8,6))
            ax3.imshow(change_mask, cmap=change_cmap, vmin=0, vmax=3)
            ax3.set_title(f"Change Highlights ({after_date})")
            ax3.axis('off')
            
            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Water Increase'),
                Patch(facecolor='red', label='Land Change'),
                Patch(facecolor='green', label='Vegetation Increase')
            ]
            ax3.legend(handles=legend_elements, loc='lower right')
            
            st.pyplot(fig3)

            # Change Detection Visualization
            st.subheader("Change Detection Analysis")
            
            # Calculate total area changed
            total_pixels = b_mask_cnn.size
            changed_pixels = np.sum(diff > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Display change metrics
            st.metric("Total Area Changed", f"{change_percentage:.2f}%", 
                     delta=f"{changed_pixels} pixels changed", delta_color="inverse")

            # Class Distribution Analysis
            st.subheader("Land Cover Class Distribution")
            
            # Get class statistics
            unique_b, count_b = np.unique(b_mask_cnn, return_counts=True)
            unique_a, count_a = np.unique(a_mask_cnn, return_counts=True)

            # Create percentage data
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
                    "Area Change (sq km)": (after_pct - before_pct) * 0.01 * 100  # Assuming 100 sq km area
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
            col5, col6 = st.columns(2)
            with col5:
                fig4, ax4 = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = ax4.pie(
                    df["Before (%)"], 
                    labels=df["Class"], 
                    autopct=lambda p: f'{p:.1f}%\n({p*0.01*100:.1f} sq km)',
                    colors=[class_colors[classes[i]] for i in range(4)],
                    startangle=90
                )
                ax4.set_title(f"Before {before_date}")
                ax4.axis('equal')
                st.pyplot(fig4)
                
            with col6:
                fig5, ax5 = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = ax5.pie(
                    df["After (%)"], 
                    labels=df["Class"], 
                    autopct=lambda p: f'{p:.1f}%\n({p*0.01*100:.1f} sq km)',
                    colors=[class_colors[classes[i]] for i in range(4)],
                    startangle=90
                )
                ax5.set_title(f"After {after_date}")
                ax5.axis('equal')
                st.pyplot(fig5)

            # Calamity Analysis Section
            st.subheader("Calamity Assessment")
            st.warning(calamity_result)
            
            # Detailed impact assessment
            st.markdown("**Potential Impact Assessment**")
            impacts = []
            for class_id in range(4):
                change = df.iloc[class_id]["Change (%)"]
                if analysis_type == "Water Body":
                    if classes[class_id] == "Water" and change > 5:
                        impacts.append(f"- Water bodies increased by {change:.2f}% (potential flood risk)")
                    elif classes[class_id] == "Water" and change < -5:
                        impacts.append(f"- Water bodies decreased by {-change:.2f}% (potential drought risk)")
                else:
                    if classes[class_id] == "Vegetation" and change < -5:
                        impacts.append(f"- Vegetation decreased by {-change:.2f}% (potential deforestation)")
                    elif classes[class_id] == "Urban" and change > 5:
                        impacts.append(f"- Urban area increased by {change:.2f}% (urban expansion)")
            
            if impacts:
                st.markdown("\n".join(impacts))
            else:
                st.info("No significant changes detected that would indicate immediate risk")

            # Recommendations based on analysis
            st.markdown("**Recommended Actions**")
            if "Flood" in calamity_result:
                st.markdown("""
                - Activate flood warning systems
                - Prepare emergency shelters
                - Monitor river levels hourly
                """)
            elif "Drought" in calamity_result:
                st.markdown("""
                - Implement water conservation measures
                - Monitor reservoir levels
                - Prepare drought contingency plans
                """)
            elif "Deforestation" in calamity_result:
                st.markdown("""
                - Dispatch forest rangers for verification
                - Check for illegal logging activity
                - Initiate reforestation planning
                """)
            else:
                st.markdown("""
                - Continue routine monitoring
                - Document changes for long-term analysis
                - Schedule next imaging session
                """)

        else:
            st.error("CNN model not found. Please ensure the model is in the correct directory.")

        st.button("Restart Analysis", on_click=reset, type="primary")
    else:
        st.warning("Please upload images in the previous steps.")
