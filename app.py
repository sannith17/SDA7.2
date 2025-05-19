import streamlit as st
import numpy as np
import cv2
import datetime
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dummy CNN and SVM models
def dummy_svm_model(img_array):
    return np.random.choice([0, 1])

def dummy_cnn_model(img_array):
    classes = ['Water', 'Vegetation', 'Urban', 'Land']
    output = np.random.choice(classes, size=(100, 100), p=[0.2, 0.3, 0.2, 0.3])
    return output

# Alignment using ECC
def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    try:
        cc, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        diff_mask = cv2.absdiff(img1, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)
        return aligned, aligned_black
    except Exception as e:
        st.error(f"Image alignment failed: {e}")
        return img2, img2

# Classification visualization
def get_classification_map(cnn_output):
    color_map = {
        'Water': (0, 0, 255),
        'Vegetation': (0, 255, 0),
        'Urban': (128, 128, 128),
        'Land': (210, 180, 140)
    }
    heatmap = np.zeros((cnn_output.shape[0], cnn_output.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        heatmap[cnn_output == cls] = color
    return heatmap

def classification_stats(cnn_output):
    flat = cnn_output.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    stats = dict(zip(unique, counts))
    df = pd.DataFrame({'Class': list(stats.keys()), 'Count': list(stats.values())})
    return df

# Download helper
def get_download_link(img, filename, label):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'

# Streamlit UI
st.set_page_config(page_title="Satellite Analysis App", layout="wide")
st.title("🌍 Satellite Data Change Detection Dashboard")

# Page Selector
page = st.sidebar.selectbox("Navigate Pages", ["1. Choose Model", "2. Upload Images", "3. Align Images", "4. Output & Analysis"])

# Session variables
if "model_choice" not in st.session_state: st.session_state.model_choice = "SVM-KMeans"
if "before_img" not in st.session_state: st.session_state.before_img = None
if "after_img" not in st.session_state: st.session_state.after_img = None
if "aligned_img" not in st.session_state: st.session_state.aligned_img = None
if "aligned_black" not in st.session_state: st.session_state.aligned_black = None
if "cnn_output" not in st.session_state: st.session_state.cnn_output = None

# Page 1: Choose Model
if page.startswith("1"):
    st.header("🔍 Select Detection Model")
    st.session_state.model_choice = st.radio("Choose Model Type:", ["SVM-KMeans", "SVM-CNN"])
    st.success(f"✅ You selected: {st.session_state.model_choice}")

# Page 2: Upload
elif page.startswith("2"):
    st.header("📤 Upload Satellite Images")
    before_file = st.file_uploader("Upload BEFORE image", type=["png", "jpg", "jpeg"], key="before")
    after_file = st.file_uploader("Upload AFTER image", type=["png", "jpg", "jpeg"], key="after")
    date_upload = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if before_file and after_file:
        st.session_state.before_img = np.array(Image.open(before_file).convert("RGB"))
        st.session_state.after_img = np.array(Image.open(after_file).convert("RGB"))
        st.success(f"🗓️ Images uploaded successfully on {date_upload}")

# Page 3: Alignment
elif page.startswith("3"):
    st.header("🧭 Image Alignment")
    if st.session_state.before_img is not None and st.session_state.after_img is not None:
        aligned, black_output = align_images(st.session_state.before_img, st.session_state.after_img)
        st.session_state.aligned_img = aligned
        st.session_state.aligned_black = black_output
        col1, col2, col3 = st.columns(3)
        col1.image(st.session_state.before_img, caption="Before")
        col2.image(st.session_state.after_img, caption="After")
        col3.image(st.session_state.aligned_black, caption="Aligned Output (Black background)")
    else:
        st.warning("Please upload images first.")

# Page 4: Output
elif page.startswith("4"):
    st.header("📊 Output Analysis & Calamity Detection")
    if st.session_state.aligned_black is not None:
        model_type = st.session_state.model_choice
        # Run CNN if selected
        if "CNN" in model_type:
            cnn_result = dummy_cnn_model(st.session_state.aligned_black)
            st.session_state.cnn_output = cnn_result
            class_map_img = get_classification_map(cnn_result)
            df_stats = classification_stats(cnn_result)
            st.subheader("🧭 Classified Image (After)")
            st.image(class_map_img, caption="CNN Output - Classified")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📋 Land Class Table")
                st.dataframe(df_stats)
            with col2:
                st.subheader("📊 Pie Chart")
                fig, ax = plt.subplots()
                ax.pie(df_stats["Count"], labels=df_stats["Class"], autopct="%1.1f%%", startangle=90)
                st.pyplot(fig)

            # Heatmap
            st.subheader("🔥 Heatmap (Change Areas Only)")
            heatmap = np.zeros_like(cnn_result, dtype=np.uint8)
            heatmap[np.random.rand(*cnn_result.shape) > 0.85] = 255  # dummy change mask
            fig2, ax2 = plt.subplots()
            sns.heatmap(heatmap, cbar=False, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

            # Calamity possibility
            total_water = df_stats[df_stats["Class"] == "Water"]["Count"].values[0] if "Water" in df_stats["Class"].values else 0
            water_change = np.random.randint(5, 30)
            st.markdown("### ⚠️ Calamity Possibility")
            if water_change > 20:
                st.error("Flooding Detected: High Water Body Increase")
            else:
                st.success("No Major Calamity Detected")

            # Download
            st.markdown("### ⬇️ Downloads")
            result_img = Image.fromarray(class_map_img)
            download_img = get_download_link(result_img, "classified_output.png", "📥 Download Classified Image")
            download_csv = df_stats.to_csv(index=False).encode("utf-8")
            b64_csv = base64.b64encode(download_csv).decode()
            download_csv_link = f'<a href="data:file/csv;base64,{b64_csv}" download="classification_stats.csv">📥 Download Table (CSV)</a>'
            st.markdown(download_img, unsafe_allow_html=True)
            st.markdown(download_csv_link, unsafe_allow_html=True)
        else:
            st.warning("Only CNN-based analysis supported in current demo for full analysis.")
    else:
        st.warning("Please align images first.")
