import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io
import datetime
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from tensorflow.keras.models import load_model
import joblib

cnn_model = load_model("cnn_model.h5")
svm_model = joblib.load("svm_model.pkl")


st.set_page_config(layout="wide")

# Persistent states
if "page" not in st.session_state:
    st.session_state.page = 1
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None
if "before_image" not in st.session_state:
    st.session_state.before_image = None
if "after_image" not in st.session_state:
    st.session_state.after_image = None
if "aligned_before" not in st.session_state:
    st.session_state.aligned_before = None
if "aligned_after" not in st.session_state:
    st.session_state.aligned_after = None

# Page Navigation
def next_page():
    st.session_state.page += 1

# PAGE 1: Model Selection
if st.session_state.page == 1:
    st.title("Satellite Image Analysis - Model Selection")
    st.session_state.model_choice = st.selectbox("Choose a model combination", ["CNN-SVM", "SVM-KMeans"])
    if st.button("Next"):
        next_page()

# PAGE 2: Upload Images
elif st.session_state.page == 2:
    st.title("Upload Satellite Images")
    before = st.file_uploader("Upload 'Before' Image", type=["jpg", "png", "jpeg"], key="before")
    after = st.file_uploader("Upload 'After' Image", type=["jpg", "png", "jpeg"], key="after")

    if before and after:
        before_img = Image.open(before)
        after_img = Image.open(after)
        st.session_state.before_image = before_img
        st.session_state.after_image = after_img
        today = datetime.date.today()
        st.write(f"Images uploaded on: **{today.strftime('%B %d, %Y')}**")
        st.image([before_img, after_img], caption=["Before Image", "After Image"], width=300)

    if st.button("Next"):
        next_page()

# PAGE 3: Image Alignment
elif st.session_state.page == 3:
    st.title("Align Images")
    def align_images(img1, img2):
        img1 = np.array(img1.convert("L"))
        img2 = np.array(img2.convert("L"))

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(np.array(st.session_state.after_image), M, (img1.shape[1], img1.shape[0]))

        return st.session_state.before_image, Image.fromarray(aligned)

    if st.session_state.before_image and st.session_state.after_image:
        b, a = align_images(st.session_state.before_image, st.session_state.after_image)
        st.session_state.aligned_before = b
        st.session_state.aligned_after = a
        st.image([b, a], caption=["Aligned Before", "Aligned After"], width=300)
    if st.button("Next"):
        next_page()

# PAGE 4: Output Analysis
elif st.session_state.page == 4:
    st.title("Land Cover Classification and Change Detection")

    def classify_image_kmeans(img, n_clusters=3):
        img_np = np.array(img.resize((256, 256)))
        X = img_np.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        segmented = kmeans.labels_.reshape(256, 256)
        return segmented

    def classify_image_svm(img):
        img_np = np.array(img.resize((256, 256)))
        flat = img_np.reshape(-1, 3)
        labels = svm_model.predict(flat)
        return labels.reshape(256, 256)

    model_choice = st.session_state.model_choice
    after_img = st.session_state.aligned_after
    before_img = st.session_state.aligned_before

    if model_choice == "CNN-SVM":
        result_before = classify_image_svm(before_img)
        result_after = classify_image_svm(after_img)
    else:  # SVM-KMeans
        result_before = classify_image_kmeans(before_img)
        result_after = classify_image_kmeans(after_img)

    changed_mask = result_before != result_after
    total_pixels = changed_mask.size
    changed_pixels = np.sum(changed_mask)
    changed_percent = round((changed_pixels / total_pixels) * 100, 2)

    # Show "After" image with only changed areas colored
    color_img = np.array(after_img.resize((256, 256))).copy()
    mask_3d = np.stack([changed_mask]*3, axis=-1)
    color_img[~mask_3d] = 0

    st.subheader("Changed Areas (After Image)")
    st.image(color_img, caption=f"Changed areas highlighted ({changed_percent}%)", use_column_width=True)

    # Piechart and Table
    unique, counts = np.unique(result_after, return_counts=True)
    labels_map = {0: "Water", 1: "Vegetation", 2: "Land"}
    filtered = {labels_map[k]: v for k, v in zip(unique, counts) if k in labels_map}

    fig, ax = plt.subplots()
    ax.pie(filtered.values(), labels=filtered.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    st.pyplot(fig)

    st.subheader("Area Distribution")
    table_data = [{"Class": k, "Pixels": v, "Percentage": f"{(v / total_pixels) * 100:.2f}%"} for k, v in filtered.items()]
    st.table(table_data)

    st.markdown(f"### Calamity Possibility: {'⚠️ Possible Calamity Detected' if changed_percent > 30 else '✅ No Major Change Detected'}")

    # Download CSV Report
    csv_str = "Class,Pixels,Percentage\n" + "\n".join([f"{row['Class']},{row['Pixels']},{row['Percentage']}" for row in table_data])
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="classification_report.csv">📄 Download CSV Report</a>'
    st.markdown(href, unsafe_allow_html=True)
