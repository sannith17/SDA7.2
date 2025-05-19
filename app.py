import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load Models
@st.cache_resource
def load_svm_model():
    return SVC(probability=True)

@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("cnn_model.h5")
    return model

def preprocess_image(img, size=(64, 64)):
    img = cv2.resize(img, size)
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    return np.expand_dims(img, axis=0)

def apply_pca(image, n_components=3):
    reshaped = image.reshape(-1, 3)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)
    reconstructed = pca.inverse_transform(reduced).reshape(image.shape).astype(np.uint8)
    return reconstructed

def align_images(img1, img2):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) > 4:
        src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        h, w = img2.shape[:2]
        aligned = cv2.warpPerspective(img1, M, (w, h))
        mask = cv2.warpPerspective(np.ones_like(img1), M, (w, h))
        black_bg = np.zeros_like(img2)
        output = np.where(mask == 0, black_bg, aligned)
        return output
    return img1

# PAGE 1: Upload
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "PCA + Model Detection", "Alignment", "Classification Map"])

if page == "Upload":
    st.title("Upload Satellite Images")
    uploaded_before = st.file_uploader("Upload Before Image", type=["jpg", "png", "jpeg"], key="before")
    uploaded_after = st.file_uploader("Upload After Image", type=["jpg", "png", "jpeg"], key="after")

    if uploaded_before and uploaded_after:
        image1 = Image.open(uploaded_before)
        image2 = Image.open(uploaded_after)
        st.image([image1, image2], caption=["Before", "After"], width=300)
        image1.save("before.png")
        image2.save("after.png")

elif page == "PCA + Model Detection":
    st.title("PCA + Detection Models")
    model_choice = st.selectbox("Choose Model", ["SVM", "CNN"])
    image = cv2.imread("after.png")
    reduced_img = apply_pca(image)

    if model_choice == "SVM":
        svm = load_svm_model()
        dummy_data = np.random.rand(100, 3)
        dummy_labels = np.random.randint(0, 2, 100)
        svm.fit(dummy_data, dummy_labels)
        reshaped = reduced_img.reshape(-1, 3)
        prediction = svm.predict(reshaped).reshape(image.shape[:2])
        st.image(prediction * 255, caption="SVM Prediction", clamp=True)
    else:
        model = load_cnn_model()
        processed = preprocess_image(image)
        pred = model.predict(processed)[0]
        pred_class = np.argmax(pred)
        st.write("CNN Prediction Class:", pred_class)
        st.bar_chart(pred)

elif page == "Alignment":
    st.title("Image Alignment and Crop")
    img1 = cv2.imread("before.png")
    img2 = cv2.imread("after.png")
    aligned = align_images(img1, img2)
    st.image([img1, img2, aligned], caption=["Before", "After", "Aligned & Cropped"], width=300)

elif page == "Classification Map":
    st.title("Land Type Classification")

    image = cv2.imread("after.png")
    reshaped = image.reshape(-1, 3)
    kmeans = RandomForestClassifier()
    dummy_labels = np.random.randint(0, 4, reshaped.shape[0])
    kmeans.fit(reshaped, dummy_labels)
    labels = kmeans.predict(reshaped).reshape(image.shape[:2])

    cmap = {
        0: (0, 255, 0),     # Vegetation
        1: (128, 128, 128), # Urban
        2: (255, 255, 0),   # Land
        3: (0, 0, 255)      # Water
    }

    colored_map = np.zeros_like(image)
    for label, color in cmap.items():
        colored_map[labels == label] = color

    st.image(colored_map, caption="Land Classification")

    st.download_button("Download Labeled Image", data=cv2.imencode('.png', colored_map)[1].tobytes(),
                       file_name="classified_output.png")

    df = pd.DataFrame({"class": labels.flatten()})
    csv = df.value_counts().reset_index()
    st.dataframe(csv)
    st.download_button("Download Report", csv.to_csv(index=False), "report.csv")

    if st.checkbox("Show Geospatial Overlay"):
        m = folium.Map(location=[0, 0], zoom_start=2)
        folium.Marker(location=[0, 0], popup="Sample Overlay").add_to(m)
        st_folium(m, width=700)