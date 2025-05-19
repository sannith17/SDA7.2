import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import folium
from streamlit_folium import st_folium
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

st.set_page_config(layout="wide")

# Load or define your models here (dummy placeholders)
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("cnn_model.h5")
    except Exception:
        model = None
    return model

@st.cache_resource
def load_svm_model():
    # Load your actual trained SVM model here (sklearn)
    # Dummy example: model = joblib.load("svm_model.pkl")
    model = SVC(probability=True)
    return model

cnn_model = load_cnn_model()
svm_model = load_svm_model()

def align_images(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = img2.shape[:2]
        aligned_img = cv2.warpPerspective(img1, H, (width, height))
        return aligned_img, H
    else:
        st.warning("Not enough matches found for alignment.")
        return img1, None

def crop_black_borders(img):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img[y:y+h, x:x+w]
        return cropped
    else:
        return img

def plot_cnn_prediction(pred_probs, class_labels):
    fig, ax = plt.subplots()
    ax.bar(class_labels, pred_probs)
    ax.set_ylabel("Probability")
    ax.set_ylim([0,1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_heatmap(img1, img2):
    # Simple absolute difference heatmap
    diff = cv2.absdiff(img1, img2)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return heatmap

def create_geospatial_map():
    m = folium.Map(location=[20, 0], zoom_start=2)
    folium.TileLayer('OpenStreetMap').add_to(m)
    # Example polygon overlays:
    folium.GeoJson(data={
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"class": "Vegetation"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}},
            {"type": "Feature",
             "properties": {"class": "Water"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[-10, -10], [-5, -10], [-5, -5], [-10, -5], [-10, -10]]]}}
        ]
    }, style_function=lambda x: {
        'fillColor': {'Vegetation': 'green', 'Water': 'blue', 'Urban': 'gray', 'Land': 'brown'}.get(x['properties']['class'], 'black'),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    }).add_to(m)
    return m

def image_to_bytes(img):
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buffer)
    return io_buf

# Page 1: Upload and select model
def page1():
    st.title("Satellite Data Analysis - Model Selection and Input")
    st.write("Upload satellite images and choose your model.")
    uploaded_before = st.file_uploader("Upload BEFORE Image", type=['png','jpg','jpeg'])
    uploaded_after = st.file_uploader("Upload AFTER Image", type=['png','jpg','jpeg'])

    model_choice = st.selectbox("Select Model", ["SVM", "CNN", "KMeans"])

    if uploaded_before and uploaded_after:
        img_before = np.array(Image.open(uploaded_before).convert('RGB'))
        img_after = np.array(Image.open(uploaded_after).convert('RGB'))

        st.image([img_before, img_after], caption=["Before", "After"], width=300)

        if st.button("Run Analysis"):
            st.session_state['img_before'] = img_before
            st.session_state['img_after'] = img_after
            st.session_state['model_choice'] = model_choice
            st.success("Images and model saved. Go to next pages for results.")

# Page 2: Run detection & CNN visualization & download reports
def page2():
    st.title("Detection & Model Prediction")
    if 'img_before' not in st.session_state or 'img_after' not in st.session_state or 'model_choice' not in st.session_state:
        st.warning("Upload images and select model on Page 1 first.")
        return

    img_before = st.session_state['img_before']
    img_after = st.session_state['img_after']
    model_choice = st.session_state['model_choice']

    st.subheader("Model Selected: " + model_choice)

    # Dummy detection logic (Replace with real models)
    def dummy_detect(img):
        # Threshold on blue channel for water detection (simple)
        blue = img[:,:,2]
        water_mask = (blue > 100).astype(np.uint8)*255
        return water_mask

    if model_choice == "SVM":
        # Dummy SVM prediction: random class probabilities
        classes = ["Water", "Land", "Urban", "Vegetation"]
        pred_probs = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
        detection_mask = dummy_detect(img_after)
    elif model_choice == "CNN" and cnn_model is not None:
        # Preprocess image for CNN (assuming classification)
        img_resized = cv2.resize(img_after, (64,64))
        x = img_to_array(img_resized)/255.0
        x = np.expand_dims(x, axis=0)
        preds = cnn_model.predict(x)[0]
        classes = ["Water", "Land", "Urban", "Vegetation"]
        pred_probs = preds
        detection_mask = dummy_detect(img_after)
    elif model_choice == "KMeans":
        # Dummy kmeans clustering to segment image
        pixel_values = img_after.reshape((-1,3))
        kmeans = KMeans(n_clusters=4, random_state=42).fit(pixel_values)
        labels = kmeans.labels_.reshape(img_after.shape[:2])
        detection_mask = (labels==0).astype(np.uint8)*255
        classes = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"]
        pred_probs = [1.0/4]*4
    else:
        st.error("CNN model not loaded or invalid selection.")
        return

    st.image(detection_mask, caption="Detection Mask", width=400)

    if model_choice in ["SVM","CNN"]:
        st.subheader("Model Prediction Probabilities")
        fig = plot_cnn_prediction(pred_probs, classes)
        st.pyplot(fig)

    # Download buttons
    csv_buf = io.StringIO()
    df = pd.DataFrame({"Class": classes, "Probability": pred_probs})
    df.to_csv(csv_buf, index=False)
    st.download_button("Download CSV Summary", data=csv_buf.getvalue(), file_name="summary.csv", mime="text/csv")

    annotated_img_buf = image_to_bytes(detection_mask)
    st.download_button("Download Detection Mask Image", data=annotated_img_buf, file_name="detection_mask.png", mime="image/png")

# Page 3: Image Alignment, cropping and side-by-side comparison
def page3():
    st.title("Image Alignment and Comparison")

    if 'img_before' not in st.session_state or 'img_after' not in st.session_state:
        st.warning("Upload images on Page 1 first.")
        return

    img_before = st.session_state['img_before']
    img_after = st.session_state['img_after']

    st.subheader("Original Reference Image (After)")
    st.image(img_after, width=350)

    st.subheader("Aligning 'Before' image to 'After' image...")
    aligned_img, H = align_images(img_before, img_after)

    if H is None:
        st.warning("Alignment failed; displaying original before image.")
        aligned_img = img_before

    cropped_aligned_img = crop_black_borders(aligned_img)

    st.subheader("Aligned and Cropped Image")
    st.image(cropped_aligned_img, width=350)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_after, caption="Reference Image (After)", width=350)
    with col2:
        st.image(cropped_aligned_img, caption="Aligned & Cropped Before Image", width=350)

    # Side-by-side difference heatmap
    try:
        cropped_aligned_resized = cv2.resize(cropped_aligned_img, (img_after.shape[1], img_after.shape[0]))
        heatmap = generate_heatmap(cropped_aligned_resized, img_after)
        st.subheader("Difference Heatmap (Aligned vs Reference)")
        st.image(heatmap, width=700)
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")

# Page 4: Map classification (vegetation, land, urban, water only)
def page4():
    st.title("Geospatial Map and Land Classification")

    if 'img_after' not in st.session_state:
        st.warning("Upload images on Page 1 first.")
        return

    img_after = st.session_state['img_after']

    # Dummy classification mask: color-coded map
    h, w = img_after.shape[:2]
    class_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Fake segmentation: thresholds for demonstration
    blue = img_after[:,:,2]
    green = img_after[:,:,1]
    red = img_after[:,:,0]

    # Water: high blue & low red/green
    water_mask = (blue > 120) & (green < 100) & (red < 100)
    # Vegetation: high green, moderate red and blue
    vegetation_mask = (green > 120) & (red < 100) & (blue < 100)
    # Urban: high red & green, moderate blue
    urban_mask = (red > 130) & (green > 130) & (blue < 100)
    # Land (others)
    land_mask = ~(water_mask | vegetation_mask | urban_mask)

    class_map[water_mask] = [0, 0, 255]       # Blue
    class_map[vegetation_mask] = [0, 255, 0]  # Green
    class_map[urban_mask] = [128, 128, 128]   # Gray
    class_map[land_mask] = [165, 42, 42]      # Brown

    st.image(class_map, caption="Classified Map (Water, Vegetation, Urban, Land)", width=600)

    st.subheader("Geospatial Overlay Map")
    m = create_geospatial_map()
    st_folium(m, width=700, height=450)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Model Select", "Detection & Prediction", "Image Alignment", "Map Classification"])

    if page == "Upload & Model Select":
        page1()
    elif page == "Detection & Prediction":
        page2()
    elif page == "Image Alignment":
        page3()
    elif page == "Map Classification":
        page4()

if __name__ == "__main__":
    main()
