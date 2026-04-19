import streamlit as st
from PIL import Image
import os
import uuid
import numpy as np
from ultralytics import YOLO

# Directories (optional: keep upload folder for record)
UPLOAD_DIR = os.path.join("predicts", "uploaded_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load YOLO segmentation model
model = YOLO("best.pt")  # make sure best.pt is in same folder (or give full path)

# Streamlit title
st.markdown("""
    <h1 style="
        color: white; 
        text-align: center; 
        font-size: 2.2rem; 
        font-weight: 700; 
        margin-bottom: 20px;
    ">
        ♻️ Waste Segmentation & Classification using YOLO26
    </h1>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload an image to visualize segmentation and predicted waste types",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Save uploaded image (optional)
    file_ext = uploaded_file.name.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, unique_filename)
    image.save(uploaded_path)

    # Run YOLO segmentation
    # NOTE: We do NOT rely on saved annotated files now.
    results = model.predict(uploaded_path)

    # Get labels safely
    classes = results[0].names
    labels = []
    if results[0].boxes is not None and results[0].boxes.cls is not None:
        cls_list = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        labels = sorted(set(classes[c] for c in cls_list))

    # Create annotated image (in-memory)
    annotated_np = results[0].plot()  # returns numpy array in BGR
    annotated_rgb = annotated_np[..., ::-1]  # BGR -> RGB
    annotated_img = Image.fromarray(annotated_rgb)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("🖼 Segmented Output")
        st.image(annotated_img, caption="YOLO Segmentation Output", use_container_width=True)

    # Prediction result below both images
    st.subheader("🧠 Model Prediction")
    if len(labels) > 0:
        st.success("Detected: " + ", ".join(labels))
    else:
        st.warning("No objects detected.")
