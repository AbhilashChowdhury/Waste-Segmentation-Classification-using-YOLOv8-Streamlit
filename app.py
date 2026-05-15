import streamlit as st
from PIL import Image
import os
import uuid
import numpy as np
from ultralytics import YOLO

# ===============================
# Directories
# ===============================
UPLOAD_DIR = os.path.join("predicts", "uploaded_images")
ANNOTATED_DIR = os.path.join("predicts", "annotated_images")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# ===============================
# Load YOLO Model
# ===============================
model = YOLO("best.pt")  # make sure best.pt exists

# ===============================
# Session State Initialization
# ===============================
if "run_model" not in st.session_state:
    st.session_state.run_model = False

if "image" not in st.session_state:
    st.session_state.image = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ===============================
# Title
# ===============================
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

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload an image to visualize segmentation and predicted waste types",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# Handle Upload (PREVIEW REMOVED ONLY HERE)
# ===============================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Convert RGBA → RGB if needed
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Reset run state when new image uploaded
    st.session_state.run_model = False
    st.session_state.image = image
    st.session_state.file_name = uploaded_file.name

# ===============================
# Run Button (enabled only after upload)
# ===============================
run_button = st.button(
    "🚀 Run Model",
    disabled=uploaded_file is None
)

if run_button:
    st.session_state.run_model = True

# ===============================
# Processing (ONLY when button clicked)
# ===============================
if st.session_state.run_model and st.session_state.image is not None:

    image = st.session_state.image

    # ===============================
    # Save uploaded image
    # ===============================
    file_ext = st.session_state.file_name.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, unique_filename)

    image.save(uploaded_path)

    # ===============================
    # Run YOLO
    # ===============================
    results = model.predict(uploaded_path)

    # ===============================
    # Extract Labels
    # ===============================
    classes = results[0].names
    labels = []

    if results[0].boxes is not None and results[0].boxes.cls is not None:
        cls_list = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        labels = sorted(set(classes[c] for c in cls_list))

    # ===============================
    # Create Annotated Image
    # ===============================
    annotated_np = results[0].plot()  # BGR format
    annotated_rgb = annotated_np[..., ::-1]  # Convert BGR → RGB
    annotated_img = Image.fromarray(annotated_rgb)

    # ===============================
    # Save Annotated Image
    # ===============================
    annotated_filename = f"annotated_{unique_filename}"
    annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)

    annotated_img.save(annotated_path)

    # ===============================
    # Display Images
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image)

    with col2:
        st.subheader("🖼 Annotated Image")
        st.image(annotated_img)

    # ===============================
    # Predictions
    # ===============================
    st.subheader("🧠 Model Prediction")

    if len(labels) > 0:
        st.success("Detected: " + ", ".join(labels))
    else:
        st.warning("No objects detected.")

    # ===============================
    # Download Button
    # ===============================
    with open(annotated_path, "rb") as file:
        st.download_button(
            label="📥 Download Annotated Image",
            data=file,
            file_name=annotated_filename,
            mime="image/png"
        )
