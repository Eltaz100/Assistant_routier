from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="RoadSign Assistant",
    page_icon="🚦",
    layout="centered",
)

st.title("🚦 RoadSign Assistant")
st.write("Upload an image to detect road traffic signs.")

model_path = Path("models/best.pt")

if not model_path.is_file():
    st.warning(
        "Trained model not found. Add your trained weights to models/best.pt."
    )
    st.stop()

model = YOLO(str(model_path))

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    results = model.predict(image, conf=confidence)

    annotated_image = results[0].plot()
    annotated_image = annotated_image[:, :, ::-1]

    st.image(
        annotated_image,
        caption="Detected traffic signs",
        use_container_width=True,
    )

    detections = results[0].boxes

    st.subheader("Detection results")
    if len(detections) == 0:
        st.info("No traffic signs detected.")
    else:
        for box in detections:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            label = results[0].names[class_id]
            st.write(f"**{label}** — confidence: {score:.1%}")
