import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "model/my_model.pt"
CONF_THRESH = 0.5
# ---------------------------------------

st.set_page_config(
    page_title="Candy Detection App",
    page_icon="🍫",
    layout="centered"
)

st.title("🍬 Candy Detection using YOLO")
st.write("Upload an image and the model will detect and label all candies.")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
class_names = model.names

uploaded_file = st.file_uploader(
    "Upload a candy image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    with st.spinner("Detecting candies..."):
        results = model(img_np, conf=CONF_THRESH, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls.item())
            conf = box.conf.item()

            label = f"{class_names[cls_id]} ({conf:.2f})"

            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_np,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        st.subheader("Predicted Image")
        st.image(img_np, use_container_width=True)

        st.success(f"Detected {len(results.boxes)} candy objects 🎉")

    else:
        st.warning("No candies detected in this image.")

st.markdown("---")
st.markdown(
    "<center>YOLO Candy Detection App • Built with Streamlit</center>",
    unsafe_allow_html=True
)
