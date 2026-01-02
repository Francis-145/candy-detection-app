import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

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

# Load YOLO model (cached so it loads only once)
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
class_names = model.names

# Upload image
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

    # Copy image for drawing
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls.item())
            conf = box.conf.item()

            label = f"{class_names[cls_id]} ({conf:.2f})"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, max(y1 - 15, 10)), label, fill="green")

        st.subheader("Predicted Image")
        st.image(output_image, use_container_width=True)

        st.success(f"Detected {len(results.boxes)} candy objects 🎉")

    else:
        st.warning("No candies detected in this image.")

st.markdown("---")
st.markdown(
    "<center>YOLO Candy Detection App • Built with Streamlit</center>",
    unsafe_allow_html=True
)
