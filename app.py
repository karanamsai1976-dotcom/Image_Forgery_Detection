import io

import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"
ELA_QUALITY = 92
IMAGE_SIZE = (128, 128)
CLASS_LABELS = ("Authentic", "Forged")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model(MODEL_PATH, compile=False)


def error_level_analysis(image: Image.Image) -> Image.Image:
    """Recompress the image as JPEG and amplify the pixel differences."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=ELA_QUALITY)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(image, resaved)
    extrema = diff.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    scale = 255.0 / max_diff

    return ImageEnhance.Brightness(diff).enhance(scale)


def predict(image: Image.Image):
    ela_image = error_level_analysis(image)
    arr = np.array(ela_image.resize(IMAGE_SIZE)).astype("float32") / 255.0
    arr = arr.reshape(-1, *IMAGE_SIZE, 3)

    probabilities = get_model().predict(arr, verbose=0)[0]
    label_idx = int(np.argmax(probabilities))
    return CLASS_LABELS[label_idx], float(probabilities[label_idx]), ela_image, probabilities


st.set_page_config(page_title="Image Forgery Detection", page_icon="🔍", layout="centered")

st.title("🔍 Image Forgery Detection")
st.caption("Error Level Analysis (ELA) + CNN, trained on the CASIA2 dataset")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    label, confidence, ela_image, probabilities = predict(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(ela_image, caption="ELA (what the model sees)", use_container_width=True)

    st.subheader("Prediction")
    if label == "Forged":
        st.error(f"**{label}** — {confidence:.1%} confidence")
    else:
        st.success(f"**{label}** — {confidence:.1%} confidence")

    with st.expander("Raw probabilities"):
        st.write({CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))})
else:
    st.info("Upload a JPG or PNG image to check whether it has been digitally tampered with.")
