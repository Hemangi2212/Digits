import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import joblib

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# ---------------- Load model ----------------
@st.cache_resource
def load_model(path='GradBoosting.pkl'):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


# ---------------- Preprocessing ----------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('L')                   # grayscale
    img = img.resize((8, 8))                 # match sklearn digits size
    arr = np.asarray(img).astype(np.float32)

    # normalize: invert and scale to 0..16
    arr = (255 - arr) / 255 * 16
    return arr.flatten().reshape(1, -1)


# ---------------- UI ----------------
st.title("✏️ Digit Recognizer")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("**Draw a digit (0–9) and click Predict.**")
with col2:
    clear_canvas = st.button("🧹 Clear")

# Canvas area
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode='freedraw',
    key='canvas',
    update_streamlit=clear_canvas  # resets on clear
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        img_data = canvas_result.image_data.astype("uint8")
        pil_img = Image.fromarray(img_data)

        # preprocess
        features = preprocess_image(pil_img)

        # predict
        model = load_model('GradBoosting.pkl')
        if model:
            try:
                pred = model.predict(features)[0]
                st.markdown(f"<h2 style='text-align:center; color:#00C851;'>Predicted Digit: {int(pred)}</h2>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
