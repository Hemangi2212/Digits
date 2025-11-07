"""
app.py — Streamlit Digit Classifier App
--------------------------------------
Predicts hand-drawn digits (0–9) using a Gradient Boosting Classifier model trained on sklearn's `load_digits` dataset.

🧠 Model file: GradBoosting.pkl

How to run:
1. Place `GradBoosting.pkl` in the same folder as this file.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import joblib

st.set_page_config(page_title="Digit Classifier", layout="centered")

# --------------------- Helpers ---------------------
@st.cache_resource
def load_model(path='GradBoosting.pkl'):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Could not load model from '{path}': {e}")
        return None


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('L')  # grayscale
    img_resized = img.resize((8, 8), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(img_resized).astype(np.float32)
    arr_scaled = (1.0 - (arr / 255.0)) * 16.0  # invert + scale to 0..16
    return arr_scaled.flatten().reshape(1, -1)

# --------------------- UI ---------------------
st.title("🖊️ Digit Recognizer — Gradient Boosting Model")
st.markdown(
    "Draw a digit (0–9) in the box and click **Predict** to see what your model thinks!"
)

# Sidebar
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width", 1, 50, 12)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key='canvas'
)

if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Please draw a digit in the canvas first.")
    else:
        # Convert drawing to PIL
        img_data = canvas_result.image_data.astype('uint8')
        pil_img = Image.fromarray(img_data)

        if pil_img.mode == 'RGBA':
            background = Image.new("RGBA", pil_img.size, bg_color)
            pil_img = Image.alpha_composite(background, pil_img).convert('RGB')

        features = preprocess_image(pil_img)

        model = load_model('GradBoosting.pkl')
        if model is not None:
            try:
                pred = model.predict(features)[0]
                st.success(f"Predicted Digit: {int(pred)}")

                try:
                    probs = model.predict_proba(features)[0]
                    top3 = np.argsort(probs)[-3:][::-1]
                    st.write("Top Predictions:")
                    st.table({str(i): float(probs[i]) for i in top3})
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Created by Hema — Powered by Streamlit & Gradient Boosting Classifier ✨")
