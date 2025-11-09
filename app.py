 # app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# ---------- Load model ----------
@st.cache_resource
def load_model(path="GradBoosting.pkl"):
    return joblib.load(path)

model = load_model()

# ---------- Preprocessing helper ----------
def preprocess_for_digits(pil_img: Image.Image):
    img = pil_img.convert("L")
    arr = np.asarray(img)

    thresh = 50
    fg_mask = (arr > thresh) & (arr < 255 - thresh)
    if not fg_mask.any():
        fg_mask = arr > (np.mean(arr) + 10)

    coords = np.argwhere(fg_mask)
    if coords.size == 0:
        empty = np.zeros((8, 8), dtype=np.float32)
        return empty.flatten().reshape(1, -1), Image.fromarray((empty / 16.0 * 255).astype(np.uint8))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((x0, y0, x1, y1))

    w, h = cropped.size
    max_side = max(w, h)
    pad_x = (max_side - w) // 2
    pad_y = (max_side - h) // 2
    padded = ImageOps.expand(cropped, border=(pad_x, pad_y, max_side - w - pad_x, max_side - h - pad_y), fill=0)

    small = padded.resize((8, 8), Image.Resampling.LANCZOS)
    small_arr = np.asarray(small).astype(np.float32)

    if np.mean(small_arr) > 127:
        small_arr = 255.0 - small_arr

    scaled = (small_arr / 255.0) * 16.0
    feat = scaled.flatten().reshape(1, -1).astype(np.float32)
    viz = (scaled / 16.0 * 255).astype(np.uint8)
    viz_img = Image.fromarray(viz).resize((120, 120), Image.Resampling.NEAREST)

    return feat, viz_img

# ---------- UI ----------
st.markdown(
    """
    <h1 style='text-align:center;color:#FFD700;'>✏️ Digit Recognizer — Gradient Boosting</h1>
    <h4 style='text-align:center;color:#FFFFFF;'>Draw a digit (0–9) below and click <b>Predict</b></h4>
    """,
    unsafe_allow_html=True,
)

# Centering the drawing area
col_space_left, col_canvas, col_space_right = st.columns([1, 2, 1])
with col_canvas:
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=False,  # Hides undo, redo, delete buttons
    )

# Centering buttons
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True, layout='centered)
col1, col2 = st.columns([1, 1])
with col1:
    predict = st.button("🔍 Predict", use_container_width=True)


show_debug = st.checkbox("Show preprocessed 8×8 image (debug)", value=False)

# Prediction logic
if predict:
    if canvas_result.image_data is None:
        st.warning("Please draw something first.")
    else:
        img_arr = canvas_result.image_data.astype("uint8")
        pil = Image.fromarray(img_arr)

        features, viz_img = preprocess_for_digits(pil)
        if show_debug:
            st.image(viz_img, width=120, caption="Preprocessed 8x8 view")

        try:
            pred = model.predict(features)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs) * 100)

            st.markdown(
                f"<h2 style='text-align:center;color:#00FF7F;'>Predicted Digit: {int(pred)}</h2>",
                unsafe_allow_html=True,
            )
            if confidence is not None:
                st.markdown(
                    f"<h3 style='text-align:center;color:#FFD700;'>Confidence: {confidence:.2f}%</h3>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")
