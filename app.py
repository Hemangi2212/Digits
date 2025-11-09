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
    """
    Convert a PIL RGBA/RGB image from canvas to sklearn 'digits' features:
    - convert to grayscale
    - get bounding box of non-background pixels
    - crop, pad to square and center
    - resize to 8x8 using ANTIALIAS
    - invert & scale to 0..16
    returns: feature vector shape (1,64), and a small visualization image (8x8) for debugging
    """
    # ensure grayscale
    img = pil_img.convert("L")  # 0 (black) .. 255 (white) where drawing likely white on black background
    arr = np.asarray(img)

    # If canvas used white strokes on black background, keep that; if opposite, invert accordingly.
    # We'll detect if strokes are dark or bright by checking average of non-empty area.
    # Consider pixels significantly different from background (background near 0 or 255).
    # First compute a simple threshold to separate foreground:
    thresh = 50
    # detect where there is drawing (non-background) by thresholding from both sides
    fg_mask = (arr > thresh) & (arr < 255 - thresh)
    if not fg_mask.any():
        # fallback: treat white strokes on black as foreground
        fg_mask = arr > (np.mean(arr) + 10)

    coords = np.argwhere(fg_mask)
    if coords.size == 0:
        # nothing drawn, return a blank 8x8
        empty = np.zeros((8, 8), dtype=np.float32)
        return empty.flatten().reshape(1, -1), Image.fromarray((empty / 16.0 * 255).astype(np.uint8))

    # bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slice end exclusive
    cropped = img.crop((x0, y0, x1, y1))

    # pad to square
    w, h = cropped.size
    max_side = max(w, h)
    pad_x = (max_side - w) // 2
    pad_y = (max_side - h) // 2
    padded = ImageOps.expand(cropped, border=(pad_x, pad_y, max_side - w - pad_x, max_side - h - pad_y), fill=0)

    # resize to 8x8 using ANTIALIAS (LANCZOS)
    small = padded.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to numpy and invert/scale to 0..16 expected by sklearn digits
    small_arr = np.asarray(small).astype(np.float32)

    # Determine if strokes are white on black (common here). load_digits uses white(16) on dark background (0),
    # so we want strokes -> large values. If drawn strokes are bright (high values), invert: (255 - pixel)
    # If drawn strokes are dark (low values), we might not invert. We'll check mean intensity of small_arr.
    if np.mean(small_arr) > 127:
        # strokes are bright -> invert so that stroke becomes high values like load_digits
        small_arr = 255.0 - small_arr

    # Scale 0..255 -> 0..16
    scaled = (small_arr / 255.0) * 16.0

    # final feature vector
    feat = scaled.flatten().reshape(1, -1).astype(np.float32)

    # create a visualization image scaled back to 0..255 for display (enlarged)
    viz = (scaled / 16.0 * 255).astype(np.uint8)
    viz_img = Image.fromarray(viz).resize((120, 120), Image.Resampling.NEAREST)

    return feat, viz_img

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;color:#FFD700;'>✏️ Digit Recognizer — Gradient Boosting</h1>", unsafe_allow_html=True)
st.write("Draw a digit (0–9) and click Predict. Use Clear to start over.")

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    predict = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear Canvas", use_container_width=True)

# Clear canvas: do not fully reload app, just reset session key used by canvas
if clear:
    # the canvas uses key 'canvas' internally; removing it then rerunning clears drawing
    # For safety we set a sentinel in session_state and rerun
    st.session_state.canvas = None
    st.rerun()

# Optional debug toggle
show_debug = st.checkbox("Show preprocessed 8×8 image (debug)", value=True)

# Prediction
if predict:
    if canvas_result.image_data is None:
        st.warning("Please draw something first.")
    else:
        # Convert the canvas image array -> PIL
        img_arr = canvas_result.image_data.astype("uint8")
        pil = Image.fromarray(img_arr)  # RGBA or RGB

        features, viz_img = preprocess_for_digits(pil)

        # Optional: show the 8x8 visualization so you can see what the model receives
        if show_debug:
            st.markdown("**Preprocessed (8×8) view — what the model sees:**")
            st.image(viz_img, width=120)

        # Predict
        try:
            pred = model.predict(features)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs) * 100)

            # show only predicted number + confidence
            st.markdown(f"<h2 style='text-align:center;color:#00FF7F;'>Predicted Digit: {int(pred)}</h2>", unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f"<h3 style='text-align:center;color:#FFD700;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
