import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# ---------- Load Model ----------
@st.cache_resource
def load_model(path="GradBoosting.pkl"):
    return joblib.load(path)

model = load_model()

# ---------- Preprocessing Function ----------
def preprocess_for_digits(pil_img: Image.Image):
    """
    Convert a PIL image (RGBA/RGB) from Streamlit canvas to sklearn 'digits' format:
    - convert to grayscale
    - crop bounding box of drawing
    - pad to square
    - resize to 8×8
    - invert and scale to 0–16
    """
    # Convert to grayscale
    img = pil_img.convert("L")  # 0–255

    # Invert to make strokes white on black (like sklearn digits)
    img = ImageOps.invert(img)

    # Convert to numpy
    arr = np.array(img)

    # Threshold to keep only the drawn part
    coords = np.column_stack(np.where(arr > 30))  # anything >30 is drawing
    if coords.size == 0:
        empty = np.zeros((8, 8), dtype=np.float32)
        return empty.flatten().reshape(1, -1), Image.fromarray((empty * 16).astype(np.uint8))

    # Bounding box of the drawing
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = img.crop((x0, y0, x1, y1))

    # Pad to make square
    w, h = cropped.size
    max_side = max(w, h)
    padded = ImageOps.pad(cropped, (max_side, max_side), color=0)

    # Resize to 8×8
    small = padded.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize (0–16)
    small_arr = np.array(small).astype(np.float32)
    scaled = (small_arr / 255.0) * 16.0

    # Feature vector for model
    feat = scaled.flatten().reshape(1, -1)

    # For visualization (enlarged view)
    viz_img = Image.fromarray((scaled / 16.0 * 255).astype(np.uint8)).resize((120, 120), Image.Resampling.NEAREST)

    return feat, viz_img

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;color:#FFD700;'>✏️ Digit Recognizer — Gradient Boosting</h1>", unsafe_allow_html=True)
st.write("Draw a digit (0–9) below and click **Predict**.")

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

if clear:
    st.experimental_rerun()

show_debug = st.checkbox("Show preprocessed 8×8 image (debug)", value=True)

# ---------- Prediction ----------
if predict:
    if canvas_result.image_data is None:
        st.warning("Please draw something first!")
    else:
        img_arr = canvas_result.image_data.astype("uint8")
        pil = Image.fromarray(img_arr)

        features, viz_img = preprocess_for_digits(pil)

        if show_debug:
            st.markdown("**Model Input Preview (8×8)**")
            st.image(viz_img, width=120)

        try:
            pred = model.predict(features)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = np.max(probs) * 100

            st.markdown(f"<h2 style='text-align:center;color:#00FF7F;'>Predicted Digit: {int(pred)}</h2>", unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f"<h3 style='text-align:center;color:#FFD700;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
