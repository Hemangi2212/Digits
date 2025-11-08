import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import joblib

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="centered")

# ---------- STYLING ----------
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .main-title {
            text-align: center;
            color: #FFD700;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #BBBBBB;
            font-size: 18px;
        }
        .result-box {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0px 0px 12px rgba(255,215,0,0.4);
            margin-top: 20px;
        }
        .predict-btn {
            background-color: #00C851 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
            font-size: 18px;
        }
        .clear-btn {
            background-color: #FF4444 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='main-title'>🔢 Digit Recognizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Draw a digit (0–9) and let the AI guess with confidence!</div>", unsafe_allow_html=True)
st.write("")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model(path='GradBoosting.pkl'):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ---------- IMAGE PREPROCESS ----------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('L')          # grayscale
    img = img.resize((8, 8))        # match sklearn digits
    arr = np.asarray(img).astype(np.float32)
    arr = (255 - arr) / 255 * 16    # invert + scale 0–16
    return arr.flatten().reshape(1, -1)


# ---------- CANVAS ----------
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=14,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear Canvas", use_container_width=True)

if clear_btn:
    st.session_state.canvas = None
    st.rerun()


# ---------- PREDICTION ----------
if predict_btn:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        img_data = canvas_result.image_data.astype("uint8")
        pil_img = Image.fromarray(img_data)

        features = preprocess_image(pil_img)
        model = load_model('GradBoosting.pkl')

        if model:
            try:
                pred = model.predict(features)[0]

                # Confidence (probability)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    confidence = probs[int(pred)] * 100
                else:
                    confidence = 0

                st.markdown(
                    f"""
                    <div class='result-box'>
                        <h2 style='color:#00C851;'>Predicted Digit: {int(pred)}</h2>
                        <h3 style='color:#FFD700;'>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
