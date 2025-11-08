import streamlit as st
import numpy as np
import cv2
import pickle
from streamlit_drawable_canvas import st_canvas

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🔢",
    layout="centered",
)

# ------------------- LOAD MODEL -------------------
with open("GradBoosting.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------- PAGE STYLE -------------------
st.markdown("""
    <style>
        body { background-color: #0E1117; }
        .main { background-color: #0E1117; color: white; }
        h1 { text-align: center; color: #FFD700; }
        .result-box {
            text-align: center;
            border-radius: 12px;
            padding: 20px;
            background: #1E1E1E;
            box-shadow: 0 0 20px rgba(255,215,0,0.3);
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .stButton>button {
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>✏️ Digit Recognizer — Gradient Boosting Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Draw a digit (0–9) and click Predict!</p>", unsafe_allow_html=True)

# ------------------- CANVAS -------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear Canvas", use_container_width=True)

# ------------------- CLEAR FUNCTION -------------------
if clear_btn:
    st.session_state.canvas = None
    st.rerun()

# ------------------- PREDICT FUNCTION -------------------
if predict_btn and canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # resize to 8x8 for sklearn digits dataset
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    
    # scale pixel values (0-255 -> 0-16)
    scaled = (resized / 16.0).astype(np.float32)
    flat = scaled.flatten().reshape(1, -1)
    
    pred = model.predict(flat)[0]
    
    # get confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(flat)[0]
        confidence = np.max(prob) * 100
    else:
        confidence = np.nan
    
    st.markdown(
        f"""
        <div class="result-box">
            <h2 style='color:#00FF7F;'>Predicted Digit: {pred}</h2>
            <h3 style='color:#FFD700;'>Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
