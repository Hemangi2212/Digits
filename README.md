```markdown
# ✏️ Digit Recognizer — Gradient Boosting

An interactive **Streamlit web app** that recognizes hand-drawn digits (0–9) using a **Gradient Boosting Classifier** trained on the classic **Digits dataset**.  
Draw a digit, click **Predict**, and see your model identify it in real-time!

---

## 🚀 Features

- 🖊️ Draw digits directly on the screen  
- 🧠 Uses a trained **Gradient Boosting** model (`GradBoosting.pkl`)  
- 🪄 Automatic preprocessing (grayscale, resizing, normalization)  
- 📊 Optional debug view to visualize the 8×8 preprocessed image  
- 🎨 Clean, centered layout with a modern design  

---

## 🧩 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Pillow (PIL)**
- **NumPy**
- **Joblib**
- **streamlit-drawable-canvas**

---

## 📁 Project Structure

Digit_Recognizer/
│
├── app.py                 # Main Streamlit application
├── GradBoosting.pkl       # Trained Gradient Boosting model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation



---

## ⚙️ Installation & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/digit-recognizer.git
   cd digit-recognizer
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. The app will open in your browser at:

   ```
   http://localhost:8501
   ```

---

## 🧮 Model Info

* Trained on **scikit-learn’s digits dataset**
* Model used: `GradientBoostingClassifier`
* Input shape: `8×8` grayscale images (values scaled 0–16)



## 💡 Example Output


## 🖼️ Preview

![App Screenshot](https://github.com/user-attachments/assets/63d87aa6-76e7-4865-a5d0-8b7eb7df2ca2)
![Prediction Screenshot](https://github.com/user-attachments/assets/b390ee55-a0ba-4bd0-beed-d03ceb65a9e9)



