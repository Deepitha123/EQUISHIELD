import streamlit as st
import joblib
from PIL import Image

# Load ML model and vectorizer
ml_model = joblib.load("bias_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Configure Streamlit page
st.set_page_config(
    page_title="Bias Detector AI", 
    page_icon="🔍", 
    layout="centered"
)

# Custom CSS for Professional Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f7fc;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #1e3a8a;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #475569;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1e3a8a, #3b82f6);
            color: white !important;
            border-radius: 8px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3b82f6, #1e3a8a);
            transform: scale(1.05);
        }
        .result-box {
            background-color: #e0f2fe;
            padding: 15px;
            border-radius: 10px;
            font-size: 22px;
            text-align: center;
            color: #0c4a6e;
            font-weight: bold;
            margin-top: 20px;
        }
        .confidence-text {
            text-align: center;
            font-size: 18px;
            color: #475569;
            margin-top: 10px;
        }
        .warning-text {
            color: red;
            font-size: 18px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown('<p class="title">🔍 AI-Powered Bias Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect Bias in Text Using Machine Learning</p>', unsafe_allow_html=True)
st.write("---")  # Divider

# Image Banner (Ensure image is in the project folder)
image = Image.open("bias_detector.png")
st.image(image, use_container_width=True)

# User Input Section
st.markdown("### ✍️ Enter a Sentence for Bias Analysis:")
user_input = st.text_area(
    "💬 **Type your text below:**", 
    placeholder="e.g., Men are natural leaders, while women are nurturing...", 
    height=150
)

# Analyze Button
if st.button("🚀 Analyze Bias"):
    if user_input.strip():
        # ML Model Prediction
        input_vector = vectorizer.transform([user_input])
        ml_prediction = ml_model.predict(input_vector)[0]

        # Confidence Score (if model supports probabilities)
        if hasattr(ml_model, "predict_proba"):
            confidence = max(ml_model.predict_proba(input_vector)[0]) * 100
            confidence_text = f" **Confidence:** {confidence:.2f}%"
        else:
            confidence_text = ""

        # Display Results
        st.markdown('<p class="result-box">🤖 <strong>Bias Classification:</strong> {}</p>'.format(ml_prediction), unsafe_allow_html=True)
        st.markdown(f"<p class='confidence-text'>{confidence_text}</p>", unsafe_allow_html=True)
    else:
        st.markdown('<p class="warning-text">⚠️ Please enter some text to analyze. 🚀</p>', unsafe_allow_html=True)
