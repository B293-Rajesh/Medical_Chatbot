import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Import your RAG function
from rag_utils import get_answer

# ----------------------
# Load CNN model
# ----------------------
MODEL_PATH = "medical_model.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ['Dental', 'Skin', 'Lungs']  # Must match your dataset folders
IMG_SIZE = 128

# ----------------------
# Streamlit UI
# ----------------------
st.title("Medical Chatbot with Image Diagnosis & RAG")

# Sidebar: Upload image
st.sidebar.header("Upload Medical Image (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

# Sidebar: Ask a medical question
st.sidebar.header("Ask a Medical Question")
user_query = st.sidebar.text_input("Type your question here:")

# ----------------------
# Image Prediction
# ----------------------
if uploaded_file is not None:
    filepath = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.success(f"Predicted Disease: {predicted_class}")

    os.remove(filepath)

# ----------------------
# RAG Chatbot Answer
# ----------------------
if user_query:
    st.write("You asked:", user_query)

    # ✅ Call your RAG function
    answer = get_answer(user_query)
    st.info(answer)