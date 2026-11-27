import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Malaria Detection App")
st.write("Upload a microscopic blood cell image to detect whether it is **Parasitized** or **Uninfected**.")

# Load the trained model (Replace 'malaria_model.h5' with your actual model filename)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria_model.h5")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure your model file (e.g., 'malaria_model.h5') is in the same GitHub repository.")
    st.stop()

# Image Preprocessing Function
def preprocess_image(image):
    # Resize image to match the input shape your model was trained on (commonly 64x64 or 128x128 or 224x224)
    img = image.resize((64, 64))
    img_array = np.array(img)
    
    # Ensure 3 color channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Normalize pixel values (0 to 1)
    img_array = img_array / 255.0
    
    # Add batch dimension: (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File Uploader
uploaded_file = st.file_uploader("Choose a cell image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Blood Cell Image", use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing cell image..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Adjust threshold/class logic based on how your model outputs predictions
            # Assuming binary classification: 0 = Parasitized, 1 = Uninfected (standard alphabetical folder order)
            score = float(prediction[0][0])
            
            st.subheader("Prediction Result:")
            if score < 0.5:
                confidence = (1 - score) * 100
                st.error(f"🦠 **Parasitized (Malaria Detected)** — Confidence: {confidence:.2f}%")
            else:
                confidence = score * 100
                st.success(f"✅ **Uninfected (Healthy Cell)** — Confidence: {confidence:.2f}%")
                
