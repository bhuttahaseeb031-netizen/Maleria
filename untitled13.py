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

# Dark Theme Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Malaria Detection App")
st.write("Upload a microscopic blood cell image to detect whether it is **Parasitized** or **Uninfected**.")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria_model.h5")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure your model file ('malaria_model.h5') is uploaded in the same GitHub repository.")
    st.stop()

# Image Preprocessing Function
def preprocess_image(image):
    # Ensure 3 color channels (RGB) — handles RGBA and Grayscale images safely
    image = image.convert("RGB")
    # Resize image to match model input shape (change to (128, 128) or (224, 224) if your model uses a different size)
    img = image.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension: (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File Uploader
uploaded_file = st.file_uploader("Choose a blood cell image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Blood Cell Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing cell image..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            # Handle both single-output (sigmoid) and two-class (softmax) models
            if prediction.shape[-1] == 1:
                score = float(prediction[0][0])
                # Note: Usually 0 = Parasitized, 1 = Uninfected in standard Kaggle Malaria dataset
                if score < 0.5:
                    label = "Parasitized (Malaria Detected)"
                    confidence = (1 - score) * 100
                    st.error(f"**Result:** {label} ({confidence:.2f}% confidence)")
                else:
                    label = "Uninfected (Healthy Cell)"
                    confidence = score * 100
                    st.success(f"**Result:** {label} ({confidence:.2f}% confidence)")
            else:
                class_names = ["Parasitized", "Uninfected"]
                predicted_idx = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0])) * 100
                if predicted_idx == 0:
                    st.error(f"**Result:** Parasitized (Malaria Detected) ({confidence:.2f}% confidence)")
                else:
                    st.success(f"**Result:** Uninfected (Healthy Cell) ({confidence:.2f}% confidence)")
