import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Page Config
st.set_page_config(
    page_title="Malaria Cell Detection", page_icon="🔬", layout="centered"
)

# Dark Mode CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #1c212b;
        color: #ffffff;
        border: 1px solid #30363d;
    }
    .stButton > button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2ea043;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Malaria Detection App")
st.caption("Upload a blood cell image to check for Parasitized or Uninfected cells.")


@st.cache_resource
def load_model():
    # Replace 'malaria_model.h5' with your actual model filename in GitHub
    return tf.keras.models.load_model("malaria_model.h5")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.info("Ensure your model file is in the repository and listed properly.")
    st.stop()


def preprocess_image(image, target_size=(64, 64)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


# Patient Name Input
patient_name = st.text_input("Patient Name", placeholder="Enter patient name...")

if patient_name.strip():
    st.info(f"Patient: **{patient_name.strip()}**")
else:
    st.caption("Optional: Enter a patient name to include in the result.")

# Image Upload & Prediction
uploaded_file = st.file_uploader(
    "Upload cell image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cell Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing cell image..."):
            processed = preprocess_image(image, target_size=(64, 64))
            preds = model.predict(processed, verbose=0)
            score = float(preds[0][0])

            st.subheader("Diagnostic Result")
            if patient_name.strip():
                st.write(f"**Patient Name:** {patient_name.strip()}")

            # Adjust threshold/label order based on your model training classes
            if score < 0.5:
                confidence = (1.0 - score) * 100.0
                st.error(
                    f"🦠 **Parasitized** — Confidence: {confidence:.2f}%"
                )
            else:
                confidence = score * 100.0
                st.success(f"✅ **Uninfected** — Confidence: {confidence:.2f}%")
