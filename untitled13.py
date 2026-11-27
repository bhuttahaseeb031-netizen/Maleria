import os
import streamlit as st
import tensorflow as tf

MODEL_PATH = "malaria_model.h5"  # Change to your actual model filename in GitHub


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
if model is None:
    st.warning(
        f"Model file `{MODEL_PATH}` not found in the repository. "
        "Upload the trained model file to GitHub or update `MODEL_PATH` in `untitled13.py`."
    )
    st.stop()
