import os
import streamlit as st
import tensorflow as tf

MODEL_FILE = "malaria_model.h5"  # Replace with your actual model filename


@st.cache_resource
def load_model():
  if not os.path.exists(MODEL_FILE):
    return None
  return tf.keras.models.load_model(MODEL_FILE)


model = load_model()

if model is None:
  st.info(
      f"Model file `{MODEL_FILE}` abhi repository mein nahi mili. Model file"
      " upload karein ya filename update karein."
  )
  st.stop()
