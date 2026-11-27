import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
from datetime import datetime
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaria Cell Detection", page_icon="🔬", layout="centered")

# --- CUSTOM DARK THEME CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Malaria Parasite Detection System")
st.write("Upload a microscopic blood smear cell image to detect Malaria infection and download a PDF diagnostic report.")

# --- PATIENT DETAILS ---
st.subheader("Patient Information")
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="Enter full name")
with col2:
    patient_age = st.text_input("Patient Age / ID", placeholder="e.g., 24 / P-102")

# --- LOAD MODEL ---
@st.cache_resource
def load_malaria_model():
    # Apni model file ka exact naam yahan likhein (e.g., 'malaria_model.h5')
    return tf.keras.models.load_model("malaria_model.h5")

try:
    model = load_malaria_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model file load nahi ho saki: {e}. Check karein ke model file GitHub repo mein maujood hai aur uska naam sahi hai.")

# --- PDF GENERATOR FUNCTION ---
def create_pdf(name, age_id, diagnosis, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Malaria Diagnostic Lab Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Patient Name: {name if name else 'N/A'}", ln=True)
    pdf.cell(0, 10, f"Patient Age / ID: {age_id if age_id else 'N/A'}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Diagnosis Result: {diagnosis}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Model Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 8, "Disclaimer: This AI-generated report is for screening assistance only. Please consult a qualified pathologist or physician for clinical confirmation.")
    return pdf.output(dest="S").encode("latin-1")

# --- IMAGE UPLOAD & PREDICTION ---
uploaded_file = st.file_uploader("Upload Blood Smear Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cell Image", width=260)

    if st.button("Analyze Cell Image", type="primary"):
        # Model ke input size ke mutabiq resize karein (aksar 64x64 ya 128x128 hota hai)
        input_shape = model.input_shape[1:3]
        img_resized = image.resize(input_shape)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        # Binary (1 output) aur Categorical (2 outputs) dono handle karta hai
        if prediction.shape[-1] == 1:
            prob = float(prediction[0][0])
            if prob < 0.5:
                label = "Parasitized (Malaria Detected)"
                confidence = (1 - prob) * 100
            else:
                label = "Uninfected (Normal)"
                confidence = prob * 100
        else:
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100
            label = "Parasitized (Malaria Detected)" if class_idx == 0 else "Uninfected (Normal)"

        st.markdown(f"""
        <div class="report-card">
            <h3>Result: {label}</h3>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # --- PDF DOWNLOAD BUTTON ---
        pdf_bytes = create_pdf(patient_name, patient_age, label, confidence)
        st.download_button(
            label="📄 Save / Download PDF Report",
            data=pdf_bytes,
            file_name=f"Malaria_Report_{patient_name or 'Patient'}.pdf",
            mime="application/pdf"
        )
