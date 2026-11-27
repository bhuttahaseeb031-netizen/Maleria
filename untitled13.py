import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaria Cell Detection", page_icon="🔬", layout="centered")

# --- CLEAN DARK THEME CSS (HIGH CONTRAST) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #f0f6fc;
    }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-top: 15px;
        color: #ffffff;
    }
    .med-card {
        background-color: #1c2128;
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #2ea043;
        margin-top: 12px;
        margin-bottom: 15px;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Malaria Parasite Detection System")
st.write("Upload a microscopic blood smear cell image to detect Malaria infection, view recommended care, and download a PDF diagnostic report.")

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
    return tf.keras.models.load_model("malaria_model.h5")

try:
    model = load_malaria_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model file load nahi ho saki: {e}. Check karein ke model file GitHub repo mein maujood hai aur uska naam sahi hai.")

# --- MEDICINE / CARE GUIDANCE HELPER ---
def get_medicine_info(is_infected):
    if is_infected:
        return {
            "title": "Commonly Prescribed Antimalarial Medicines (Doctor Consultation Required)",
            "items": [
                "Artemether-Lumefantrine (Coartem) - Standard first-line ACT therapy",
                "Chloroquine Phosphate - Used for chloroquine-sensitive Plasmodium strains",
                "Atovaquone-Proguanil (Malarone) - Oral antimalarial treatment option",
                "Paracetamol (Acetaminophen) - For fever and body aches (avoid NSAIDs until evaluated)"
            ],
            "note": "Important: Exact medicine and dosage depend on Plasmodium species (P. falciparum vs P. vivax), patient weight, and local drug resistance. Consult a physician immediately."
        }
    else:
        return {
            "title": "Supportive Care & Preventive Guidance (Uninfected Result)",
            "items": [
                "No antimalarial medication required based on this screening",
                "Maintain adequate hydration (ORS / clean fluids) if experiencing mild fever or fatigue",
                "Paracetamol may be used for symptomatic fever relief if advised by a doctor",
                "Use mosquito nets and DEET/Picaridin repellents for prevention"
            ],
            "note": "Note: If fever or chills persist despite an uninfected result, consult a doctor for a repeat blood smear or Rapid Diagnostic Test (RDT)."
        }

# --- PDF GENERATOR FUNCTION (FIXED WIDTHS) ---
def create_pdf(name, age_id, diagnosis, confidence, med_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    page_w = pdf.epw

    pdf.set_font("Arial", "B", 18)
    pdf.cell(page_w, 12, "Malaria Diagnostic Lab Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "", 12)
    pdf.cell(page_w, 9, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(page_w, 9, f"Patient Name: {name if name else 'N/A'}", ln=True)
    pdf.cell(page_w, 9, f"Patient Age / ID: {age_id if age_id else 'N/A'}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(page_w, 10, f"Diagnosis Result: {diagnosis}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(page_w, 9, f"Model Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(6)

    # Medicine Section in PDF
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 8, med_info["title"])
    pdf.ln(2)

    pdf.set_font("Arial", "", 11)
    for item in med_info["items"]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_w, 7, f"- {item}")
    pdf.ln(3)

    pdf.set_font("Arial", "I", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 7, med_info["note"])
    pdf.ln(8)

    pdf.set_font("Arial", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 6, "Medical Disclaimer: This AI-generated report is for screening and educational purposes only. Do not self-medicate; consult a licensed physician for prescription and dosage.")
    return bytes(pdf.output())

# --- IMAGE UPLOAD & PREDICTION ---
uploaded_file = st.file_uploader("Upload Blood Smear Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cell Image", width=260)

    if st.button("Analyze Cell Image", type="primary"):
        input_shape = model.input_shape[1:3]
        img_resized = image.resize(input_shape)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        if prediction.shape[-1] == 1:
            prob = float(prediction[0][0])
            if prob < 0.5:
                is_infected = True
                label = "Parasitized (Malaria Detected)"
                confidence = (1 - prob) * 100
            else:
                is_infected = False
                label = "Uninfected (Normal)"
                confidence = prob * 100
        else:
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100
            is_infected = (class_idx == 0)
            label = "Parasitized (Malaria Detected)" if is_infected else "Uninfected (Normal)"

        med_info = get_medicine_info(is_infected)

        st.markdown(f"""
        <div class="report-card">
            <h3>Result: {label}</h3>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        med_list_html = "".join([f"<li>{m}</li>" for m in med_info["items"]])
        st.markdown(f"""
        <div class="med-card">
            <h4>💊 {med_info['title']}</h4>
            <ul>{med_list_html}</ul>
            <p style="font-size: 0.9em; color: #c9d1d9;"><em>{med_info['note']}</em></p>
        </div>
        """, unsafe_allow_html=True)

        pdf_bytes = create_pdf(patient_name, patient_age, label, confidence, med_info)
        st.download_button(
            label="📄 Save / Download PDF Report",
            data=pdf_bytes,
            file_name=f"Malaria_Report_{patient_name or 'Patient'}.pdf",
            mime="application/pdf"
        )
