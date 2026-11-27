import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🧬",
    layout="centered"
)

# --- NEW CYBER-MEDICAL TEAL & EMERALD DARK THEME ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #0f292e 0%, #090d16 55%, #06090f 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #115e59 100%);
        padding: 22px 24px;
        border-radius: 16px;
        border: 1px solid #2dd4bf;
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.25);
        margin-bottom: 22px;
        text-align: center;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 28px;
        letter-spacing: 0.5px;
    }
    .hero-banner p {
        color: #ccfbf1;
        margin: 6px 0 0 0;
        font-size: 14px;
    }
    .patient-card {
        background-color: rgba(15, 23, 42, 0.85);
        border: 1px solid #1e293b;
        border-left: 4px solid #14b8a6;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .result-box {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #14b8a6;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .med-box {
        background-color: rgba(13, 148, 136, 0.12);
        border: 1px dashed #2dd4bf;
        padding: 14px;
        border-radius: 10px;
        margin-top: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0d9488, #10b981);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;
        letter-spacing: 0.3px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #14b8a6, #34d399);
        box-shadow: 0 0 14px rgba(20, 184, 166, 0.5);
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: #090d16;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-weight: 800;
        width: 100%;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d9488 !important;
        color: #ffffff !important;
    }
    .footer-credit {
        text-align: center;
        margin-top: 38px;
        padding: 14px;
        border-top: 1px solid #1e293b;
        color: #2dd4bf;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
    """Pure Python PDF generator immune to copy-paste line breaks."""
    nl = chr(10)
    p_name = patient_name.strip() if patient_name and patient_name.strip() else "Not Specified"
    p_gender = patient_gender if patient_gender else "Not Specified"

    all_lines = [
        "MALARIA AI DIAGNOSTIC & CLINICAL REPORT",
        title,
        "------------------------------------------------------------",
        "Patient Name: " + p_name,
        "Gender: " + p_gender,
        "Generated On: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "------------------------------------------------------------",
    ] + [str(k) + ": " + str(v) for k, v in lines] + [
        "--------------------------------
