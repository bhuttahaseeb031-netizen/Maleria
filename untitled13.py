import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🧬",
    layout="wide"
)

# --- HIGH-VISIBILITY CYBER-MEDICAL DARK THEME ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
        color: #f8fafc;
    }
    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.8);
    }
    label, p, span, div, h1, h2, h3, h4, td, th {
        color: #f1f5f9 !important;
    }
    input, select, textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0d9488 0%, #0369a1 100%);
        padding: 18px 24px;
        border-radius: 14px;
        border: 1px solid #38bdf8;
        margin-bottom: 18px;
        text-align: center;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 26px;
    }
    .hero-banner p {
        color: #e0f2fe !important;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
    .section-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-top: 4px solid #14b8a6;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 14px;
    }
    .report-sheet {
        background-color: #0f172a;
        border: 2px solid #2dd4bf;
        border-radius: 12px;
        padding: 18px;
        margin-top: 14px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
    }
    .report-header {
        background-color: #0d9488;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 16px;
        margin-bottom: 12px;
        text-align: center;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
    }
    .report-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #1e293b;
        font-size: 14px;
    }
    .report-label {
        font-weight: 700;
        color: #38bdf8 !important;
        width: 38%;
    }
    .med-box {
        background-color: #134e4a;
        border: 1px solid #2dd4bf;
        padding: 12px;
        border-radius: 8px;
        margin-top: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0d9488, #0284c7);
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 700;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #14b8a6, #0ea5e9);
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: #000000 !important;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 800;
        width: 100%;
    }
    .footer-credit {
        text-align: center;
        margin-top: 28px;
        padding: 12px;
        border-top: 1px solid #334155;
        color: #38bdf8 !important;
        font-size: 16px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
    """Pure Python PDF generator with short lines."""
    nl = chr(10)
    sep = "-" * 48
    p_
