import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🧬",
    layout="wide"
)

# --- SAFE & CLEAN DARK STYLING (NO BLANK SCREEN OVERRIDE) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0d9488 0%, #0369a1 100%);
        padding: 18px 24px;
        border-radius: 14px;
        border: 1px solid #38bdf8;
        margin-bottom: 18px;
        text-align: center;
        color: #ffffff;
    }
    .hero-banner h1 {
        color: #ffffff;
        margin: 0;
        font-size: 26px;
    }
    .hero-banner p {
        color: #e0f2fe;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
    .section-title {
        background-color: #111827;
        border-left: 5px solid #14b8a6;
        padding: 10px 14px;
        border-radius: 8px;
        color: #38bdf8;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .report-sheet {
        background-color: #0f172a;
        border: 2px solid #2dd4bf;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #f8fafc;
    }
    .report-header {
        background-color: #0d9488;
        color: #ffffff;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 10px;
        text-align: center;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
    }
    .report-table td {
        padding: 7px 8px;
        border-bottom: 1px solid #1e293b;
        font-size: 14px;
        color: #f1f5f9;
    }
    .report-label {
        font-weight: 700;
        color: #38bdf8 !important;
        width: 38%;
    }
    .med-box {
        background-color: #134e4a;
        border: 1px solid #2dd4bf;
        padding: 10px 12px;
        border-radius: 8px;
        margin-top: 10px;
        color: #ffffff;
        font-size: 14px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0d9488, #0284c7);
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 700;
        width: 100%;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: #000000;
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
        color: #38bdf8;
        font-size: 16px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
    """Pure Python PDF generator with short lines."""
    nl = chr(10)
    sep = "-" * 48
    p_name = patient_name.strip() if patient_name and patient_name.strip() else "Not Specified"
    p_gender = patient_gender if patient_gender else "Not Specified"

    all_lines = [
        "MALARIA AI DIAGNOSTIC REPORT",
        title,
        sep,
        "Patient Name: " + p_name,
        "Gender: " + p_gender,
        "Generated On: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        sep,
    ] + [str(k) + ": " + str(v) for k, v in lines] + [
        sep,
        "Disclaimer: Automated AI screening report for educational",
        "purposes only. Consult a doctor before taking medicine.",
        sep,
        "Created by Haseeb Bhutta",
    ]

    ops = ["BT", "/F1 10.5 Tf", "45 785 Td", "15 TL"]
    for idx, line in enumerate(all_lines):
        clean = (
            str(line)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .replace("\\", "\\\\")
            .replace("(", "\\(") .replace(")", "\\)")
        )
        if idx == 0:
            ops.extend(["/F2 14 Tf", "(" + clean + ") Tj", "/F1 10.5 Tf"])
        else:
            ops.extend(["T*", "(" + clean + ") Tj"])
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")

    o1 = f"""1 0 obj{nl}<< /Type /
