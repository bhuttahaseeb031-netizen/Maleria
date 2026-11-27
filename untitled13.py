import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🔬",
    layout="centered"
)

# --- CUSTOM DARK MODE THEME ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .result-box {
        background-color: #161b22;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    .stDownloadButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stDownloadButton>button:hover {
        background-color: #388bfd;
    }
    .footer-credit {
        text-align: center;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #30363d;
        color: #8b949e;
        font-size: 15px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
    """Pure Python PDF generator (safe against copy-paste line breaks)."""
    nl = chr(10)
    p_name = patient_name.strip() if patient_name and patient_name.strip() else "Not Specified"
    p_gender = patient_gender if patient_gender else "Not Specified"

    all_lines = [
        "MALARIA AI DIAGNOSTIC REPORT",
        title,
        "------------------------------------------------------------",
        "Patient Name: " + p_name,
        "Gender: " + p_gender,
        "Generated On: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "------------------------------------------------------------",
    ] + [str(k) + ": " + str(v) for k, v in lines] + [
        "------------------------------------------------------------",
        "Disclaimer: Automated AI screening report for educational",
        "and preliminary screening purposes only. Consult a doctor.",
        "------------------------------------------------------------",
        "Created by Haseeb Bhutta",
    ]

    content_ops = ["BT", "/F1 11 Tf", "50 780 Td", "16 TL"]
    for idx, line in enumerate(all_lines):
        safe_line = (
            str(line)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .replace("\\", "\\\\")
            .replace("(", "\\(") .replace(")", "\\)")
        )
        if idx == 0:
            content_ops.append("/F2 15 Tf")
            content_ops.append("(" + safe_line + ") Tj")
            content_ops.append("/F1 11 Tf")
        else:
            content_ops.append("T*")
            content_ops.append("(" + safe_line + ") Tj")
    content_ops.append("ET")
    stream_data = nl.join(content_ops).encode("ascii")

    obj1 = ("1 0 obj" + nl + "<< /Type /Catalog /Pages 2 0 R >>" + nl + "endobj" + nl).encode("ascii")
    obj2 = ("2 0 obj" + nl + "<< /Type /Pages /Kids [3 0 R] /Count 1 >>" + nl + "endobj" + nl).encode("ascii")
    obj3 = (
        "3 0 obj" + nl
        + "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        + nl + "endobj" + nl
    ).encode("ascii")
    obj4 = ("4 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>" + nl + "endobj" + nl).encode("ascii")
    obj5 = ("5 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>" + nl + "endobj" + nl).encode("ascii")
    obj6 = (
        ("6 0 obj" + nl + "<< /Length " + str(len(stream_data)) + " >>" + nl + "stream" + nl).encode("ascii")
        + stream_data
        + (nl + "endstream" + nl + "endobj" + nl).encode("ascii")
    )

    objects = [obj1, obj2, obj3, obj4, obj5, obj6]
    pdf_bytes = bytearray(("%PDF-1.4" + nl).encode("ascii"))
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj)

    xref_pos = len(pdf_bytes)
    pdf_bytes.extend(("xref" + nl + "0 " + str(len(objects) + 1) + nl).encode("ascii"))
    pdf_bytes.extend(("0000000000 65535 f " + nl).encode("ascii"))
    for off in offsets[1:]:
        pdf_bytes.extend((f"{off:010d} 00000 n " + nl).encode("ascii"))

    trailer = (
        "trailer" + nl
        + "<< /Size " + str(len(objects) + 1) + " /
