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
    """Pure Python PDF generator with short lines."""
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

    ops = ["BT", "/F1 11 Tf", "50 780 Td", "16 TL"]
    for idx, line in enumerate(all_lines):
        clean = (
            str(line)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .replace("\\", "\\\\")
            .replace("(", "\\(") .replace(")", "\\)")
        )
        if idx == 0:
            ops.extend(["/F2 15 Tf", "(" + clean + ") Tj", "/F1 11 Tf"])
        else:
            ops.extend(["T*", "(" + clean + ") Tj"])
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")

    o1 = ("1 0 obj" + nl + "<< /Type /Catalog /Pages 2 0 R >>" + nl + "endobj" + nl).encode("ascii")
    o2 = ("2 0 obj" + nl + "<< /Type /Pages /Kids [3 0 R] /Count 1 >>" + nl + "endobj" + nl).encode("ascii")
    o3_body = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    o3 = ("3 0 obj" + nl + o3_body + nl + "endobj" + nl).encode("ascii")
    o4 = ("4 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>" + nl + "endobj" + nl).encode("ascii")
    o5 = ("5 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>" + nl + "endobj" + nl).encode("ascii")
    o6 = ("6 0 obj" + nl + "<< /Length " + str(len(stream)) + " >>" + nl + "stream" + nl).encode("ascii") + stream + (nl + "endstream" + nl + "endobj" + nl).encode("ascii")

    objs = [o1, o2, o3, o4, o5, o6]
    buf = bytearray(("%PDF-1.4" + nl).encode("ascii"))
    offs = [0]
    for obj in objs:
        offs.append(len(buf))
        buf.extend(obj)

    xref = len(buf)
    buf.extend(("xref" + nl + "0 7" + nl).encode("ascii"))
    buf.extend(("0000000000 65535 f " + nl).encode("ascii"))
    for off in offs[1:]:
        buf.extend((str(off).zfill(10) + " 00000 n " + nl).encode("ascii"))

    t_dict = "<< /Size 7 /Root 1 0 R >>"
    buf.extend(("trailer
