import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🧬",
    layout="wide"
)

css = "".join([
    "<style>",
    ".stApp { background-color: #0b1120; color: #f8fafc; }",
    ".hero-banner { background: linear-gradient(135deg, #0d9488, #0369a1); ",
    "padding: 18px; border-radius: 14px; border: 1px solid #38bdf8; ",
    "margin-bottom: 18px; text-align: center; color: #ffffff; }",
    ".hero-banner h1 { color: #ffffff; margin: 0; font-size: 26px; }",
    ".hero-banner p { color: #e0f2fe; margin: 4px 0 0 0; font-size: 14px; }",
    ".section-title { background-color: #111827; border-left: 5px solid #14b8a6; ",
    "padding: 10px 14px; border-radius: 8px; color: #38bdf8; ",
    "font-size: 18px; font-weight: 700; margin-bottom: 12px; }",
    ".report-sheet { background-color: #0f172a; border: 2px solid #2dd4bf; ",
    "border-radius: 12px; padding: 16px; margin: 12px 0; color: #f8fafc; }",
    ".report-header { background-color: #0d9488; color: #ffffff; ",
    "padding: 8px 12px; border-radius: 8px; font-weight: 800; ",
    "font-size: 15px; margin-bottom: 10px; text-align: center; }",
    ".report-table { width: 100%; border-collapse: collapse; }",
    ".report-table td { padding: 7px 8px; border-bottom: 1px solid #1e293b; ",
    "font-size: 14px; color: #f1f5f9; }",
    ".report-label { font-weight: 700; color: #38bdf8 !important; width: 38%; }",
    ".med-box { background-color: #134e4a; border: 1px solid #2dd4bf; ",
    "padding: 10px 12px; border-radius: 8px; margin-top: 10px; ",
    "color: #ffffff; font-size: 14px; }",
    ".stButton>button { background: linear-gradient(90deg, #0d9488, #0284c7); ",
    "color: #ffffff; border-radius: 8px; border: none; padding: 10px; ",
    "font-weight: 700; width: 100%; }",
    ".stDownloadButton>button { background: linear-gradient(90deg, #d97706, #f59e0b); ",
    "color: #000000; border-radius: 8px; border: none; padding: 10px; ",
    "font-weight: 800; width: 100%; }",
    ".footer-credit { text-align: center; margin-top: 28px; padding: 12px; ",
    "border-top: 1px solid #334155; color: #38bdf8; ",
    "font-size: 16px; font-weight: 700; }",
    "</style>"
])
st.markdown(css, unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
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
        "Disclaimer: Automated AI screening report for",
        "educational purposes only. Consult a doctor.",
        sep,
        "Created by Haseeb Bhutta",
    ]

    ops = []
    ops.append("BT")
    ops.append("/F1 10.5 Tf")
    ops.append("45 785 Td")
    ops.append("15 TL")

    for idx, line in enumerate(all_lines):
        clean = (
            str(line)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .replace("\\", "\\\\")
            .replace("(", "\\(") .replace(")", "\\)")
        )
        if idx == 0:
            ops.append("/F2 14 Tf")
            ops.append("(" + clean + ") Tj")
            ops.append("/F1 10.5 Tf")
        else:
            ops.append("T*")
            ops.append("(" + clean + ") Tj")
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")

    o1_str = "1 0 obj" + nl + "<< /Type /Catalog /Pages 2 0 R >>" + nl + "endobj" + nl
    o2_str = "2 0 obj" + nl + "<< /Type /Pages /Kids [3 0 R] /Count 1 >>" + nl + "endobj" + nl

    p1 = "<< /Type /Page /Parent 2 0 R "
    p2 = "/MediaBox [0 0 612 842] "
    p3 = "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
    p4 = "/Contents 6 0 R >>"
    o3_str = "3 0 obj" + nl + p1 + p2 + p3 + p4 + nl + "endobj" + nl

    f1_a = "<< /Type /Font /Subtype /Type1 "
    f1_b = "/BaseFont /Helvetica >>"
    o4_str = "4 0 obj" + nl + f1_a + f1_b + nl + "endobj" + nl

    f2_a = "<< /Type /Font /Subtype /Type1 "
    f2_b = "/BaseFont /Helvetica-Bold >>"
    o5_str = "5 0 obj" + nl + f2_a + f2_b + nl + "endobj" +
