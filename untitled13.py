import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Malaria AI Detector", page_icon="🧬", layout="wide")

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


def make_obj(num, body):
    nl = chr(10)
    return (str(num) + " 0 obj" + nl + body + nl + "endobj" + nl).encode("ascii")


def create_pdf(title, p_name, p_gender, lines):
    nl = chr(10)
    sep = "-" * 45
    name_val = p_name.strip() if p_name and p_name.strip() else "Not Specified"
    all_lines = [
        "MALARIA AI DIAGNOSTIC REPORT",
        title,
        sep,
        "Patient Name: " + name_val,
        "Gender: " + str(p_gender),
        "Date: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        sep,
    ] + [str(k) + ": " + str(v) for k, v in lines] + [
        sep,
        "Disclaimer: Consult a doctor before medicine.",
        "Created by Haseeb Bhutta",
    ]
    ops = ["BT", "/F1 10.5 Tf", "45 785 Td", "15 TL"]
    for idx, line in enumerate(all_lines):
        clean = str(line).encode("ascii", "ignore").decode("ascii")
        clean = clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if idx == 0:
            ops.extend(["/F2 14 Tf", "(" + clean + ") Tj", "/F1 10.5 Tf"])
        else:
            ops.extend(["T*", "(" + clean + ") Tj"])
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")
    p_dict = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    o6 = ("6 0 obj" + nl + "<< /Length " + str(len(stream)) + " >>" + nl + "stream" + nl).encode("ascii") + stream + (nl + "endstream" + nl + "endobj" + nl).encode("ascii")
    objs = [
        make_obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        make_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        make_obj(3, p_dict),
        make_obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        make_obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        o6,
    ]
    buf = bytearray(("%PDF-1.4" + nl).encode("ascii"))
    offs = [0]
    for o in objs:
        offs.append(len(buf))
        buf.extend(o)
    xref = len(buf)
    buf.extend(("xref" + nl + "0 7" + nl + "0000000000 65535 f " + nl).encode("ascii"))
    for off in offs[1:]:
        buf.extend((str(off).zfill(10) + " 00000 n " + nl).encode("ascii"))
    buf.extend(("trailer" + nl + "<< /Size 7 /Root 1 0 R >>" + nl + "startxref" + nl + str(xref) + nl + "%%EOF").encode("ascii"))
    return bytes(buf)


def show_report_card(header_title, rows, meds_text):
    tr_list = ['<tr><td class="report-label">' + k + '</td><td>' + v + '</td></tr>' for k, v in rows]
    html = (
        '<div class="report-sheet">'
        '<div class="report-header">' + header_title + '</div>'
        '<table class="report-table">' + "".join(tr_list) + '</table>'
        '<div class="med-box"><strong>💊 Suggested Medicines (Reference):</strong><br/>' + meds_text + '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=
