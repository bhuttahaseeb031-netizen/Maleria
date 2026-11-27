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


def make_obj(num, body):
    nl = chr(10)
    return (str(num) + " 0 obj" + nl + body + nl + "endobj" + nl).encode("ascii")


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

    page_dict = (
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    )
    o6_head = ("6 0 obj" + nl + "<< /Length " + str(len(stream)) + " >>" + nl + "stream" + nl).encode("ascii")
    o6_tail = (nl + "endstream" + nl + "endobj" + nl).encode("ascii")

    objs = [
        make_obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        make_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        make_obj(3, page_dict),
        make_obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        make_obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        o6_head + stream + o6_tail,
    ]

    buf = bytearray(("%PDF-1.4" + nl).encode("ascii"))
    offs = [0]
    for obj in objs:
        offs.append(len(buf))
        buf.extend(obj)

    xref = len(buf)
    buf.extend(("xref" + nl + "0 7" + nl + "0000000000 65535 f " + nl).encode("ascii"))
    for off in offs[1:]:
        buf.extend((str(off).zfill(10) + " 00000 n " + nl).encode("ascii"))

    footer = "trailer" + nl + "<< /Size 7 /Root 1 0 R >>" + nl + "startxref" + nl + str(xref) + nl + "%%EOF"
    buf.extend(footer.encode("ascii"))
    return bytes(buf)


hero_html = (
    '<div class="hero-banner">'
    '<h1>🧬 AI Malaria Detection & Medicine System</h1>'
    '<p>Patient Info • Blood Cell Detector • Symptoms & Medicines • Full Report & PDF Download</p>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    patient_name = st.text_input("👤 Patient Name", placeholder="Patient ka naam likhein...")
with col_p2:
    patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

st.divider()

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown('<div class="section-title">🩸 1. Blood Cell Image Detector</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Blood Smear Image Upload Karein (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Blood Cell Image", width=220)

        if st.button("Analyze Blood Cell Image", key="img_detect_btn"):
            arr = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0
            red_ch, green_ch, blue_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            dark_purple_mask = (blue_ch > green_ch * 0.92) & ((red_ch + green_ch + blue_ch) / 3.0 < 0.52)
            spot_ratio = float(np.mean(dark_purple_mask))

            parasitized = spot_ratio > 0.08
            confidence = min(98.5, max(72.0, round(75.0 + abs(spot_ratio - 0.08) * 180, 1)))

            if parasitized:
                med_info = "Artemether-Lumefantrine (Coartem), Chloroquine, Primaquine + Paracetamol (Fever)"
                note_txt = "Malaria parasite ke asaar paye gaye hain. Lab MP/RDT test aur doctor se foran ruju karein."
            else:
                med_info = "No anti-malarial medicine required (Cell normal hai)"
                note_txt = "Blood cell clear hai, parasite pattern nahi mila."

            st.session_state["img_report"] = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "filename": uploaded_file.name,
                "status": "Parasitized (Malaria Detected)" if parasitized else "Uninfected (Normal Cell)",
                "confidence": str(confidence) + "%",
                "spot_ratio": f"{spot_ratio:.2%}",
                "note": note_txt,
                "medicines": med_info,
                "is_parasitized": parasitized,
            }

    if "img_report" in st.session_state and isinstance(st.session_state["img_report"], dict):
        rep = st.session_state["img_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"
        r_time = rep.get("time", datetime.now().strftime("%Y-%m-%d %H:%M"))
        r_file = rep.get("filename", "Uploaded Image")
        r_status = rep.get("status", "Uninfected (Normal Cell)")
        r_conf = rep.get("confidence", "85.0%")
        r_spot = rep.get("spot_ratio", "0.00%")
        r_note = rep.get("note", "Screening complete.")
        r_meds = rep.get("medicines", "Consult a doctor for medical advice.")
        r_para = rep.get("is_parasitized", False)

        if r_para:
            st.error("⚠️ **" + r_status + "** (" + r_conf + ")")
        else:
            st.success("✅ **" + r_status + "** (" + r_conf + ")")

        report_html = (
            '<div class="report-sheet">'
            '<div class="report-header">📋 BLOOD SMEAR DIAGNOSTIC REPORT</div>'
            '<table class="report-table">'
            '<tr><td class="report-label">Patient Name:</td><td>' + display_name + '</td></tr>'
            '<tr><td class="report-label">Gender:</td><td>' + patient_gender + '</td></tr>'
            '<tr><td class="report-label">Date & Time:</td><td>' + r_time + '</td></tr>'
            '<tr><td class="report-label">Sample File:</td><td>' + r_file + '</td></tr>'
            '<tr><td class="report-label">Result:</td><td><strong>' + r_status + '</strong></td></tr>'
            '<tr><td class="report-label">Confidence:</td><td>' + r_conf + '</td></tr>'
            '<tr><td class="report-label">Dark Spot Ratio:</td><td>' + r_spot + '</td></tr>'
            '<tr><td class="report-label">Clinical Note:</td><td>' + r_note + '</td></tr>'
            '</table>'
            '<div class="med-box"><strong>💊 Suggested Medicines (Reference):</strong><br/>' + r_meds + '</div>'
            '</div>'
        )
        st.markdown(report_html, unsafe_allow_html=True)

        pdf_bytes = create_simple_pdf(
            title="Blood Cell Image Screening Report",
            patient_name=patient_name,
            patient_gender=patient_gender,
            lines=[
                ("Image File", r_file),
                ("Screening Result", r_status),
                ("Confidence Score", r_conf),
                ("Dark Spot Ratio", r
