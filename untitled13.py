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
    /* Background & Global Text Visibility */
    [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
        color: #f8fafc;
    }
    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.8);
    }
    label, p, span, div, h1, h2, h3, h4 {
        color: #f1f5f9 !important;
    }
    /* Input boxes styling */
    input, select, textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
    }
    /* Hero Banner */
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
    /* Section Cards */
    .section-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-top: 4px solid #14b8a6;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 14px;
    }
    .result-box {
        background-color: #0f172a;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #2dd4bf;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .med-box {
        background-color: #134e4a;
        border: 1px solid #2dd4bf;
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    /* Buttons */
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

    o1 = f"""1 0 obj{nl}<< /Type /Catalog /Pages 2 0 R >>{nl}endobj{nl}""".encode("ascii")
    o2 = f"""2 0 obj{nl}<< /Type /Pages /Kids [3 0 R] /Count 1 >>{nl}endobj{nl}""".encode("ascii")
    o3_a = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
    o3_b = "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    o3 = f"""3 0 obj{nl}{o3_a}{o3_b}{nl}endobj{nl}""".encode("ascii")
    o4 = f"""4 0 obj{nl}<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>{nl}endobj{nl}""".encode("ascii")
    o5 = f"""5 0 obj{nl}<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>{nl}endobj{nl}""".encode("ascii")
    o6 = (
        f"""6 0 obj{nl}<< /Length {len(stream)} >>{nl}stream{nl}""".encode("ascii")
        + stream
        + f"""{nl}endstream{nl}endobj{nl}""".encode("ascii")
    )

    objs = [o1, o2, o3, o4, o5, o6]
    buf = bytearray(f"""%PDF-1.4{nl}""".encode("ascii"))
    offs = [0]
    for obj in objs:
        offs.append(len(buf))
        buf.extend(obj)

    xref = len(buf)
    buf.extend(f"""xref{nl}0 7{nl}""".encode("ascii"))
    buf.extend(f"""0000000000 65535 f {nl}""".encode("ascii"))
    for off in offs[1:]:
        buf.extend(f"""{str(off).zfill(10)} 00000 n {nl}""".encode("ascii"))

    buf.extend(
        f"""trailer{nl}<< /Size 7 /Root 1 0 R >>{nl}startxref{nl}{xref}{nl}%%EOF""".encode("ascii")
    )
    return bytes(buf)


# --- HERO HEADER ---
st.markdown("""
    <div class="hero-banner">
        <h1>🧬 AI Malaria Detection & Medicine System</h1>
        <p>Complete View: Patient Info • Blood Cell Detector • Symptoms & Medicines • PDF Download</p>
    </div>
""", unsafe_allow_html=True)

# --- PATIENT INFO BAR ---
col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    patient_name = st.text_input("👤 Patient Name", placeholder="Patient ka naam likhein...")
with col_p2:
    patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

st.markdown("---")

# --- BOTH DETECTORS VISIBLE SIDE BY SIDE ---
left_col, right_col = st.columns(2, gap="large")

# ==========================================
# LEFT COLUMN: BLOOD CELL IMAGE DETECTOR
# ==========================================
with left_col:
    st.markdown('<div class="section-card"><h3>🩸 1. Blood Cell Image Detector</h3></div>', unsafe_allow_html=True)
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
                "filename": uploaded_file.name,
                "status": "Parasitized (Malaria Detected)" if parasitized else "Uninfected (Normal Cell)",
                "confidence": str(confidence) + "%",
                "spot_ratio": f"{spot_ratio:.2%}",
                "note": note_txt,
                "medicines": med_info,
                "is_parasitized": parasitized,
            }

    if "img_report" in st.session_state:
        rep = st.session_state["img_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"
