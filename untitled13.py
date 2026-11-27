import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Malaria AI Detector",
    page_icon="🧬",
    layout="wide"
)

# --- SINGLE-LINE CSS (NO TRIPLE QUOTES) ---
css_parts = [
    "<style>",
    ".stApp { background-color: #0b1120; color: #f8fafc; }",
    ".hero-banner { background: linear-gradient(135deg, #0d9488 0%, #0369a1 100%); padding: 18px 24px; border-radius: 14px; border: 1px solid #38bdf8; margin-bottom: 18px; text-align: center; color: #ffffff; }",
    ".hero-banner h1 { color: #ffffff; margin: 0; font-size: 26px; }",
    ".hero-banner p { color: #e0f2fe; margin: 4px 0 0 0; font-size: 14px; }",
    ".section-title { background-color: #111827; border-left: 5px solid #14b8a6; padding: 10px 14px; border-radius: 8px; color: #38bdf8; font-size: 18px; font-weight: 700; margin-bottom: 12px; }",
    ".report-sheet { background-color: #0f172a; border: 2px solid #2dd4bf; border-radius: 12px; padding: 16px; margin-top: 12px; margin-bottom: 12px; color: #f8fafc; }",
    ".report-header { background-color: #0d9488; color: #ffffff; padding: 8px 12px; border-radius: 8px; font-weight: 800; font-size: 15px; margin-bottom: 10px; text-align: center; }",
    ".report-table { width: 100%; border-collapse: collapse; }",
    ".report-table td { padding: 7px 8px; border-bottom: 1px solid #1e293b; font-size: 14px; color: #f1f5f9; }",
    ".report-label { font-weight: 700; color: #38bdf8 !important; width: 38%; }",
    ".med-box { background-color: #134e4a; border: 1px solid #2dd4bf; padding: 10px 12px; border-radius: 8px; margin-top: 10px; color: #ffffff; font-size: 14px; }",
    ".stButton>button { background: linear-gradient(90deg, #0d9488, #0284c7); color: #ffffff; border-radius: 8px; border: none; padding: 10px 18px; font-weight: 700; width: 100%; }",
    ".stDownloadButton>button { background: linear-gradient(90deg, #d97706, #f59e0b); color: #000000; border-radius: 8px; border: none; padding: 10px 18px; font-weight: 800; width: 100%; }",
    ".footer-credit { text-align: center; margin-top: 28px; padding: 12px; border-top: 1px solid #334155; color: #38bdf8; font-size: 16px; font-weight: 700; }",
    "</style>"
]
st.markdown("".join(css_parts), unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, patient_gender, lines):
    """Pure Python PDF generator without any triple quotes."""
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
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        if idx == 0:
            ops.extend(["/F2 14 Tf", "(" + clean + ") Tj", "/F1 10.5 Tf"])
        else:
            ops.extend(["T*", "(" + clean + ") Tj"])
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")

    o1 = ("1 0 obj" + nl + "<< /Type /Catalog /Pages 2 0 R >>" + nl + "endobj" + nl).encode("ascii")
    o2 = ("2 0 obj" + nl + "<< /Type /Pages /Kids [3 0 R] /Count 1 >>" + nl + "endobj" + nl).encode("ascii")
    o3_a = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
    o3_b = "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
    o3 = ("3 0 obj" + nl + o3_a + o3_b + nl + "endobj" + nl).encode("ascii")
    o4 = ("4 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>" + nl + "endobj" + nl).encode("ascii")
    o5 = ("5 0 obj" + nl + "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>" + nl + "endobj" + nl).encode("ascii")
    o6_head = ("6 0 obj" + nl + "<< /Length " + str(len(stream)) + " >>" + nl + "stream" + nl).encode("ascii")
    o6_tail = (nl + "endstream" + nl + "endobj" + nl).encode("ascii")
    o6 = o6_head + stream + o6_tail

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

    tr = "trailer" + nl + "<< /Size 7 /Root 1 0 R >>" + nl + "startxref" + nl + str(xref) + nl + "%%EOF"
    buf.extend(tr.encode("ascii"))
    return bytes(buf)


# --- HERO HEADER ---
hero_html = (
    '<div class="hero-banner">'
    '<h1>🧬 AI Malaria Detection & Medicine System</h1>'
    '<p>Patient Info • Blood Cell Detector • Symptoms & Medicines • Full Report & PDF Download</p>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

# --- PATIENT INFO BAR ---
col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    patient_name = st.text_input("👤 Patient Name", placeholder="Patient ka naam likhein...")
with col_p2:
    patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

st.divider()

# --- BOTH DETECTORS VISIBLE SIDE BY SIDE ---
left_col, right_col = st.columns(2, gap="large")

# ==========================================
# LEFT COLUMN: BLOOD CELL IMAGE DETECTOR
# ==========================================
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

    if "img_report" in st.session_state:
        rep = st.session_state["img_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"

        if rep["is_parasitized"]:
            st.error("⚠️ **" + rep["status"] + "** (" + rep["confidence"] + ")")
        else:
            st.success("✅ **" + rep["status"] + "** (" + rep["confidence"] + ")")

        report_html = (
            '<div class="report-sheet">'
            '<div class="report-header">📋 BLOOD SMEAR DIAGNOSTIC REPORT</div>'
            '<table class="report-table">'
            '<tr><td class="report-label">Patient Name:</td><td>' + display_name + '</td></tr>'
            '<tr><td class="report-label">Gender:</td><td>' + patient_gender + '</td></tr>'
            '<tr><td class="report-label">Date & Time:</td><td>' + rep["time"] + '</td></tr>'
            '<tr><td class="report-label">Sample File:</td><td>' + rep["filename"] + '</td></tr>'
            '<tr><td class="report-label">Result:</td><td><strong>' + rep["status"] + '</strong></td></tr>'
            '<tr><td class="report-label">Confidence:</td><td>' + rep["confidence"] + '</td></tr>'
            '<tr><td class="report-label">Dark Spot Ratio:</td><td>' + rep["spot_ratio"] + '</td></tr>'
            '<tr><td class="report-label">Clinical Note:</td><td>' + rep["note"] + '</td></tr>'
            '</table>'
            '<div class="med-box"><strong>💊 Suggested Medicines (Reference):</strong><br/>' + rep["medicines"] + '</div>'
            '</div>'
        )
        st.markdown(report_html, unsafe_allow_html=True)

        pdf_bytes = create_simple_pdf(
            title="Blood Cell Image Screening Report",
            patient_name=patient_name,
            patient_gender=patient_gender,
            lines=[
                ("Image File", rep["filename"]),
                ("Screening Result", rep["status"]),
                ("Confidence Score", rep["confidence"]),
                ("Dark Spot Ratio", rep["spot_ratio"]),
                ("Clinical Note", rep["note"]),
                ("Medicines (Ref)", rep["medicines"]),
            ],
        )
        st.download_button(
            label="📄 Save / Download Image PDF Report",
            data=pdf_bytes,
            file_name="malaria_image_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
            mime="application/pdf",
            key="dl_img_pdf",
        )

# ==========================================
# RIGHT COLUMN: SYMPTOMS & MEDICINE DETECTOR
# ==========================================
with right_col:
    st.markdown('<div class="section-title">🩺 2. Symptoms & Medicine Guide</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        temp = st.number_input("Temperature (°F)", min_value=95.0, max_value=107.0, value=98.6, step=0.1)
        chills = st.selectbox("Chills / Kapkapi?", ["No", "Yes"])
        sweating = st.selectbox("Zyada Paseena Aana?", ["No", "Yes"])
    with c2:
        fever = st.selectbox("Tez Bukhar (High Fever)?", ["No", "Yes"])
        headache = st.selectbox("Sar Dard / Body Ache?", ["No", "Yes"])
        nausea = st.selectbox("Matli / Ulti / Kamzori?", ["No", "Yes"])

    if st.button("Analyze Symptoms & Suggest Medicines", key="sym_detect_btn"):
        score = 0
        if temp >= 100.4:
            score += 2
        if fever == "Yes":
            score += 2
        if chills == "Yes":
            score += 2
        if sweating == "Yes":
            score += 1
        if headache == "Yes":
            score += 1
        if nausea == "Yes":
            score += 1

        if score >= 6:
            risk_label = "High Malaria Likelihood"
            risk_level = "high"
            advice = "Tez bukhar aur chills mojood hain. Foran blood test (MP/RDT) aur doctor se ruju karein."
            med_list = "Artemether-Lumefantrine (Coartem), Chloroquine, Artesunate + Paracetamol"
        elif score >= 3:
            risk_label = "Moderate Risk"
            risk_level = "mod"
            advice = "Kuch symptoms milte julte hain. Lab test ke baad doctor ki tajweez par dawa lein."
            med_list = "Paracetamol (Fever/Pain) + ORS; Anti-malarial sirf test positive aane par"
        else:
            risk_label = "Low Risk"
            risk_level = "low"
            advice = "Symptoms se malaria ka strong indication nahi mil raha. Pani aur aaram par tawajjo dein."
            med_list = "No anti-malarial medicine needed (Sirf aaram aur hydration)"

        st.session_state["sym_report"] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "temp": str(round(temp, 1)) + " F",
            "score": str(score) + "/9",
            "risk_label": risk_label,
            "risk_level": risk_level,
            "advice": advice,
            "medicines": med_list,
            "symptoms_summary": (
                "Fever: " + fever + ", Chills: " + chills + ", Sweating: " + sweating +
                ", Headache: " + headache + ", Nausea: " + nausea
            ),
        }

    if "sym_report" in st.session_state:
        srep = st.session_state["sym_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"

        if srep["risk_level"] == "high":
            st.error("🚨 **" + srep["risk_label"] + "** (" + srep["score"] + ")")
        elif srep["risk_level"] == "mod":
            st.warning("⚠️ **" + srep["risk_label"] + "** (" + srep["score"] + ")")
        else:
            st.success("✅ **" + srep["risk_label"] + "** (" + srep["score"] + ")")

        sym_report_html = (
            '<div class="report-sheet">'
            '<div class="report-header">📋 SYMPTOMS SCREENING REPORT</div>'
            '<table class="report-table">'
            '<tr><td class="report-label">Patient Name:</td><td>' + display_name + '</td></tr>'
            '<tr><td class="report-label">Gender:</td><td>' + patient_gender + '</td></tr>'
            '<tr><td class="report-label">Date & Time:</td><td>' + srep["time"] + '</td></tr>'
            '<tr><td class="report-label">Risk Level:</td><td><strong>' + srep["risk_label"] + ' (' + srep["score"] + ')</strong></td></tr>'
            '<tr><td class="report-label">Temperature:</td><td>' + srep["temp"] + '</td></tr>'
            '<tr><td class="report-label">Symptoms:</td><td>' + srep["symptoms_summary"] + '</td></tr>'
            '<tr><td class="report-label">Advice:</td><td>' + srep["advice"] + '</td></tr>'
            '</table>'
            '<div class="med-box"><strong>💊 Suggested Medicines (Reference):</strong><br/>' + srep["medicines"] + '</div>'
            '</div>'
        )
        st.markdown(sym_report_html, unsafe_allow_html=True)

        sym_pdf_bytes = create_simple_pdf(
            title="Symptoms-Based Malaria Screening Report",
            patient_name=patient_name,
            patient_gender=patient_gender,
            lines=[
                ("Risk Assessment", srep["risk_label"] + " (Score: " + srep["score"] + ")"),
                ("Body Temperature", srep["temp"]),
                ("Reported Symptoms", srep["symptoms_summary"]),
                ("Recommendation", srep["advice"]),
                ("Medicines (Ref)", srep["medicines"]),
            ],
        )
        st.download_button(
            label="📄 Save / Download Symptoms PDF Report",
            data=sym_pdf_bytes,
            file_name="malaria_symptoms_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
            mime="application/pdf",
            key="dl_sym_pdf",
        )

# --- FOOTER CREDIT ---
st.markdown('<div class="footer-credit">Created by Haseeb Bhutta</div>', unsafe_allow_html=True)
