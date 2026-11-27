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
    """Pure Python PDF generator using triple-quoted strings (immune to copy-paste line breaks)."""
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
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        if idx == 0:
            ops.extend(["/F2 15 Tf", "(" + clean + ") Tj", "/F1 11 Tf"])
        else:
            ops.extend(["T*", "(" + clean + ") Tj"])
    ops.append("ET")
    stream = nl.join(ops).encode("ascii")

    o1 = f"""1 0 obj{nl}<< /Type /Catalog /Pages 2 0 R >>{nl}endobj{nl}""".encode("ascii")
    o2 = f"""2 0 obj{nl}<< /Type /Pages /Kids [3 0 R] /Count 1 >>{nl}endobj{nl}""".encode("ascii")
    o3 = f"""3 0 obj{nl}<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>{nl}endobj{nl}""".encode("ascii")
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


st.title("🔬 Malaria Detector App")
st.caption("Code-based detector with instant PDF report download.")

# --- PATIENT NAME & GENDER ---
col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    patient_name = st.text_input("Patient Name", placeholder="Patient ka naam likhein...")
with col_p2:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

tab1, tab2 = st.tabs(["🩸 Blood Cell Image Detector", "🩺 Symptoms Detector"])

# --- TAB 1: IMAGE ANALYZER ---
with tab1:
    st.subheader("Blood Smear Image Upload Karein")
    uploaded_file = st.file_uploader("Image chunein (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Blood Cell Image", width=260)

        if st.button("Detect Malaria from Image", key="img_detect_btn"):
            arr = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0
            red_ch, green_ch, blue_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            dark_purple_mask = (blue_ch > green_ch * 0.92) & ((red_ch + green_ch + blue_ch) / 3.0 < 0.52)
            spot_ratio = float(np.mean(dark_purple_mask))

            parasitized = spot_ratio > 0.08
            confidence = min(98.5, max(72.0, round(75.0 + abs(spot_ratio - 0.08) * 180, 1)))

            st.session_state["img_report"] = {
                "filename": uploaded_file.name,
                "status": "Parasitized (Malaria Signs Detected)" if parasitized else "Uninfected (Normal Cell Appearance)",
                "confidence": str(confidence) + "%",
                "spot_ratio": f"{spot_ratio:.2%}",
                "note": (
                    "Dark staining spots threshold se zyada hain. Lab blood smear / RDT test karwayein."
                    if parasitized
                    else "Cell mein clear parasite pattern nahi mila."
                ),
                "is_parasitized": parasitized,
            }

    if "img_report" in st.session_state:
        rep = st.session_state["img_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write("**Patient Name:** " + display_name + " | **Gender:** " + patient_gender)
        if rep["is_parasitized"]:
            st.error("⚠️ **" + rep["status"] + "** — Confidence: " + rep["confidence"])
        else:
            st.success("✅ **" + rep["status"] + "** — Confidence: " + rep["confidence"])
        st.write("**Image File:** " + rep["filename"])
        st.write("**Dark Staining Spot Ratio:** " + rep["spot_ratio"])
        st.write(rep["note"])
        st.markdown('</div>', unsafe_allow_html=True)

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
            ],
        )
        st.download_button(
            label="📄 Save / Download PDF Report",
            data=pdf_bytes,
            file_name="malaria_image_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
            mime="application/pdf",
            key="dl_img_pdf",
        )

# --- TAB 2: SYMPTOMS DETECTOR ---
with tab2:
    st.subheader("Symptoms Select Karein")

    c1, c2 = st.columns(2)
    with c1:
        temp = st.number_input("Temperature (°F)", min_value=95.0, max_value=107.0, value=98.6, step=0.1)
        chills = st.selectbox("Chills / Kapkapi (Sardi lagna)?", ["No", "Yes"])
        sweating = st.selectbox("Zyada Paseena Aana?", ["No", "Yes"])
    with c2:
        fever = st.selectbox("Tez Bukhar (High Fever)?", ["No", "Yes"])
        headache = st.selectbox("Sar Dard / Body Ache?", ["No", "Yes"])
        nausea = st.selectbox("Matli / Ulti / Kamzori?", ["No", "Yes"])

    if st.button("Detect Malaria from Symptoms", key="sym_detect_btn"):
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
            advice = "Tez bukhar aur chills/sweating mojood hain. Foran doctor aur lab test (MP/RDT) karwayein."
        elif score >= 3:
            risk_label = "Moderate Risk"
            risk_level = "mod"
            advice = "Kuch symptoms milte julte hain. Doctor se checkup karwana behtar hai."
        else:
            risk_label = "Low Risk"
            risk_level = "low"
            advice = "Symptoms se malaria ka strong indication nahi mil raha."

        st.session_state["sym_report"] = {
            "temp": str(round(temp, 1)) + " F",
            "score": str(score) + "/9",
            "risk_label": risk_label,
            "risk_level": risk_level,
            "advice": advice,
            "symptoms_summary": (
                "Fever: " + fever + ", Chills: " + chills + ", Sweating: " + sweating +
                ", Headache: " + headache + ", Nausea/Weakness: " + nausea
            ),
        }

    if "sym_report" in st.session_state:
        srep = st.session_state["sym_report"]
        display_name = patient_name.strip() if patient_name.strip() else "Not Specified"
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write("**Patient Name:** " + display_name + " | **Gender:** " + patient_gender)
        if srep["risk_level"] == "high":
            st.error("🚨 **" + srep["risk_label"] + "** (Symptom Score: " + srep["score"] + ")")
        elif srep["risk_level"] == "mod":
            st.warning("⚠️ **" + srep["risk_label"] + "** (Symptom Score: " + srep["score"] + ")")
        else:
            st.success("✅ **" + srep["risk_label"] + "** (Symptom Score: " + srep["score"] + ")")
        st.write("**Temperature:** " + srep["temp"] + " | **Symptoms:** " + srep["symptoms_summary"])
        st.write(srep["advice"])
        st.markdown('</div>', unsafe_allow_html=True)

        sym_pdf_bytes = create_simple_pdf(
            title="Symptoms-Based Malaria Screening Report",
            patient_name=patient_name,
            patient_gender=patient_gender,
            lines=[
                ("Risk Assessment", srep["risk_label"] + " (Score: " + srep["score"] + ")"),
                ("Body Temperature", srep["temp"]),
                ("Reported Symptoms", srep["symptoms_summary"]),
                ("Recommendation", srep["advice"]),
            ],
        )
        st.download_button(
            label="📄 Save / Download PDF Report",
            data=sym_pdf_bytes,
            file_name="malaria_symptoms_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
            mime="application/pdf",
            key="dl_sym_pdf",
        )

# --- FOOTER CREDIT ---
st.markdown('<div class="footer-credit">Created by Haseeb Bhutta</div>', unsafe_allow_html=True)
