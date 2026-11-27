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
    </style>
""", unsafe_allow_html=True)


def create_simple_pdf(title, patient_name, lines):
    """Pure Python PDF generator (no fpdf or external library needed)."""
    p_name = patient_name.strip() if patient_name and patient_name.strip() else "Not Specified"
    all_lines = [
        "MALARIA AI DIAGNOSTIC REPORT",
        title,
        "------------------------------------------------------------",
        f"Patient Name: {p_name}",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "------------------------------------------------------------",
    ] + [f"{k}: {v}" for k, v in lines] + [
        "------------------------------------------------------------",
        "Disclaimer: Automated AI screening report for educational",
        "and preliminary screening purposes only. Consult a doctor.",
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
            content_ops.append(f"({safe_line}) Tj")
            content_ops.append("/F1 11 Tf")
        else:
            content_ops.append("T*")
            content_ops.append(f"({safe_line}) Tj")
    content_ops.append("ET")
    stream_data = "\n".join(content_ops).encode("ascii")

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>\nendobj\n"
    )
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")
    objects.append(
        f"6 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
        + stream_data
        + b"\nendstream\nendobj\n"
    )

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj)

    xref_pos = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf_bytes.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("ascii")
    )
    return bytes(pdf_bytes)


st.title("🔬 Malaria Detector App")
st.caption("Code-based detector with instant PDF report download.")

patient_name = st.text_input("Patient Name (Optional)", placeholder="Patient ka naam likhein...")

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
                "confidence": f"{confidence}%",
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
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        if patient_name:
            st.write(f"**Patient:** {patient_name}")
        if rep["is_parasitized"]:
            st.error(f"⚠️ **{rep['status']}** — Confidence: {rep['confidence']}")
        else:
            st.success(f"✅ **{rep['status']}** — Confidence: {rep['confidence']}")
        st.write(f"**Image File:** {rep['filename']}")
        st.write(f"**Dark Staining Spot Ratio:** {rep['spot_ratio']}")
        st.write(rep["note"])
        st.markdown('</div>', unsafe_allow_html=True)

        pdf_bytes = create_simple_pdf(
            title="Blood Cell Image Screening Report",
            patient_name=patient_name,
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
            file_name=f"malaria_image_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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
            "temp": f"{temp}
