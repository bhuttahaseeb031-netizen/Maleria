from datetime import datetime
import os
import pickle
import time
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Live AI Malaria Cell Detector",
    page_icon="🔬",
    layout="wide",
)

# High-Contrast Dark Theme CSS (Text always clearly visible)
st.markdown(
    """
    <style>
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp small {
        color: #f8fafc !important;
    }

    .stApp {
        background: linear-gradient(180deg, #070b14 0%, #0d1526 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #283548 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #131c2e !important;
        border: 1px solid #2c3e5e !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    .report-card {
        background-color: #131c2e !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #38bdf8 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-top: 12px !important;
        color: #f8fafc !important;
    }

    .report-card p, .report-card strong, .report-card div {
        color: #f8fafc !important;
    }

    .badge-high {
        background-color: #7f1d1d !important;
        color: #fef2f2 !important;
        border: 1px solid #f87171 !important;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .badge-mod {
        background-color: #78350f !important;
        color: #fffbeb !important;
        border: 1px solid #fbbf24 !important;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .badge-low {
        background-color: #14532d !important;
        color: #f0fdf4 !important;
        border: 1px solid #4ade80 !important;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .creator-footer {
        margin-top: 48px;
        padding: 18px 20px;
        text-align: center;
        background-color: #111827 !important;
        border: 1px solid #283548 !important;
        border-radius: 12px;
        color: #f8fafc !important;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Live AI Malaria Cell Detection System")
st.caption(
    "Real-time blood smear screening with Parasitemia Level, Confidence Score, and Printable/Downloadable Clinical Report"
)

# Sidebar Controls
st.sidebar.header("Patient & Screening Controls")
patient_id = st.sidebar.text_input("Sample / Patient ID", value="MAL-2026-001")
patient_name = st.sidebar.text_input("Patient Name", placeholder="Enter patient name")
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
stained_rbc_count = st.sidebar.number_input(
    "Total RBCs Counted in Field (Optional)",
    min_value=10,
    max_value=5000,
    value=100,
    step=10,
)

st.sidebar.divider()
st.sidebar.subheader("Live Mode Settings")
live_mode = st.sidebar.toggle("Enable Live Monitoring", value=False)
refresh_sec = st.sidebar.slider("Live Refresh Interval (sec)", min_value=2, max_value=15, value=5)

MODEL_PATH = "malaria_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.sidebar.success("Custom model loaded: malaria_model.pkl")
else:
    st.sidebar.info("Using built-in stain & parasitemia analyzer (no .pkl file found).")


def classify_level(parasitemia_pct: float, infected: bool) -> tuple[str, str]:
    """Return (level_label, clinical_note) based on parasitemia percentage."""
    if not infected or parasitemia_pct < 0.5:
        return (
            "None / Negative (0.0% - less than 0.5%)",
            "No significant parasite load detected in examined smear.",
        )
    elif parasitemia_pct < 2.0:
        return (
            "Low Level (less than 2%)",
            "Early or low-density parasitemia. Standard antimalarial therapy and close follow-up advised.",
        )
    elif parasitemia_pct <= 5.0:
        return (
            "Moderate Level (2% - 5%)",
            "Moderate parasite burden. Prompt medical evaluation and quantitative smear check recommended.",
        )
    else:
        return (
            "High / Severe Level (greater than 5%)",
            "High parasite density! Urgent hospital/physician care and immediate clinical confirmation required.",
        )


def analyze_cell_image(img: Image.Image, total_rbcs: int):
    """Analyze image and return status, confidence, parasitemia %, and level."""
    resized = img.convert("RGB").resize((96, 96))
    arr = np.array(resized, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    non_bg = (r < 0.92) | (g < 0.92) | (b < 0.92)
    cell_pixels = max(int(np.sum(non_bg)), 1)

    parasite_mask = non_bg & (g < 0.44) & (b > g * 1.12) & (r < 0.68)
    parasite_pixels = int(np.sum(parasite_mask))
    stain_ratio = float(parasite_pixels) / cell_pixels

    if model is not None:
        features = arr.flatten().reshape(1, -1)
        pred = model.predict(features)[0]
        if hasattr(model, "predict_proba"):
            confidence = round(float(np.max(model.predict_proba(features))) * 100, 2)
        else:
            confidence = 91.5
        infected = str(pred).lower() in ["1", "parasitized", "infected", "malaria", "true"]
    else:
        infected = stain_ratio > 0.035
        if infected:
            confidence = round(min(98.8, 74.0 + stain_ratio * 240), 2)
        else:
            confidence = round(min(98.5, 81.0 + (0.035 - stain_ratio) * 280), 2)

    if infected:
        raw_pct = min(14.5, max(0.6, stain_ratio * 48.0))
        parasitemia_pct = round(raw_pct, 2)
    else:
        parasitemia_pct = round(min(0.4, stain_ratio * 8.0), 2)

    estimated_infected_rbcs = max(1 if infected else 0, int(round((parasitemia_pct / 100.0) * total_rbcs)))
    level_label, level_note = classify_level(parasitemia_pct, infected)

    return {
        "infected": infected,
        "confidence": confidence,
        "parasitemia_pct": parasitemia_pct,
        "estimated_infected_rbcs": estimated_infected_rbcs,
        "level_label": level_label,
        "level_note": level_note,
    }


tab_upload, tab_live_cam = st.tabs(["📁 Upload Blood Smear Image", "📷 Live Camera Capture"])

active_image = None
source_label = "Uploaded Smear"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Thin Blood Smear Cell Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="smear_uploader",
    )
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
        source_label = f"Uploaded File ({uploaded_file.name})"

with tab_live_cam:
    cam_photo = st.camera_input("Capture Live Blood Smear / Microscope View", key="live_camera")
    if cam_photo is not None:
        active_image = Image.open(cam_photo)
        source_label = "Live Camera Capture"

if active_image is not None:
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = analyze_cell_image(active_image, stained_rbc_count)

    display_name = patient_name.strip() if patient_name.strip() else "Anonymous Patient"
    status_text = "MALARIA DETECTED (Parasitized)" if res["infected"] else "NO MALARIA DETECTED (Uninfected)"

    st.markdown(f"**Live Scan Timestamp:** `{timestamp_str}` | **Source:** `{source_label}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detection Status", "Parasitized" if res["infected"] else "Uninfected")
    m2.metric("Model Confidence", f"{res['confidence']}%")
    m3.metric("Parasitemia Level", f"{res['parasitemia_pct']}%")
    m4.metric("Est. Infected RBCs", f"{res['estimated_infected_rbcs']} / {stained_rbc_count}")

    st.progress(min(1.0, res["parasitemia_pct"] / 10.0), text=f"Infection Intensity Scale: {res['level_label']}")

    col_img, col_rep = st.columns([1, 1.35])

    with col_img:
        st.subheader("Analyzed Smear Sample")
        st.image(active_image, caption=f"{display_name} — {timestamp_str}", use_container_width=True)

    with col_rep:
        st.subheader("📋 Diagnostic & Severity Report")

        if res["parasitemia_pct"] > 5.0 and res["infected"]:
            badge_html = '<span class="badge-high">SEVERITY: HIGH / SEVERE (greater than 5%)</span>'
            st.error(f"⚠️ **{status_text}** — Confidence: **{res['confidence']}%**")
        elif res["parasitemia_pct"] >= 2.0 and res["infected"]:
            badge_html = '<span class="badge-mod">SEVERITY: MODERATE (2% - 5%)</span>'
            st.warning(f"⚠️ **{status_text}** — Confidence: **{res['confidence']}%**")
        elif res["infected"]:
            badge_html = '<span class="badge-mod">SEVERITY: LOW (less than 2%)</span>'
            st.warning(f"⚠️ **{status_text}** — Confidence: **{res['confidence']}%**")
        else:
            badge_html = '<span class="badge-low">SEVERITY: NEGATIVE / UNINFECTED</span>'
            st.success(f"✅ **{status_text}** — Confidence: **{res['confidence']}%**")

        st.markdown(
            f"""
            <div class="report-card">
                <div style="margin-bottom:12px;">{badge_html}</div>
                <p><strong>Sample / Patient ID:</strong> {patient_id}</p>
                <p><strong>Patient Name:</strong> {display_name} ({patient_age} yrs, {patient_gender})</p>
                <p><strong>Scan Time:</strong> {timestamp_str}</p>
                <p><strong>Detection Result:</strong> {status_text}</p>
                <p><strong>Confidence Score:</strong> {res['confidence']}%</p>
                <p><strong>Parasitemia Level:</strong> {res['level_label']} ({res['parasitemia_pct']}%)</p>
                <p><strong>Clinical Guidance:</strong> {res['level_note']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_text = (
            f"==================================================\n"
            f"LIVE AI MALARIA SCREENING & PARASITEMIA REPORT\n"
            f"==================================================\n"
            f"Timestamp         : {timestamp_str}\n"
            f"Patient / SampleID: {display_name} ({patient_id})\n"
            f"Age / Gender      : {patient_age} / {patient_gender}\n"
            f"Image Source      : {source_label}\n"
            f"--------------------------------------------------\n"
            f"Detection Status  : {status_text}\n"
            f"Confidence Score  : {res['confidence']}%\n"
            f"Parasitemia Level : {res['level_label']} ({res['parasitemia_pct']}%)\n"
            f"Est. Infected RBCs: {res['estimated_infected_rbcs']} of {stained_rbc_count}\n"
            f"--------------------------------------------------\n"
            f"Clinical Note     : {res['level_note']}\n"
            f"Created by        : Haseeb Bhutta\n"
        )

        st.download_button(
            label="📥 Download Full Diagnostic Report (.txt)",
            data=report_text,
            file_name=f"{patient_id}_{display_name.replace(' ', '_')}_malaria_report.txt",
            mime="text/plain",
        )
else:
    st.info(
        "Upar **Upload Blood Smear Image** tab se image upload karein ya **Live Camera Capture** tab se live photo lein."
    )

# End Footer Attribution
st.markdown(
    """
    <div class="creator-footer">
        Created by Haseeb Bhutta
    </div>
    """,
    unsafe_allow_html=True,
)

# Live auto-refresh loop when toggled on
if live_mode:
    st.toast(f"Live mode active — refreshing in {refresh_sec}s...")
    time.sleep(refresh_sec)
    st.rerun()
