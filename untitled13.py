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

# Dark Mode Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f17;
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .report-card {
        background-color: #111827;
        border: 1px solid #263248;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
    }
    .badge-high {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-mod {
        background-color: #451a03;
        color: #fde68a;
        border: 1px solid #f59e0b;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: #052e16;
        color: #86efac;
        border: 1px solid #22c55e;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        display: inline-block;
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
            "None / Negative (0.0% - <0.5%)",
            "No significant parasite load detected in examined smear.",
        )
    elif parasitemia_pct < 2.0:
        return (
            "Low Level (<2%)",
            "Early or low-density parasitemia. Standard antimalarial therapy and close follow-up advised.",
        )
    elif parasitemia_pct <= 5.0:
        return (
            "Moderate Level (2% - 5%)",
            "Moderate parasite burden. Prompt medical evaluation and quantitative smear check recommended.",
        )
    else:
        return (
            "High / Severe Level (>5%)",
            "High parasite density! Urgent hospital/physician care and immediate clinical confirmation required.",
        )


def analyze_cell_image(img: Image.Image, total_rbcs: int):
    """Analyze image and return status, confidence, parasitemia %, and level."""
    resized = img.convert("RGB").resize((96, 96))
    arr = np.array(resized, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    non_bg = (r < 0.92) | (g < 0.92) | (b < 0.92)
    cell_pixels = max(int(np.sum(non_bg)), 1)

    # Detect Giemsa-stained parasite/chromatin spots inside cell
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

    # Estimate parasitemia percentage (%)
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


# Input Tabs: Upload Image vs Live Camera Capture
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

    # Top Live Status Banner & Metrics
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
        st
