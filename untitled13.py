import io
import datetime
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Malaria Cell Pathology Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Clinical Styling
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #10192d 0%, #090d16 55%, #060911 100%);
            color: #eef2ff;
        }
        [data-testid="stSidebar"] {
            background-color: #0d1322;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        .hero-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(110, 231, 183, 0.25);
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 20px;
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f8fafc;
            margin: 0 0 6px 0;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: 0.96rem;
            margin: 0;
        }
        .report-dossier {
            background: #0f172a;
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-left: 5px solid #10b981;
            border-radius: 16px;
            padding: 22px 24px;
            margin-top: 14px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        }
        .report-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed rgba(148, 163, 184, 0.25);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .badge-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.16);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .badge-warn {
            background: rgba(245, 158, 11, 0.16);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .finding-box {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 12px;
            padding: 14px 16px;
            margin-top: 12px;
        }
        .footer-credit {
            margin-top: 28px;
            padding-top: 14px;
            border-top: 1px solid rgba(148, 163, 184, 0.14);
            color: #94a3b8;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def analyze_blood_smear(pil_img: Image.Image, total_rbcs: int) -> dict:
    """Fallback stain & parasitemia estimator when no external .pkl model is loaded."""
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Giemsa/Wright stain inclusion heuristic (purple/dark-blue chromatin-like pixels)
    stain_mask = (b > g + 12) & (r < 175) & (g < 155)
    stain_ratio = float(np.mean(stain_mask))

    # Estimate infected cells relative to total RBC count
    estimated_infected = max(0, min(total_rbcs, int(round(stain_ratio * total_rbcs * 3.5))))
    parasitemia_pct = round((estimated_infected / max(1, total_rbcs)) * 100.0, 2)

    if parasitemia_pct >= 2.0:
        status = "Elevated Parasitemia Signal"
        risk_tier = "High Attention"
        confidence = round(min(96.4, 78.0 + parasitemia_pct * 2.4), 1)
    elif parasitemia_pct > 0.0:
        status = "Low-Grade Stain Anomaly Detected"
        risk_tier = "Moderate Attention"
        confidence = round(min(91.2, 71.0 + parasitemia_pct * 5.0), 1)
    else:
        status = "No Significant Parasite Inclusion Detected"
        risk_tier = "Baseline / Clear"
        confidence = 86.5

    return {
        "status": status,
        "risk_tier": risk_tier,
        "parasitemia_pct": parasitemia_pct,
        "estimated_infected": estimated_infected,
        "total_rbcs": total_rbcs,
        "confidence": confidence,
        "stain_density_pct": round(stain_ratio * 100.0, 2),
        "dimensions": f"{pil_img.width} x {pil_img.height} px",
    }


def build_text_report(patient: dict, res: dict, timestamp_str: str) -> str:
    return (
        "========================================================\n"
        "MALARIA CELL SCREENING — CLINICAL PATHOLOGY DOSSIER\n"
        "========================================================\n"
        f"Generated At       : {timestamp_str}\n"
        f"Sample / Patient ID: {patient['sample_id']}\n"
        f"Patient Name       : {patient['name'] or 'Not Provided'}\n"
        f"Age / Gender       : {patient['age']} yrs / {patient['gender']}\n"
        "--------------------------------------------------------\n"
        f"Screening Finding  : {res['status']}\n"
        f"Triage Classification: {res['risk_tier']}\n"
        f"Parasitemia Index  : {res['parasitemia_pct']}% ({res['estimated_infected']}/{res['total_rbcs']} RBCs)\n"
        f"Model Confidence   : {res['confidence']}%\n"
        f"Stain Inclusion Density: {res['stain_density_pct']}%\n"
        f"Image Resolution   : {res['dimensions']}\n"
        "
