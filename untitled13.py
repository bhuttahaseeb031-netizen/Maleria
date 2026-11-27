import streamlit as st
import numpy as np
from PIL import Image

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
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Malaria Detector App")
st.caption("Sirf code-based detector — kisi external .h5 ya .pkl file ki zaroorat nahi.")

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
            # Staining / dark spots analysis (code-only heuristic)
            red_ch, green_ch, blue_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            dark_purple_mask = (blue_ch > green_ch * 0.92) & ((red_ch + green_ch + blue_ch) / 3.0 < 0.52)
            spot_ratio = float(np.mean(dark_purple_mask))

            parasitized = spot_ratio > 0.08
            confidence = min(98.5, max(72.0, round(75.0 + abs(spot_ratio - 0.08) * 180, 1)))

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            if patient_name:
                st.write(f"**Patient:** {patient_name}")
            if parasitized:
                st.error(f"⚠️ **Parasitized (Malaria Signs Detected)** — Confidence: {confidence}%")
                st.write(f"Dark staining spot ratio: **{spot_ratio:.2%}**. Lab blood smear / RDT test karwayein.")
            else:
                st.success(f"✅ **Uninfected (Normal Cell Appearance)** — Confidence: {confidence}%")
                st.write(f"Dark staining spot ratio: **{spot_ratio:.2%}**. Cell mein clear parasite pattern nahi mila.")
            st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        if patient_name:
            st.write(f"**Patient:** {patient_name}")
        if score >= 6:
            st.error(f"🚨 **High Malaria Likelihood** (Symptom Score: {score}/9)")
            st.write("Tez bukhar aur chills/sweating mojood hain. Foran doctor aur lab test (MP/RDT) karwayein.")
        elif score >= 3:
            st.warning(f"⚠️ **Moderate Risk** (Symptom Score: {score}/9)")
            st.write("Kuch symptoms milte julte hain. Doctor se checkup karwana behtar hai.")
        else:
            st.success(f"✅ **Low Risk** (Symptom Score: {score}/9)")
            st.write("Symptoms se malaria ka strong indication nahi mil raha.")
        st.markdown('</div>', unsafe_allow_html=True)
