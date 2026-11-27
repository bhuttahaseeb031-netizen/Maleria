# --- PDF GENERATOR FUNCTION ---
def create_pdf(name, age_id, diagnosis, confidence, med_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    page_w = pdf.epw  # Exact printable width

    pdf.set_font("Arial", "B", 18)
    pdf.cell(page_w, 12, "Malaria Diagnostic Lab Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "", 12)
    pdf.cell(page_w, 9, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(page_w, 9, f"Patient Name: {name if name else 'N/A'}", ln=True)
    pdf.cell(page_w, 9, f"Patient Age / ID: {age_id if age_id else 'N/A'}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(page_w, 10, f"Diagnosis Result: {diagnosis}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(page_w, 9, f"Model Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(6)

    # Medicine Section in PDF
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 8, med_info["title"])
    pdf.ln(2)

    pdf.set_font("Arial", "", 11)
    for item in med_info["items"]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_w, 7, f"- {item}")
    pdf.ln(3)

    pdf.set_font("Arial", "I", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 7, med_info["note"])
    pdf.ln(8)

    pdf.set_font("Arial", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_w, 6, "Medical Disclaimer: This AI-generated report is for screening and educational purposes only. Do not self-medicate; consult a licensed physician for prescription and dosage.")
    return bytes(pdf.output())
