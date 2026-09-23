import pdfplumber

pdf_path = "data/raw/National Family Health Survey (NFHS-6) 2023-2024 Fact Sheets.pdf"



with pdfplumber.open(pdf_path) as pdf:
    kept, skipped = 0, 0
    for page in pdf.pages:
        text = page.extract_text() or ""
        if len(text) > 100:
            kept += 1
        else:
            skipped += 1
    print(f"Kept: {kept}, Skipped: {skipped}")