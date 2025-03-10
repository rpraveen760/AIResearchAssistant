import os
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with error handling."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
        return ""

# Path to your PDF
pdf_path = "AttentionIsAllYouNeed.pdf"

if os.path.exists(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    # Save extracted text to a file
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(pdf_text)

    # Print first 1000 characters to verify
    print(pdf_text[:1000])
else:
    print(f"❌ PDF not found: {pdf_path}")
