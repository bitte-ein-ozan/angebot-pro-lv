import pdfplumber
import sys

pdf_path = "/Users/ozan/Desktop/6551200 LV_Industriefußboden.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            print(f"--- Page {i+1} ---")
            text = page.extract_text()
            if text:
                print(text)
            else:
                print("[No text found on this page]")
            print("\n")
except Exception as e:
    print(f"Error reading PDF: {e}")
