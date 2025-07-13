import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

if __name__ == "__main__":
    raw_text = extract_text_from_pdf("data/iso27001.pdf")
    with open("data/iso_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    print("✅ Text extracted and saved to data/iso_text.txt")
