import os
import requests

# Save to: ocr_nlp_pipeline/data/pdf_samples/
output_dir = os.path.join("data", "pdf_samples")
os.makedirs(output_dir, exist_ok=True)

sample_urls = {
    "invoice_1": "https://github.com/ArkhamArchivist/InvoiceParserData/raw/main/data/invoices/invoice1.pdf",
    "invoice_2": "https://github.com/ArkhamArchivist/InvoiceParserData/raw/main/data/invoices/invoice2.pdf",
    "form_uscis": "https://www.uscis.gov/sites/default/files/document/forms/i-9-paper-version.pdf"
}

for name, url in sample_urls.items():
    file_path = os.path.join(output_dir, f"{name}.pdf")
    print(f"Downloading {name}...")
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

print("✅ Download complete.")
