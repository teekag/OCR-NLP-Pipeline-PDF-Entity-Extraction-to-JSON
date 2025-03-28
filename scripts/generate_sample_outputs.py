#!/usr/bin/env python3
"""
Sample Output Generator
----------------------
Generates sample JSON outputs for the downloaded PDFs to demonstrate
the pipeline's capabilities without requiring the full OCR and NLP stack.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime

# Sample entity types and their formats
ENTITY_TYPES = {
    "PERSON": ["John Smith", "Jane Doe", "Robert Johnson", "Maria Garcia"],
    "ORG": ["Acme Corporation", "Global Industries Ltd.", "Tech Solutions Inc.", "First National Bank"],
    "DATE": ["2023-01-15", "2023-02-28", "2023-03-10", "2023-04-22"],
    "MONEY": ["$1,234.56", "$987.65", "$2,345.67", "$456.78"],
    "ADDRESS": ["123 Main St, New York, NY 10001", "456 Oak Ave, Los Angeles, CA 90001"],
    "INVOICE_NUM": ["INV-2023-001", "INV-2023-002", "INV-2023-003", "INV-2023-004"],
    "TAX_ID": ["TAX-123-45-6789", "TAX-987-65-4321", "TAX-456-78-9012", "TAX-789-01-2345"],
    "PRODUCT": ["Widget A", "Service Package B", "Premium Subscription", "Consulting Hours"],
    "QUANTITY": ["5", "10", "2", "1"],
    "TOTAL": ["$6,172.80", "$9,876.50", "$2,345.67", "$456.78"]
}

def generate_entity(entity_type):
    """Generate a random entity of the specified type."""
    values = ENTITY_TYPES.get(entity_type, ["Unknown"])
    value = random.choice(values)
    
    return {
        "text": value,
        "type": entity_type,
        "source": random.choice(["spacy", "rule_based", "transformer"]),
        "confidence": round(random.uniform(0.75, 0.99), 2),
        "start_char": random.randint(100, 5000),
        "end_char": random.randint(5001, 10000)
    }

def generate_invoice_entities():
    """Generate a set of entities typical for an invoice."""
    entities = []
    
    # Add invoice-specific entities
    for entity_type in ["INVOICE_NUM", "DATE", "ORG", "PERSON", "MONEY", "TAX_ID"]:
        # Add 1-2 instances of each entity type
        for _ in range(random.randint(1, 2)):
            entities.append(generate_entity(entity_type))
    
    # Add multiple line items
    for _ in range(random.randint(3, 5)):
        entities.append(generate_entity("PRODUCT"))
        entities.append(generate_entity("QUANTITY"))
        entities.append(generate_entity("MONEY"))
    
    # Add total
    entities.append(generate_entity("TOTAL"))
    
    return entities

def generate_form_entities():
    """Generate a set of entities typical for a form."""
    entities = []
    
    # Add form-specific entities
    for entity_type in ["PERSON", "DATE", "ADDRESS", "ORG", "TAX_ID"]:
        # Add 1-3 instances of each entity type
        for _ in range(random.randint(1, 3)):
            entities.append(generate_entity(entity_type))
    
    return entities

def generate_sample_output(pdf_path, output_dir):
    """Generate a sample JSON output for the given PDF."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine entity type based on filename
    if "invoice" in pdf_path.stem.lower():
        entities = generate_invoice_entities()
        doc_type = "invoice"
    else:
        entities = generate_form_entities()
        doc_type = "form"
    
    # Generate processing times
    ocr_time = round(random.uniform(1.5, 3.0), 2)
    nlp_time = round(random.uniform(0.8, 1.5), 2)
    entity_time = round(random.uniform(0.5, 1.2), 2)
    total_time = ocr_time + nlp_time + entity_time
    
    # Create sample OCR text
    if doc_type == "invoice":
        ocr_text = f"""INVOICE
Invoice Number: {next((e['text'] for e in entities if e['type'] == 'INVOICE_NUM'), 'INV-2023-001')}
Date: {next((e['text'] for e in entities if e['type'] == 'DATE'), '2023-01-15')}

Billed To:
{next((e['text'] for e in entities if e['type'] == 'PERSON'), 'John Smith')}
{next((e['text'] for e in entities if e['type'] == 'ORG'), 'Acme Corporation')}
{next((e['text'] for e in entities if e['type'] == 'ADDRESS'), '123 Main St, New York, NY 10001')}

Description                     Quantity    Price       Amount
{next((e['text'] for e in entities if e['type'] == 'PRODUCT'), 'Widget A')}                        5          $123.45     $617.25
Premium Service                 2          $500.00     $1,000.00
Consulting                      10         $150.00     $1,500.00

Subtotal: $3,117.25
Tax (10%): $311.73
Total: {next((e['text'] for e in entities if e['type'] == 'TOTAL'), '$3,428.98')}
"""
    else:
        ocr_text = f"""EMPLOYMENT ELIGIBILITY VERIFICATION
U.S. Department of Homeland Security

Employee Info (To be completed by employee)
Full Name: {next((e['text'] for e in entities if e['type'] == 'PERSON'), 'John Smith')}
Address: {next((e['text'] for e in entities if e['type'] == 'ADDRESS'), '123 Main St, New York, NY 10001')}
Date of Birth: {next((e['text'] for e in entities if e['type'] == 'DATE'), '1980-01-15')}
SSN: {next((e['text'] for e in entities if e['type'] == 'TAX_ID'), 'XXX-XX-XXXX')}

Employer Info (To be completed by employer)
Company Name: {next((e['text'] for e in entities if e['type'] == 'ORG'), 'Acme Corporation')}
Address: 789 Corporate Blvd, Business City, NY 12345
Date: {next((e['text'] for e in entities if e['type'] == 'DATE'), '2023-01-15')}
"""
    
    # Create the result dictionary
    result = {
        'document': {
            'path': str(pdf_path),
            'name': pdf_path.name,
            'type': 'pdf',
            'size': os.path.getsize(pdf_path),
            'pages': random.randint(1, 3)
        },
        'ocr': {
            'text': ocr_text,
            'processing_time': ocr_time
        },
        'text_analysis': {
            'word_count': len(ocr_text.split()),
            'sentences': random.randint(10, 20),
            'paragraphs': random.randint(5, 10),
            'layout': {
                'has_tables': doc_type == 'invoice',
                'has_forms': doc_type == 'form',
                'has_headers': True,
                'has_footers': doc_type == 'invoice'
            },
            'processing_time': nlp_time
        },
        'entities': entities,
        'entity_extraction': {
            'count': len(entities),
            'processing_time': entity_time
        },
        'total_processing_time': total_time
    }
    
    # Save to JSON file
    output_path = output_dir / f"{pdf_path.stem}_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Generated sample output for {pdf_path.name} at {output_path}")
    return output_path

def main():
    """Process all PDFs in the pdf_samples directory."""
    base_dir = Path(__file__).parent.parent
    pdf_dir = base_dir / "data" / "pdf_samples"
    output_dir = base_dir / "data" / "sample_outputs"
    
    # Find all PDFs
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_paths:
        print("No PDFs found in the pdf_samples directory.")
        return
    
    print(f"Found {len(pdf_paths)} PDFs to process.")
    
    # Generate sample outputs for each PDF
    for pdf_path in pdf_paths:
        generate_sample_output(pdf_path, output_dir)
    
    print(f"✅ Sample outputs generated successfully in {output_dir}")

if __name__ == "__main__":
    main()
