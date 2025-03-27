# OCR-NLP Pipeline: PDF + Entity Extraction to JSON

A production-grade pipeline for extracting structured information from unstructured documents using OCR and NLP techniques.

## Overview

This project implements an end-to-end document processing system that:

1. Takes scanned PDFs or images as input
2. Extracts text using advanced OCR techniques
3. Processes and cleans the extracted text
4. Identifies and extracts entities and key information
5. Outputs structured data in JSON format

The pipeline is designed to be modular, extensible, and suitable for processing various document types including forms, invoices, receipts, contracts, and more.

## Features

- **Multi-Engine OCR Support**: Integrates with both Tesseract and EasyOCR for optimal text extraction
- **Advanced Preprocessing**: Image enhancement and noise reduction for improved OCR accuracy
- **Robust Text Processing**: Cleaning, normalization, and layout analysis
- **Flexible Entity Extraction**: Support for multiple NER engines:
  - spaCy for general-purpose entity recognition
  - Flair for state-of-the-art sequence labeling
  - Transformers for deep learning-based extraction
  - Custom rule-based extractors for domain-specific entities
- **Configurable Pipeline**: Builder pattern for easy pipeline customization
- **Batch Processing**: Efficiently process multiple documents
- **Detailed Output**: Comprehensive JSON output with entities, metadata, and confidence scores

## Project Structure

```
ocr-nlp-pipeline/
├── data/
│   ├── raw/                # Original scanned PDFs or images
│   ├── processed/          # OCR-cleaned and structured text
│   └── samples/            # Sample documents and expected outputs
├── src/
│   ├── ocr_engine.py       # Tesseract or EasyOCR integration
│   ├── nlp_parser.py       # NLP preprocessing logic
│   ├── entity_extractor.py # NER using spaCy, Flair, or transformers
│   └── pipeline.py         # End-to-end document processing orchestration
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Visual demo of OCR and text cleaning
│   └── 02_pipeline_demo.ipynb    # Pipeline from PDF to JSON
├── outputs/                # Final structured outputs (e.g., JSON files)
├── diagrams/               # Dataflow, pipeline architecture, or model visuals
├── tests/                  # Unit and integration tests
├── README.md               # This documentation
└── requirements.txt        # All Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocr-nlp-pipeline.git
   cd ocr-nlp-pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install language models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Install Tesseract OCR (if not already installed):
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Basic Usage

```python
from pathlib import Path
from src.pipeline import PipelineBuilder

# Create a pipeline with default configuration
pipeline = PipelineBuilder().build()

# Process a single document
result = pipeline.process_document(
    "data/samples/invoice.pdf",
    output_dir="outputs"
)

# Access extracted entities
for entity in result['entities']:
    print(f"{entity['text']} ({entity['label']})")
```

### Custom Pipeline Configuration

```python
# Build a pipeline with custom configuration
pipeline = (
    PipelineBuilder()
    .with_ocr_engine('tesseract', lang='eng')
    .with_ocr_dpi(300)
    .with_language('en')
    .with_entity_extractor('spacy', model_name='en_core_web_sm')
    .with_entity_extractor('custom', patterns={
        'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
        'URL': [r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+']
    })
    .build()
)

# Process a batch of documents
document_paths = [
    "data/samples/invoice1.pdf",
    "data/samples/contract.pdf",
    "data/samples/receipt.jpg"
]

results = pipeline.process_batch(
    document_paths,
    output_dir="outputs",
    preprocess_ocr=True,
    clean_text=True
)
```

## Example Output

```json
{
  "document": {
    "path": "data/samples/invoice.pdf",
    "name": "invoice.pdf",
    "type": "pdf",
    "size": 245678,
    "pages": 2
  },
  "ocr": {
    "text": "Invoice #12345\nDate: 2023-03-21\nVendor: Acme Corp\n...",
    "processing_time": 1.25
  },
  "text_analysis": {
    "word_count": 156,
    "sentences": 12,
    "paragraphs": 5,
    "layout": {
      "tables": [
        {
          "start_line": 10,
          "end_line": 15,
          "content": "Item    Quantity    Price\nWidget A    5    $10.00\n..."
        }
      ],
      "lists": [],
      "structure": "tabular"
    },
    "processing_time": 0.35
  },
  "entities": [
    {
      "text": "12345",
      "label": "INVOICE_NUMBER",
      "start_char": 8,
      "end_char": 13,
      "source": "custom_pattern"
    },
    {
      "text": "2023-03-21",
      "label": "DATE",
      "start_char": 20,
      "end_char": 30,
      "source": "custom_pattern"
    },
    {
      "text": "Acme Corp",
      "label": "ORG",
      "start_char": 38,
      "end_char": 47,
      "source": "spacy"
    }
  ],
  "entity_extraction": {
    "count": 3,
    "processing_time": 0.42
  },
  "total_processing_time": 2.02
}
```

## Extending the Pipeline

### Adding Custom Entity Types

Create a custom entity extractor with domain-specific patterns:

```python
custom_patterns = {
    'INVOICE_NUMBER': [r'Invoice\s+#?\s*(\d+)', r'INV\s*#?\s*(\d+)'],
    'TOTAL_AMOUNT': [r'Total:?\s*\$?(\d+\.\d{2})'],
    'CREDIT_CARD': [r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}']
}

pipeline = (
    PipelineBuilder()
    .with_entity_extractor('custom', patterns=custom_patterns)
    .build()
)
```

### Implementing Custom Document Types

For specialized document types, extend the pipeline with custom processing:

```python
# Example for invoice processing
class InvoiceProcessor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def process_invoice(self, invoice_path):
        # Extract basic entities
        result = self.pipeline.process_document(invoice_path)
        
        # Extract invoice-specific information
        invoice_data = {
            'invoice_number': self._find_entity(result, 'INVOICE_NUMBER'),
            'date': self._find_entity(result, 'DATE'),
            'vendor': self._find_entity(result, 'ORG'),
            'total_amount': self._find_entity(result, 'TOTAL_AMOUNT'),
            'line_items': self._extract_line_items(result)
        }
        
        return invoice_data
    
    def _find_entity(self, result, entity_type):
        for entity in result['entities']:
            if entity['label'] == entity_type:
                return entity['text']
        return None
    
    def _extract_line_items(self, result):
        # Custom logic to extract line items from tables
        # ...
        return []
```

## Performance Considerations

- **OCR Quality**: The quality of OCR significantly impacts downstream NLP tasks. Use high-resolution inputs when possible.
- **Memory Usage**: Processing large documents may require significant memory. Consider chunking for very large documents.
- **Processing Speed**: For batch processing of many documents, consider parallel processing.
- **Model Size**: Larger NLP models provide better accuracy but require more resources. Choose models appropriate for your hardware.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [spaCy](https://spacy.io/)
- [Flair](https://github.com/flairNLP/flair)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
