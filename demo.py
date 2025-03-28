#!/usr/bin/env python3
"""
OCR-NLP Pipeline Demo
---------------------
Standalone demo script for the OCR-NLP Pipeline that showcases the functionality
using pre-processed sample outputs without requiring all dependencies.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo(sample_name=None):
    """
    Run a demonstration using pre-processed sample documents.
    
    Args:
        sample_name: Name of the specific sample to demo (e.g., 'invoice_1')
                    If None, runs demo for all available samples.
    
    Returns:
        True if successful, False otherwise
    """
    base_dir = Path(__file__).parent
    samples_dir = base_dir / "data" / "pdf_samples"
    outputs_dir = base_dir / "data" / "sample_outputs"
    
    if not samples_dir.exists() or not outputs_dir.exists():
        logger.error("Sample data directories not found. Please run scripts/download_sample_pdfs.py first.")
        return False
    
    # Get available samples
    sample_pdfs = list(samples_dir.glob("*.pdf"))
    sample_outputs = list(outputs_dir.glob("*_results.json"))
    
    if not sample_pdfs or not sample_outputs:
        logger.error("No sample PDFs or pre-processed outputs found.")
        logger.info("Please run scripts/download_sample_pdfs.py and scripts/generate_sample_outputs.py first.")
        return False
    
    # Filter by sample name if specified
    if sample_name:
        sample_pdfs = [pdf for pdf in sample_pdfs if sample_name in pdf.stem]
        sample_outputs = [out for out in sample_outputs if sample_name in out.stem]
        
        if not sample_pdfs or not sample_outputs:
            logger.error(f"Sample '{sample_name}' not found.")
            return False
    
    # Display demo for each sample
    for pdf_path in sample_pdfs:
        output_path = outputs_dir / f"{pdf_path.stem}_results.json"
        
        if not output_path.exists():
            logger.warning(f"No pre-processed output found for {pdf_path.name}, skipping.")
            continue
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DEMO: {pdf_path.name}")
        logger.info(f"{'=' * 80}")
        
        # Load pre-processed results
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Display document info
        logger.info(f"Document: {result['document']['name']}")
        logger.info(f"Type: {result['document']['type']}")
        logger.info(f"Pages: {result['document']['pages']}")
        
        # Display OCR sample (first 200 chars)
        ocr_text = result['ocr']['text']
        logger.info(f"\nOCR Text Sample:")
        logger.info(f"{ocr_text[:200]}...")
        
        # Display text analysis
        logger.info(f"\nText Analysis:")
        logger.info(f"Word Count: {result['text_analysis']['word_count']}")
        logger.info(f"Sentences: {result['text_analysis']['sentences']}")
        logger.info(f"Paragraphs: {result['text_analysis']['paragraphs']}")
        
        # Display layout analysis
        logger.info(f"\nLayout Analysis:")
        for key, value in result['text_analysis']['layout'].items():
            logger.info(f"  {key}: {value}")
        
        # Display entities
        logger.info(f"\nExtracted Entities ({len(result['entities'])}):")
        
        # Group entities by type
        entities_by_type = {}
        for entity in result['entities']:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Display entities by type
        for entity_type, entities in sorted(entities_by_type.items()):
            logger.info(f"\n  {entity_type} ({len(entities)}):")
            for entity in entities[:3]:  # Show up to 3 entities per type
                logger.info(f"    - {entity['text']} (confidence: {entity['confidence']})")
            
            if len(entities) > 3:
                logger.info(f"    - ... and {len(entities) - 3} more")
        
        # Display processing times
        logger.info(f"\nProcessing Times:")
        logger.info(f"OCR: {result['ocr']['processing_time']:.2f} seconds")
        logger.info(f"Text Analysis: {result['text_analysis']['processing_time']:.2f} seconds")
        logger.info(f"Entity Extraction: {result['entity_extraction']['processing_time']:.2f} seconds")
        logger.info(f"Total: {result['total_processing_time']:.2f} seconds")
        
        logger.info(f"\nFull results available at: {output_path}")
        logger.info(f"{'=' * 80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="OCR-NLP Pipeline Demo")
    
    # Define arguments
    parser.add_argument("sample", nargs="?", help="Name of specific sample to demo (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run demo
    success = run_demo(args.sample)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
