#!/usr/bin/env python3
"""
OCR-NLP Pipeline CLI Tool
-------------------------
Simple command-line interface for running the OCR-NLP pipeline on documents.

Usage:
    python run_pipeline.py input_file [output_dir]
    python run_pipeline.py --batch input_dir [output_dir]
    python run_pipeline.py --demo [sample_name]
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

from src.pipeline import PipelineBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_document(input_path, output_dir=None, config=None):
    """Process a single document through the pipeline."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False
    
    if output_dir is None:
        output_dir = Path("outputs")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline with default or custom config
    if config is None:
        pipeline = PipelineBuilder().build()
    else:
        pipeline = PipelineBuilder().build()
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing document: {input_path}")
        
        result = pipeline.process_document(
            input_path,
            output_dir=output_dir,
            preprocess_ocr=True,
            clean_text=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        entity_count = result.get('entity_extraction', {}).get('count', 0)
        
        logger.info(f"Completed in {processing_time:.2f} seconds")
        logger.info(f"Extracted {entity_count} entities")
        logger.info(f"Results saved to {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False


def process_batch(input_dir, output_dir=None, config=None):
    """Process all documents in a directory."""
    input_dir = Path(input_dir)
    
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    if output_dir is None:
        output_dir = Path("outputs")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all documents
    document_paths = []
    for ext in ['pdf', 'png', 'jpg', 'jpeg', 'tiff']:
        document_paths.extend(list(input_dir.glob(f"*.{ext}")))
    
    if not document_paths:
        logger.error(f"No documents found in {input_dir}")
        return False
    
    logger.info(f"Found {len(document_paths)} documents to process")
    
    # Create pipeline
    if config is None:
        pipeline = PipelineBuilder().build()
    else:
        pipeline = PipelineBuilder().build()
    
    try:
        start_time = datetime.now()
        
        results = pipeline.process_batch(
            document_paths,
            output_dir=output_dir,
            preprocess_ocr=True,
            clean_text=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        total_entities = sum(r.get('entity_extraction', {}).get('count', 0) for r in results)
        
        logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
        logger.info(f"Processed {len(results)} documents")
        logger.info(f"Extracted {total_entities} entities in total")
        logger.info(f"Results saved to {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return False


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
        for entity_type, entities in entities_by_type.items():
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
    parser = argparse.ArgumentParser(description="OCR-NLP Pipeline CLI Tool")
    
    # Define arguments
    parser.add_argument("input", nargs="?", help="Input file or directory")
    parser.add_argument("output", nargs="?", help="Output directory (optional)")
    parser.add_argument("--batch", action="store_true", help="Process all documents in the input directory")
    parser.add_argument("--config", help="Path to pipeline configuration file")
    parser.add_argument("--demo", nargs="?", const=True, help="Run demonstration using sample documents")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run demo mode if specified
    if args.demo is not None:
        sample_name = None if args.demo is True else args.demo
        success = run_demo(sample_name)
        sys.exit(0 if success else 1)
    
    # Ensure input is provided for normal operation
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # Process documents
    if args.batch:
        success = process_batch(args.input, args.output, args.config)
    else:
        success = process_single_document(args.input, args.output, args.config)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
