#!/usr/bin/env python3
"""
OCR-NLP Pipeline CLI Tool
-------------------------
Simple command-line interface for running the OCR-NLP pipeline on documents.

Usage:
    python run_pipeline.py input_file [output_dir]
    python run_pipeline.py --batch input_dir [output_dir]
"""

import os
import sys
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="OCR-NLP Pipeline CLI Tool")
    
    # Define arguments
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", nargs="?", help="Output directory (optional)")
    parser.add_argument("--batch", action="store_true", help="Process all documents in the input directory")
    parser.add_argument("--config", help="Path to pipeline configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process documents
    if args.batch:
        success = process_batch(args.input, args.output, args.config)
    else:
        success = process_single_document(args.input, args.output, args.config)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
