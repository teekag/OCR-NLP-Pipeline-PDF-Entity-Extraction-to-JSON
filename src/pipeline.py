"""
Pipeline Module
--------------
Orchestrates the end-to-end document processing pipeline.
Combines OCR, text preprocessing, and entity extraction.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime

from .ocr_engine import DocumentProcessor, get_ocr_engine
from .nlp_parser import TextPreprocessor, LayoutAnalyzer
from .entity_extractor import EntityExtractorFactory, EntityMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end document processing pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set up OCR engine
        ocr_config = self.config.get('ocr', {})
        ocr_engine_type = ocr_config.get('engine', 'tesseract')
        ocr_engine_params = ocr_config.get('params', {})
        
        self.ocr_engine = get_ocr_engine(ocr_engine_type, **ocr_engine_params)
        self.document_processor = DocumentProcessor(
            self.ocr_engine, 
            dpi=ocr_config.get('dpi', 300)
        )
        
        # Set up text preprocessor
        nlp_config = self.config.get('nlp', {})
        self.text_preprocessor = TextPreprocessor(
            language=nlp_config.get('language', 'en')
        )
        self.layout_analyzer = LayoutAnalyzer()
        
        # Set up entity extractors
        entity_config = self.config.get('entity_extraction', {})
        self.extractors = []
        
        for extractor_config in entity_config.get('extractors', []):
            extractor_type = extractor_config.get('type')
            extractor_params = extractor_config.get('params', {})
            
            if extractor_type:
                try:
                    extractor = EntityExtractorFactory.create_extractor(
                        extractor_type, 
                        **extractor_params
                    )
                    self.extractors.append(extractor)
                except Exception as e:
                    logger.error(f"Failed to create extractor {extractor_type}: {str(e)}")
        
        # If no extractors specified, use spaCy as default
        if not self.extractors:
            self.extractors = [EntityExtractorFactory.create_extractor('spacy')]
        
        # Set up entity merger
        self.entity_merger = EntityMerger(
            self.extractors,
            priority_order=entity_config.get('priority_order')
        )
        
        logger.info(f"Initialized pipeline with {len(self.extractors)} entity extractors")
    
    def process_document(self, document_path: Union[str, Path], 
                        output_dir: Union[str, Path] = None,
                        preprocess_ocr: bool = True,
                        clean_text: bool = True) -> Dict[str, Any]:
        """
        Process a document through the entire pipeline.
        
        Args:
            document_path: Path to the document
            output_dir: Directory to save outputs
            preprocess_ocr: Whether to preprocess images before OCR
            clean_text: Whether to clean text after OCR
            
        Returns:
            Dictionary with processing results
        """
        document_path = Path(document_path)
        
        if not document_path.exists():
            logger.error(f"Document not found: {document_path}")
            return {'error': f"Document not found: {document_path}"}
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: OCR
        logger.info(f"Processing document: {document_path}")
        ocr_start_time = datetime.now()
        
        ocr_result = self.document_processor.process_document(
            document_path, 
            preprocess=preprocess_ocr
        )
        
        ocr_time = (datetime.now() - ocr_start_time).total_seconds()
        logger.info(f"OCR completed in {ocr_time:.2f} seconds")
        
        # Handle multi-page documents
        if isinstance(ocr_result, list):
            pages = ocr_result
            full_text = "\n\n".join(pages)
        else:
            pages = [ocr_result]
            full_text = ocr_result
        
        # Save OCR text if output directory specified
        if output_dir:
            ocr_output_path = output_dir / f"{document_path.stem}_ocr.txt"
            with open(ocr_output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            logger.info(f"Saved OCR text to {ocr_output_path}")
        
        # Step 2: Text preprocessing
        logger.info("Preprocessing text")
        nlp_start_time = datetime.now()
        
        processed_text = self.text_preprocessor.process_text(
            full_text,
            clean=clean_text
        )
        
        layout_analysis = self.layout_analyzer.analyze_layout(full_text)
        
        nlp_time = (datetime.now() - nlp_start_time).total_seconds()
        logger.info(f"Text preprocessing completed in {nlp_time:.2f} seconds")
        
        # Step 3: Entity extraction
        logger.info("Extracting entities")
        entity_start_time = datetime.now()
        
        entities = self.entity_merger.merge_entities(processed_text['processed'])
        
        entity_time = (datetime.now() - entity_start_time).total_seconds()
        logger.info(f"Entity extraction completed in {entity_time:.2f} seconds")
        logger.info(f"Found {len(entities)} entities")
        
        # Step 4: Prepare results
        result = {
            'document': {
                'path': str(document_path),
                'name': document_path.name,
                'type': document_path.suffix.lower()[1:],
                'size': document_path.stat().st_size,
                'pages': len(pages)
            },
            'ocr': {
                'text': full_text,
                'processing_time': ocr_time
            },
            'text_analysis': {
                'word_count': processed_text['word_count'],
                'sentences': len(processed_text['sentences']),
                'paragraphs': len(processed_text['paragraphs']),
                'layout': layout_analysis,
                'processing_time': nlp_time
            },
            'entities': entities,
            'entity_extraction': {
                'count': len(entities),
                'processing_time': entity_time
            },
            'total_processing_time': ocr_time + nlp_time + entity_time
        }
        
        # Save results to JSON if output directory specified
        if output_dir:
            json_output_path = output_dir / f"{document_path.stem}_results.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results to {json_output_path}")
        
        return result
    
    def process_batch(self, document_paths: List[Union[str, Path]], 
                     output_dir: Union[str, Path] = None,
                     preprocess_ocr: bool = True,
                     clean_text: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            document_paths: List of paths to documents
            output_dir: Directory to save outputs
            preprocess_ocr: Whether to preprocess images before OCR
            clean_text: Whether to clean text after OCR
            
        Returns:
            List of processing results
        """
        results = []
        
        for doc_path in document_paths:
            result = self.process_document(
                doc_path,
                output_dir=output_dir,
                preprocess_ocr=preprocess_ocr,
                clean_text=clean_text
            )
            results.append(result)
        
        # Save batch summary if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            summary = {
                'batch_size': len(document_paths),
                'timestamp': datetime.now().isoformat(),
                'total_entities': sum(r.get('entity_extraction', {}).get('count', 0) for r in results),
                'total_processing_time': sum(r.get('total_processing_time', 0) for r in results),
                'documents': [r.get('document', {}).get('name', '') for r in results]
            }
            
            summary_path = output_dir / "batch_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved batch summary to {summary_path}")
        
        return results


class PipelineBuilder:
    """Builder for creating pipeline instances with custom configurations."""
    
    def __init__(self):
        """Initialize pipeline builder with default configuration."""
        self.config = {
            'ocr': {
                'engine': 'tesseract',
                'params': {
                    'lang': 'eng'
                },
                'dpi': 300
            },
            'nlp': {
                'language': 'en'
            },
            'entity_extraction': {
                'extractors': [
                    {
                        'type': 'spacy',
                        'params': {
                            'model_name': 'en_core_web_sm'
                        }
                    }
                ],
                'priority_order': None
            }
        }
    
    def with_ocr_engine(self, engine_type: str, **params) -> 'PipelineBuilder':
        """
        Set OCR engine.
        
        Args:
            engine_type: Type of OCR engine ('tesseract' or 'easyocr')
            **params: Additional parameters for the OCR engine
            
        Returns:
            Self for method chaining
        """
        self.config['ocr']['engine'] = engine_type
        self.config['ocr']['params'] = params
        return self
    
    def with_ocr_dpi(self, dpi: int) -> 'PipelineBuilder':
        """
        Set DPI for PDF conversion.
        
        Args:
            dpi: DPI value
            
        Returns:
            Self for method chaining
        """
        self.config['ocr']['dpi'] = dpi
        return self
    
    def with_language(self, language: str) -> 'PipelineBuilder':
        """
        Set language for NLP processing.
        
        Args:
            language: Language code
            
        Returns:
            Self for method chaining
        """
        self.config['nlp']['language'] = language
        return self
    
    def with_entity_extractor(self, extractor_type: str, **params) -> 'PipelineBuilder':
        """
        Add entity extractor.
        
        Args:
            extractor_type: Type of entity extractor
            **params: Additional parameters for the extractor
            
        Returns:
            Self for method chaining
        """
        extractor_config = {
            'type': extractor_type,
            'params': params
        }
        
        self.config['entity_extraction']['extractors'].append(extractor_config)
        return self
    
    def with_priority_order(self, priority_order: List[str]) -> 'PipelineBuilder':
        """
        Set priority order for entity merger.
        
        Args:
            priority_order: List of extractor source names in priority order
            
        Returns:
            Self for method chaining
        """
        self.config['entity_extraction']['priority_order'] = priority_order
        return self
    
    def build(self) -> Pipeline:
        """
        Build pipeline with current configuration.
        
        Returns:
            Configured Pipeline instance
        """
        return Pipeline(self.config)


# Example usage
if __name__ == "__main__":
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
    
    # Process a document
    sample_path = Path("../data/samples/sample.pdf")
    if sample_path.exists():
        result = pipeline.process_document(
            sample_path,
            output_dir="../outputs"
        )
        
        print(f"Processed document: {sample_path.name}")
        print(f"Found {result['entity_extraction']['count']} entities")
        print(f"Total processing time: {result['total_processing_time']:.2f} seconds")
    else:
        print(f"Sample file not found: {sample_path}")
