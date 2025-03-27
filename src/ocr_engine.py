"""
OCR Engine Module
----------------
Provides OCR capabilities for extracting text from PDFs and images.
Supports multiple OCR backends (Tesseract, EasyOCR) with a unified interface.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import pytesseract
import easyocr
import pdf2image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from an image."""
        pass
    
    @abstractmethod
    def extract_text_with_regions(self, image: np.ndarray) -> List[Dict]:
        """Extract text with bounding box regions."""
        pass


class TesseractOCR(OCREngine):
    """OCR implementation using Tesseract."""
    
    def __init__(self, lang: str = 'eng', config: str = '--psm 3'):
        """
        Initialize Tesseract OCR engine.
        
        Args:
            lang: Language code for OCR (default: 'eng')
            config: Tesseract configuration string
        """
        self.lang = lang
        self.config = config
        logger.info(f"Initialized TesseractOCR with lang={lang}, config={config}")
        
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using Tesseract.
        
        Args:
            image: NumPy array containing the image
            
        Returns:
            Extracted text as string
        """
        try:
            text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
            return text
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {str(e)}")
            return ""
    
    def extract_text_with_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text with bounding box regions.
        
        Args:
            image: NumPy array containing the image
            
        Returns:
            List of dictionaries with text and coordinates
        """
        try:
            data = pytesseract.image_to_data(image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT)
            results = []
            
            for i in range(len(data['text'])):
                if data['conf'][i] > 0:  # Filter out low confidence results
                    result = {
                        'text': data['text'][i],
                        'conf': data['conf'][i],
                        'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error in Tesseract OCR region extraction: {str(e)}")
            return []


class EasyOCR(OCREngine):
    """OCR implementation using EasyOCR."""
    
    def __init__(self, lang_list: List[str] = ['en']):
        """
        Initialize EasyOCR engine.
        
        Args:
            lang_list: List of language codes for OCR
        """
        self.lang_list = lang_list
        self.reader = easyocr.Reader(lang_list)
        logger.info(f"Initialized EasyOCR with languages: {lang_list}")
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using EasyOCR.
        
        Args:
            image: NumPy array containing the image
            
        Returns:
            Extracted text as string
        """
        try:
            results = self.reader.readtext(image)
            return ' '.join([result[1] for result in results])
        except Exception as e:
            logger.error(f"Error in EasyOCR: {str(e)}")
            return ""
    
    def extract_text_with_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text with bounding box regions.
        
        Args:
            image: NumPy array containing the image
            
        Returns:
            List of dictionaries with text and coordinates
        """
        try:
            results = self.reader.readtext(image)
            return [
                {
                    'text': result[1],
                    'conf': result[2],
                    'bbox': result[0]  # EasyOCR returns points, not x,y,w,h
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error in EasyOCR region extraction: {str(e)}")
            return []


class DocumentProcessor:
    """Processes documents (PDFs or images) for OCR."""
    
    def __init__(self, ocr_engine: OCREngine, dpi: int = 300):
        """
        Initialize document processor.
        
        Args:
            ocr_engine: OCR engine implementation
            dpi: DPI for PDF conversion (higher = better quality but slower)
        """
        self.ocr_engine = ocr_engine
        self.dpi = dpi
        logger.info(f"Initialized DocumentProcessor with {ocr_engine.__class__.__name__}, dpi={dpi}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def process_image(self, image_path: Union[str, Path], preprocess: bool = True) -> str:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply preprocessing
            
        Returns:
            Extracted text
        """
        try:
            # Load image
            image_path = Path(image_path)
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return ""
            
            # Preprocess if requested
            if preprocess:
                image = self._preprocess_image(image)
            
            # Extract text
            text = self.ocr_engine.extract_text(image)
            return text
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return ""
    
    def process_pdf(self, pdf_path: Union[str, Path], preprocess: bool = True) -> List[str]:
        """
        Process a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of extracted text, one per page
        """
        try:
            pdf_path = Path(pdf_path)
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
            
            results = []
            for i, img in enumerate(images):
                # Convert PIL Image to OpenCV format
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Preprocess if requested
                if preprocess:
                    open_cv_image = self._preprocess_image(open_cv_image)
                
                # Extract text
                text = self.ocr_engine.extract_text(open_cv_image)
                results.append(text)
                
                logger.info(f"Processed page {i+1}/{len(images)} of {pdf_path.name}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def process_document(self, doc_path: Union[str, Path], preprocess: bool = True) -> Union[str, List[str]]:
        """
        Process a document (auto-detects if it's a PDF or image).
        
        Args:
            doc_path: Path to the document
            preprocess: Whether to apply preprocessing
            
        Returns:
            Extracted text (string for images, list of strings for PDFs)
        """
        doc_path = Path(doc_path)
        
        if doc_path.suffix.lower() == '.pdf':
            return self.process_pdf(doc_path, preprocess)
        else:
            return self.process_image(doc_path, preprocess)


def get_ocr_engine(engine_type: str = 'tesseract', **kwargs) -> OCREngine:
    """
    Factory function to get the appropriate OCR engine.
    
    Args:
        engine_type: Type of OCR engine ('tesseract' or 'easyocr')
        **kwargs: Additional arguments for the specific engine
        
    Returns:
        OCR engine instance
    """
    if engine_type.lower() == 'tesseract':
        return TesseractOCR(**kwargs)
    elif engine_type.lower() == 'easyocr':
        return EasyOCR(**kwargs)
    else:
        raise ValueError(f"Unsupported OCR engine type: {engine_type}")


# Example usage
if __name__ == "__main__":
    # Example: Process a PDF with Tesseract
    ocr = get_ocr_engine('tesseract', lang='eng')
    processor = DocumentProcessor(ocr)
    
    # Replace with an actual PDF path for testing
    sample_path = Path("../data/samples/sample.pdf")
    if sample_path.exists():
        results = processor.process_document(sample_path)
        print(f"Extracted {len(results)} pages of text")
        for i, text in enumerate(results):
            print(f"Page {i+1} preview: {text[:100]}...")
    else:
        print(f"Sample file not found: {sample_path}")
