"""
NLP Parser Module
----------------
Provides text preprocessing and cleaning capabilities for OCR output.
Handles noise removal, text normalization, and sentence segmentation.
"""

import re
import string
import logging
from typing import List, Dict, Any, Optional, Union

import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Handles text preprocessing for OCR output."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize text preprocessor.
        
        Args:
            language: Language code for text processing
        """
        self.language = language
        self.stop_words = set(stopwords.words('english') if language == 'en' else [])
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(f"{language}_core_web_sm")
            logger.info(f"Loaded spaCy model: {language}_core_web_sm")
        except OSError:
            logger.warning(f"spaCy model {language}_core_web_sm not found. Using blank model.")
            self.nlp = spacy.blank(language)
    
    def clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR-generated text by removing noise and artifacts.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single one
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c == '\n')
        
        # Fix common OCR errors
        text = self._fix_common_ocr_errors(text)
        
        return text.strip()
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors.
        
        Args:
            text: OCR text
            
        Returns:
            Text with common OCR errors fixed
        """
        # Fix broken words (words split by newlines)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix common character confusions
        replacements = {
            'l': '1',  # lowercase l to 1 in numeric contexts
            'O': '0',  # uppercase O to 0 in numeric contexts
            'I': '1',  # uppercase I to 1 in numeric contexts
        }
        
        # Only replace in numeric contexts
        for char, replacement in replacements.items():
            text = re.sub(f'(?<=[0-9]){char}(?=[0-9])', replacement, text)
            text = re.sub(f'(?<=[0-9]){char}$', replacement, text)
            text = re.sub(f'^{char}(?=[0-9])', replacement, text)
        
        return text
    
    def normalize_text(self, text: str, remove_stopwords: bool = False, 
                       remove_punctuation: bool = False) -> str:
        """
        Normalize text by converting to lowercase and optionally removing stopwords.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation if requested
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove stopwords if requested
        if remove_stopwords and self.stop_words:
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_ocr_text(text)
        
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        return sentences
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_ocr_text(text)
        
        # Split by double newlines or multiple newlines
        paragraphs = re.split(r'\n\n+', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def process_text(self, text: str, clean: bool = True, normalize: bool = True,
                    remove_stopwords: bool = False, remove_punctuation: bool = False) -> Dict[str, Any]:
        """
        Process text with multiple preprocessing steps.
        
        Args:
            text: Input text
            clean: Whether to clean the text
            normalize: Whether to normalize the text
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            
        Returns:
            Dictionary with processed text and metadata
        """
        if not text:
            return {
                'original': '',
                'processed': '',
                'sentences': [],
                'paragraphs': [],
                'word_count': 0
            }
        
        # Store original text
        original = text
        
        # Clean text if requested
        if clean:
            text = self.clean_ocr_text(text)
        
        # Normalize text if requested
        if normalize:
            processed_text = self.normalize_text(
                text, 
                remove_stopwords=remove_stopwords,
                remove_punctuation=remove_punctuation
            )
        else:
            processed_text = text
        
        # Extract sentences and paragraphs
        sentences = self.segment_sentences(text)
        paragraphs = self.extract_paragraphs(text)
        
        # Count words
        word_count = len(word_tokenize(text)) if text else 0
        
        return {
            'original': original,
            'processed': processed_text,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'word_count': word_count
        }


class LayoutAnalyzer:
    """Analyzes document layout and structure."""
    
    def __init__(self):
        """Initialize layout analyzer."""
        pass
    
    def detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect potential tables in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected tables with their positions
        """
        # This is a simple heuristic approach - in a real system, 
        # you would use more sophisticated table detection
        tables = []
        
        # Look for patterns of aligned text that might indicate tables
        lines = text.split('\n')
        
        # Find consecutive lines with similar structure
        table_start = -1
        for i, line in enumerate(lines):
            # Check if line has multiple whitespace-separated segments
            segments = [s for s in line.split() if s]
            
            # If line has multiple segments with consistent spacing, it might be a table row
            if len(segments) >= 3 and re.search(r'\s{2,}', line):
                if table_start == -1:
                    table_start = i
            elif table_start != -1:
                # End of potential table
                if i - table_start >= 2:  # At least 3 rows to be considered a table
                    tables.append({
                        'start_line': table_start,
                        'end_line': i - 1,
                        'content': '\n'.join(lines[table_start:i])
                    })
                table_start = -1
        
        # Check if table extends to the end of the text
        if table_start != -1 and len(lines) - table_start >= 3:
            tables.append({
                'start_line': table_start,
                'end_line': len(lines) - 1,
                'content': '\n'.join(lines[table_start:])
            })
        
        return tables
    
    def detect_lists(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect bulleted or numbered lists in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected lists with their positions
        """
        lists = []
        
        # Look for patterns of list items
        lines = text.split('\n')
        
        # Regular expressions for common list markers
        bullet_pattern = r'^\s*[\•\-\*\+\◦\○\●]\s+'
        number_pattern = r'^\s*\d+[\.\)]\s+'
        letter_pattern = r'^\s*[a-zA-Z][\.\)]\s+'
        
        list_start = -1
        list_type = None
        
        for i, line in enumerate(lines):
            # Check if line starts with a list marker
            if re.match(bullet_pattern, line):
                if list_start == -1 or list_type != 'bullet':
                    if list_start != -1:
                        # End previous list
                        lists.append({
                            'type': list_type,
                            'start_line': list_start,
                            'end_line': i - 1,
                            'content': '\n'.join(lines[list_start:i])
                        })
                    list_start = i
                    list_type = 'bullet'
            elif re.match(number_pattern, line):
                if list_start == -1 or list_type != 'numbered':
                    if list_start != -1:
                        # End previous list
                        lists.append({
                            'type': list_type,
                            'start_line': list_start,
                            'end_line': i - 1,
                            'content': '\n'.join(lines[list_start:i])
                        })
                    list_start = i
                    list_type = 'numbered'
            elif re.match(letter_pattern, line):
                if list_start == -1 or list_type != 'lettered':
                    if list_start != -1:
                        # End previous list
                        lists.append({
                            'type': list_type,
                            'start_line': list_start,
                            'end_line': i - 1,
                            'content': '\n'.join(lines[list_start:i])
                        })
                    list_start = i
                    list_type = 'lettered'
            elif list_start != -1 and line.strip() == '':
                # Empty line ends the list
                lists.append({
                    'type': list_type,
                    'start_line': list_start,
                    'end_line': i - 1,
                    'content': '\n'.join(lines[list_start:i])
                })
                list_start = -1
                list_type = None
        
        # Check if list extends to the end of the text
        if list_start != -1:
            lists.append({
                'type': list_type,
                'start_line': list_start,
                'end_line': len(lines) - 1,
                'content': '\n'.join(lines[list_start:])
            })
        
        return lists
    
    def analyze_layout(self, text: str) -> Dict[str, Any]:
        """
        Analyze document layout.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with layout analysis results
        """
        if not text:
            return {
                'tables': [],
                'lists': [],
                'structure': 'unknown'
            }
        
        # Detect tables and lists
        tables = self.detect_tables(text)
        lists = self.detect_lists(text)
        
        # Determine overall document structure
        structure = 'plain_text'
        if tables:
            structure = 'tabular'
        elif lists:
            structure = 'list_based'
        
        return {
            'tables': tables,
            'lists': lists,
            'structure': structure
        }


# Example usage
if __name__ == "__main__":
    # Example OCR text with noise
    sample_text = """
    This is a sample OCR text with some n0ise.
    It has multiple lines and some
    common OCR errors like l instead of 1.
    
    Here's a list:
    • Item 1
    • Item 2
    • Item 3
    
    And a simple table:
    Name    Age     City
    John    30      New York
    Jane    25      Boston
    """
    
    # Process the text
    preprocessor = TextPreprocessor()
    result = preprocessor.process_text(sample_text)
    
    print(f"Word count: {result['word_count']}")
    print(f"Number of sentences: {len(result['sentences'])}")
    print(f"Number of paragraphs: {len(result['paragraphs'])}")
    
    # Analyze layout
    layout_analyzer = LayoutAnalyzer()
    layout = layout_analyzer.analyze_layout(sample_text)
    
    print(f"Document structure: {layout['structure']}")
    print(f"Tables detected: {len(layout['tables'])}")
    print(f"Lists detected: {len(layout['lists'])}")
