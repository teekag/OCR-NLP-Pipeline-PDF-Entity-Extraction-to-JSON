"""
Unit tests for the entity extractor module.
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_extractor import EntityExtractorFactory, CustomEntityExtractor


class TestEntityExtractor(unittest.TestCase):
    """Test cases for entity extractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        Apple Inc. is headquartered in Cupertino, California.
        Contact them at info@apple.com or call +1-800-275-2273.
        Their website is https://www.apple.com.
        Invoice #12345 dated 01/15/2023 for $1,299.99.
        """
        
        # Custom patterns for testing
        self.custom_patterns = {
            'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'PHONE': [r'\+?[\d\-\(\)\s]{10,20}'],
            'URL': [r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'],
            'INVOICE_NUMBER': [r'(?:Invoice|INV)\s*#?\s*(\d+)'],
            'AMOUNT': [r'\$\s*[\d,]+\.?\d*']
        }
    
    def test_custom_extractor(self):
        """Test the custom pattern-based entity extractor."""
        extractor = CustomEntityExtractor(patterns=self.custom_patterns)
        entities = extractor.extract_entities(self.sample_text)
        
        # Check that we found the expected entities
        self.assertGreater(len(entities), 0, "Should extract at least one entity")
        
        # Check for specific entity types
        entity_types = [entity['label'] for entity in entities]
        self.assertIn('EMAIL', entity_types, "Should extract email")
        self.assertIn('URL', entity_types, "Should extract URL")
        
        # Verify email content
        email_entities = [e for e in entities if e['label'] == 'EMAIL']
        self.assertEqual(email_entities[0]['text'], 'info@apple.com', "Should extract correct email")
    
    def test_factory_creation(self):
        """Test the entity extractor factory."""
        # Create custom extractor
        extractor = EntityExtractorFactory.create_extractor('custom', patterns=self.custom_patterns)
        self.assertIsInstance(extractor, CustomEntityExtractor, "Factory should create correct extractor type")
        
        # Verify supported entity types
        supported_types = extractor.get_supported_entity_types()
        for entity_type in self.custom_patterns.keys():
            self.assertIn(entity_type, supported_types, f"Should support {entity_type}")
    
    def test_entity_extraction(self):
        """Test entity extraction functionality."""
        extractor = EntityExtractorFactory.create_extractor('custom', patterns=self.custom_patterns)
        entities = extractor.extract_entities(self.sample_text)
        
        # Check invoice number extraction
        invoice_entities = [e for e in entities if e['label'] == 'INVOICE_NUMBER']
        self.assertTrue(invoice_entities, "Should extract invoice number")
        self.assertEqual(invoice_entities[0]['text'], '12345', "Should extract correct invoice number")
        
        # Check amount extraction
        amount_entities = [e for e in entities if e['label'] == 'AMOUNT']
        self.assertTrue(amount_entities, "Should extract amount")
        self.assertEqual(amount_entities[0]['text'], '$1,299.99', "Should extract correct amount")


if __name__ == '__main__':
    unittest.main()
