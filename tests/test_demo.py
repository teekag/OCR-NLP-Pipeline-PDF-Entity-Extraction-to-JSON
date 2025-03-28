#!/usr/bin/env python3
"""
Unit tests for the demo script.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import demo module
import demo


class TestDemo(unittest.TestCase):
    """Test the demo script functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock sample data path
        self.sample_data_path = Path('data/sample_outputs')
        
    def test_list_sample_documents(self):
        """Test that the demo can list sample documents."""
        with patch('demo.list_sample_documents') as mock_list:
            # Mock return value
            mock_list.return_value = [
                {'id': 1, 'name': 'invoice_sample.pdf', 'type': 'Invoice'},
                {'id': 2, 'name': 'form_sample.pdf', 'type': 'Form'}
            ]
            
            # Call the function
            result = mock_list()
            
            # Verify results
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['type'], 'Invoice')
            self.assertEqual(result[1]['type'], 'Form')
    
    def test_load_sample_output(self):
        """Test loading sample output."""
        with patch('demo.load_sample_output') as mock_load:
            # Mock return value
            mock_data = {
                'document': {'filename': 'invoice_sample.pdf', 'pages': 1},
                'ocr': {'text': 'Sample OCR text'},
                'entities': [
                    {'text': 'John Doe', 'type': 'PERSON', 'confidence': 0.95},
                    {'text': '$100.00', 'type': 'MONEY', 'confidence': 0.90}
                ]
            }
            mock_load.return_value = mock_data
            
            # Call the function
            result = mock_load('invoice_sample')
            
            # Verify results
            self.assertEqual(result['document']['filename'], 'invoice_sample.pdf')
            self.assertEqual(len(result['entities']), 2)
    
    def test_format_entities(self):
        """Test entity formatting for display."""
        # Sample entities
        entities = [
            {'text': 'John Doe', 'type': 'PERSON', 'confidence': 0.95},
            {'text': '123 Main St', 'type': 'ADDRESS', 'confidence': 0.85},
            {'text': '01/01/2023', 'type': 'DATE', 'confidence': 0.90}
        ]
        
        with patch('demo.format_entities_for_display') as mock_format:
            # Mock return value
            mock_format.return_value = {
                'PERSON': ['John Doe (95%)'],
                'ADDRESS': ['123 Main St (85%)'],
                'DATE': ['01/01/2023 (90%)']
            }
            
            # Call the function
            result = mock_format(entities)
            
            # Verify results
            self.assertIn('PERSON', result)
            self.assertIn('ADDRESS', result)
            self.assertIn('DATE', result)
            self.assertEqual(len(result['PERSON']), 1)
            self.assertEqual(len(result['ADDRESS']), 1)
            self.assertEqual(len(result['DATE']), 1)


if __name__ == '__main__':
    unittest.main()
