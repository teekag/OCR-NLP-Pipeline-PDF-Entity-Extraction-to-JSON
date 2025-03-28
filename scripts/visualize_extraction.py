#!/usr/bin/env python3
"""
Visualize extracted entities on a PDF document.
This script creates a visual representation of the extracted entities by
highlighting them on the original PDF.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define colors for different entity types
ENTITY_COLORS = {
    'PERSON': (255, 0, 0, 64),       # Red
    'ORG': (0, 255, 0, 64),          # Green
    'DATE': (0, 0, 255, 64),         # Blue
    'MONEY': (255, 255, 0, 64),      # Yellow
    'ADDRESS': (255, 0, 255, 64),    # Magenta
    'INVOICE_NUM': (0, 255, 255, 64),# Cyan
    'PRODUCT': (255, 128, 0, 64),    # Orange
    'QUANTITY': (128, 0, 255, 64),   # Purple
    'TOTAL': (255, 0, 128, 64),      # Pink
    'DEFAULT': (200, 200, 200, 64)   # Gray (for other entity types)
}

def get_entity_color(entity_type):
    """Get color for a specific entity type."""
    return ENTITY_COLORS.get(entity_type, ENTITY_COLORS['DEFAULT'])

def visualize_entities(pdf_path, json_path, output_path=None):
    """
    Create a visual representation of extracted entities on the PDF.
    
    Args:
        pdf_path: Path to the original PDF
        json_path: Path to the JSON file with extracted entities
        output_path: Path to save the visualized PDF (default: adds '_visualized' suffix)
    
    Returns:
        Path to the visualized PDF
    """
    # Set default output path if not provided
    if output_path is None:
        pdf_path = Path(pdf_path)
        output_path = pdf_path.parent / f"{pdf_path.stem}_visualized.pdf"
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Process each entity
    for entity in data.get('entities', []):
        # Skip entities without position information
        if 'position' not in entity:
            continue
        
        position = entity['position']
        page_num = position.get('page', 1) - 1  # Convert to 0-based index
        bbox = position.get('bbox')
        
        # Skip if no bounding box or invalid page
        if not bbox or page_num >= len(pdf_document) or page_num < 0:
            continue
        
        # Get the page
        page = pdf_document[page_num]
        
        # Get color for this entity type
        color = get_entity_color(entity['type'])
        
        # Convert RGBA to RGB with alpha for PyMuPDF
        r, g, b, a = color
        opacity = a / 255.0
        
        # Draw rectangle for the entity
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        page.draw_rect(rect, color=(r/255, g/255, b/255), fill=(r/255, g/255, b/255), fill_opacity=opacity)
        
        # Add entity type annotation
        page.insert_text(
            point=(bbox[0], bbox[1] - 5),
            text=entity['type'],
            fontsize=8,
            color=(0, 0, 0)
        )
    
    # Save the visualized PDF
    pdf_document.save(output_path)
    pdf_document.close()
    
    print(f"Visualized PDF saved to: {output_path}")
    return output_path

def create_entity_legend(output_path):
    """Create a legend image showing entity type colors."""
    # Create a new image with white background
    width, height = 400, 400
    image = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 20), "Entity Type Legend", fill=(0, 0, 0, 255), font=font)
    
    # Draw legend entries
    y_pos = 60
    for entity_type, color in ENTITY_COLORS.items():
        if entity_type == 'DEFAULT':
            continue
            
        # Draw color box
        r, g, b, a = color
        draw.rectangle([(20, y_pos), (50, y_pos + 20)], fill=color, outline=(0, 0, 0, 255))
        
        # Draw entity type text
        draw.text((60, y_pos), entity_type, fill=(0, 0, 0, 255), font=font)
        
        y_pos += 30
    
    # Save the legend
    image.save(output_path)
    print(f"Entity legend saved to: {output_path}")
    return output_path

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize extracted entities on a PDF document.')
    parser.add_argument('pdf_path', help='Path to the original PDF')
    parser.add_argument('json_path', help='Path to the JSON file with extracted entities')
    parser.add_argument('--output', '-o', help='Path to save the visualized PDF')
    
    args = parser.parse_args()
    
    try:
        # Visualize entities on the PDF
        output_pdf = visualize_entities(args.pdf_path, args.json_path, args.output)
        
        # Create a legend
        legend_path = str(Path(output_pdf).parent / f"{Path(output_pdf).stem}_legend.png")
        create_entity_legend(legend_path)
        
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error visualizing entities: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
