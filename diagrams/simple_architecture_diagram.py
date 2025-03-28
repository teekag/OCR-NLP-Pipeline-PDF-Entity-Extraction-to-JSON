#!/usr/bin/env python3
"""
Generate a simple architecture diagram for the OCR-NLP Pipeline using Pillow.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_architecture_diagram():
    """Create a simple architecture diagram using Pillow."""
    # Create a new image with white background
    width, height = 1000, 500
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font_large = ImageFont.truetype("Arial", 20)
        font_small = ImageFont.truetype("Arial", 16)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define colors
    box_color = (173, 216, 230)  # light blue
    arrow_color = (0, 0, 0)      # black
    text_color = (0, 0, 0)       # black
    
    # Define box positions and sizes
    box_width = 150
    box_height = 80
    margin = 30
    start_x = 50
    center_y = height // 2
    
    # Define the pipeline stages
    stages = [
        "PDF/Image\nInput",
        "OCR Engine\n(Tesseract/EasyOCR)",
        "Text\nPreprocessing",
        "Layout\nAnalysis",
        "Entity\nExtraction",
        "JSON\nOutput"
    ]
    
    # Draw boxes and labels
    x_positions = []
    for i, stage in enumerate(stages):
        x = start_x + i * (box_width + margin)
        y = center_y - box_height // 2
        
        # Draw box
        draw.rectangle([(x, y), (x + box_width, y + box_height)], 
                      outline=arrow_color, fill=box_color, width=2)
        
        # Draw text
        text_x = x + box_width // 2
        text_y = y + box_height // 2
        draw.text((text_x, text_y), stage, fill=text_color, font=font_small, anchor="mm")
        
        x_positions.append(x)
    
    # Draw arrows between boxes
    for i in range(len(stages) - 1):
        start_x = x_positions[i] + box_width
        end_x = x_positions[i + 1]
        y = center_y
        
        # Draw line
        draw.line([(start_x, y), (end_x, y)], fill=arrow_color, width=2)
        
        # Draw arrowhead
        draw.polygon([(end_x - 10, y - 5), (end_x, y), (end_x - 10, y + 5)], 
                    fill=arrow_color)
    
    # Add title
    title = "OCR-NLP Pipeline Architecture"
    draw.text((width // 2, 30), title, fill=text_color, font=font_large, anchor="mm")
    
    # Add entity extractors
    extractor_y = center_y + box_height + 30
    draw.text((x_positions[4] + box_width // 2, extractor_y), 
             "Entity Extractors:", fill=text_color, font=font_small, anchor="mm")
    
    extractors = ["SpaCy", "Flair", "Custom Rules"]
    for i, extractor in enumerate(extractors):
        ex_x = x_positions[4] + (i - 1) * (box_width // 2 + 10) + box_width // 2
        ex_y = extractor_y + 30
        
        # Draw small box
        small_width = box_width // 2
        small_height = 40
        draw.rectangle([(ex_x - small_width // 2, ex_y), 
                       (ex_x + small_width // 2, ex_y + small_height)], 
                      outline=arrow_color, fill=(255, 192, 203), width=2)  # light pink
        
        # Draw text
        draw.text((ex_x, ex_y + small_height // 2), extractor, 
                 fill=text_color, font=font_small, anchor="mm")
        
        # Draw dashed line to entity extraction
        for j in range(0, 60, 5):  # Draw dashed line
            draw.line([(ex_x, ex_y - j), (ex_x, ex_y - j - 3)], 
                     fill=arrow_color, width=1)
    
    # Save the image
    output_path = os.path.join(os.path.dirname(__file__), "pipeline_architecture.png")
    image.save(output_path)
    print(f"Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    try:
        create_architecture_diagram()
    except Exception as e:
        print(f"Error creating diagram: {e}")
        print("Please install Pillow with: pip install pillow")
