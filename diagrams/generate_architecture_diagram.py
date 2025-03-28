#!/usr/bin/env python3
"""
Generate a simple architecture diagram for the OCR-NLP Pipeline.
Requires graphviz and pydot packages.
"""

import os
import pydot

def generate_architecture_diagram():
    """Generate a simple architecture diagram for the OCR-NLP Pipeline."""
    # Create a new graph
    graph = pydot.Dot("ocr_nlp_pipeline", graph_type="digraph", rankdir="LR")
    
    # Define node styles
    graph.set_node_defaults(
        shape="box",
        style="filled",
        fillcolor="lightblue",
        fontname="Arial",
        fontsize="12"
    )
    
    # Add nodes
    input_node = pydot.Node("Input", label="PDF/Image\nInput", shape="folder", fillcolor="#FFD700")
    ocr_node = pydot.Node("OCR", label="OCR Engine\n(Tesseract/EasyOCR)")
    preproc_node = pydot.Node("Preprocessing", label="Text\nPreprocessing")
    layout_node = pydot.Node("Layout", label="Layout\nAnalysis")
    entity_node = pydot.Node("Entity", label="Entity\nExtraction")
    output_node = pydot.Node("Output", label="JSON\nOutput", shape="note", fillcolor="#90EE90")
    
    # Add nodes to graph
    graph.add_node(input_node)
    graph.add_node(ocr_node)
    graph.add_node(preproc_node)
    graph.add_node(layout_node)
    graph.add_node(entity_node)
    graph.add_node(output_node)
    
    # Add edges
    graph.add_edge(pydot.Edge(input_node, ocr_node))
    graph.add_edge(pydot.Edge(ocr_node, preproc_node))
    graph.add_edge(pydot.Edge(preproc_node, layout_node))
    graph.add_edge(pydot.Edge(layout_node, entity_node))
    graph.add_edge(pydot.Edge(entity_node, output_node))
    
    # Add subgraph for entity extractors
    extractors = pydot.Cluster("extractors", label="Entity Extractors", style="dashed", color="gray")
    spacy_node = pydot.Node("SpaCy", label="SpaCy", shape="ellipse", fillcolor="#FFC0CB")
    flair_node = pydot.Node("Flair", label="Flair", shape="ellipse", fillcolor="#FFC0CB")
    custom_node = pydot.Node("Custom", label="Custom Rules", shape="ellipse", fillcolor="#FFC0CB")
    
    extractors.add_node(spacy_node)
    extractors.add_node(flair_node)
    extractors.add_node(custom_node)
    
    graph.add_subgraph(extractors)
    
    graph.add_edge(pydot.Edge(entity_node, spacy_node, style="dashed", dir="both"))
    graph.add_edge(pydot.Edge(entity_node, flair_node, style="dashed", dir="both"))
    graph.add_edge(pydot.Edge(entity_node, custom_node, style="dashed", dir="both"))
    
    # Save the graph
    output_path = os.path.join(os.path.dirname(__file__), "pipeline_architecture.png")
    graph.write_png(output_path)
    print(f"Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    try:
        generate_architecture_diagram()
    except ImportError:
        print("Error: This script requires the pydot and graphviz packages.")
        print("Please install them with: pip install pydot graphviz")
        print("And ensure that Graphviz is installed on your system.")
