#!/usr/bin/env python3
"""
Generate API documentation for the OCR-NLP Pipeline using pdoc.
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_documentation():
    """Generate API documentation using pdoc."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Create docs directory if it doesn't exist
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Generate documentation
    cmd = [
        "pdoc", 
        "--html", 
        "--output-dir", str(docs_dir),
        "--force",
        str(project_root / "src"),
        str(project_root / "demo.py"),
        str(project_root / "run_pipeline.py")
    ]
    
    print(f"Generating documentation with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Documentation generated successfully in {docs_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_documentation()
