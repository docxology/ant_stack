"""AntStack Core: Reusable Scientific Publication Methods.

This package provides core functionality for generating high-quality scientific
publications with consistent formatting, cross-referencing, and validation.

The package follows test-driven development principles with real methods
(no mocks) and comprehensive validation as specified in .cursorrules.

Core Modules:
- figures: Figure management, cross-referencing, and visualization
- mathematics: LaTeX processing, symbol conversion, equation handling  
- publishing: PDF generation, templates, and quality validation
- analysis: Scientific analysis, energy estimation, scaling relationships

Design Principles:
- Modular, well-documented, clearly reasoned code
- Professional, functional, intelligent implementation
- Test-driven development with real data analysis
- Show, don't tell documentation approach
"""

__version__ = "1.0.0"
__author__ = "Daniel Ari Friedman"
__email__ = "daniel@activeinference.institute"

__all__ = [
    "figures",
    "mathematics", 
    "publishing",
    "analysis"
]

# Verify core dependencies are available
import sys
from pathlib import Path

def _check_dependencies():
    """Check that core dependencies are available."""
    missing = []
    
    try:
        import matplotlib
        import numpy
    except ImportError as e:
        missing.append(f"matplotlib/numpy: {e}")
    
    try:
        import yaml
    except ImportError as e:
        missing.append(f"yaml: {e}")
        
    if missing:
        print("Warning: Some optional dependencies missing:", missing)

_check_dependencies()
