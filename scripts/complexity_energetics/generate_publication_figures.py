#!/usr/bin/env python3
"""Publication figure generation with comprehensive captions and auto-numbering.

Generates all figures for the complexity_energetics manuscript using publication-quality
plotting functions with comprehensive captions, statistical analysis, and
professional styling.

Features:
- Comprehensive captions with statistical details
- Auto-numbering system for figures
- Professional visualizations with statistical analysis
- Statistical analysis overlays
- Accessibility features
- LaTeX-compatible math rendering

Usage:
    python scripts/generate_enhanced_figures.py [--output assets_dir] [--format png]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_publication_figures import run


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced manuscript figures")
    parser.add_argument("--output", default="papers/complexity_energetics/assets",
                       help="Output directory for figures")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png",
                       help="Output format for figures")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
