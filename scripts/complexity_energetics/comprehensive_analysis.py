#!/usr/bin/env python3
"""Comprehensive complexity energetics analysis with advanced methodologies.

This script orchestrates a complete analysis pipeline incorporating:
- Advanced complexity analysis with agent-based modeling
- Network complexity metrics and structural analysis
- Thermodynamic efficiency frameworks
- Complexity-entropy diagrams for intrinsic computation
- Multi-scale visualization and statistical validation
- Veridical reporting with complete empirical evidence

Usage:
    python scripts/complexity_energetics/comprehensive_analysis.py [--manifest path/to/manifest.yaml] [--output output_dir]

References:
- Agent-based modeling: https://eprints.whiterose.ac.uk/81723/
- Complexity-entropy analysis: https://arxiv.org/abs/0806.4789
- Thermodynamic computing: https://pubmed.ncbi.nlm.nih.gov/28505845/
- Reproducible research: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_comprehensive_analysis import run


def main():
    """Main analysis orchestration function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive complexity energetics analysis")
    parser.add_argument("--manifest", default="complexity_energetics/manifest.example.yaml",
                       help="Path to experiment manifest")
    parser.add_argument("--output", default="comprehensive_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--modules", nargs="+", default=["body", "brain", "mind"],
                       help="Modules to analyze")
    
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    main()
