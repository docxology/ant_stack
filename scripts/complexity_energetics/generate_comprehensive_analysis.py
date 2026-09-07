#!/usr/bin/env python3
"""Comprehensive analysis script for complexity and energetics research.

This script orchestrates the complete analysis pipeline including:
- Enhanced workload calculations with realistic complexity models
- Detailed energy breakdown and efficiency analysis
- Scaling relationship identification and power law analysis
- Theoretical limit comparisons and efficiency benchmarking
- Publication-quality figure generation with comprehensive captions

Usage:
    python scripts/generate_comprehensive_analysis.py [--manifest path/to/manifest.yaml] [--output output_dir]

References:
- Scientific computing workflows: https://doi.org/10.1371/journal.pcbi.1004668
- Reproducible research practices: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_module_scaling_analysis import run


def main():
    """Main analysis orchestration function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive complexity and energetics analysis")
    parser.add_argument("--manifest", default="complexity_energetics/manifest.example.yaml",
                       help="Path to experiment manifest")
    parser.add_argument("--output", default="analysis_output",
                       help="Output directory for results")
    parser.add_argument("--modules", nargs="+", default=["body", "brain", "mind"],
                       help="Modules to analyze")
    
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    main()
