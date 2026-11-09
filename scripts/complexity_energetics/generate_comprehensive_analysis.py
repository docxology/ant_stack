#!/usr/bin/env python3
"""Comprehensive analysis script for complexity and energetics research.

This script orchestrates the complete analysis pipeline including:
- Enhanced analytical workload calculations on fixed parameter grids
- Detailed modeled energy breakdown and operation-count efficiency proxies
- Descriptive scaling relationship and power-law analysis
- Conditional one-erased-bit-per-FLOP references (not physical efficiency)
- Publication-quality figure generation with comprehensive captions

Usage:
    python scripts/complexity_energetics/generate_comprehensive_analysis.py --output fresh_dir [--manifest path/to/manifest.yaml]

References:
- Scientific computing workflows: https://doi.org/10.1371/journal.pcbi.1004668
- Reproducible research practices: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_module_scaling_analysis import run


def main():
    """Main analysis orchestration function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive complexity and energetics analysis")
    parser.add_argument("--manifest", default="papers/complexity_energetics/manifest.example.yaml",
                       help="Path to experiment manifest (coefficient source, not sweep grid)")
    parser.add_argument("--output", required=True,
                       help="Explicit absent or empty output family directory")
    parser.add_argument("--modules", nargs="+", choices=["body", "brain", "mind"], default=["body", "brain", "mind"],
                       help="Modules to analyze")
    
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    main()
