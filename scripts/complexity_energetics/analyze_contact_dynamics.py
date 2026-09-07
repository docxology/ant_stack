#!/usr/bin/env python3
"""Contact dynamics analysis orchestrator.

Thin orchestrator for contact complexity analysis using ce.workloads methods.
Analyzes different solver algorithms and contact configurations.

Usage:
    python scripts/analyze_contact_dynamics.py [--output output_dir] [--solver pgs|lcp|mlcp]
"""

import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_contact_dynamics import run


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Analyze contact dynamics and terrain effects")
    parser.add_argument("--output", default="contact_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--solver", choices=["pgs", "lcp", "mlcp"], default="pgs",
                       help="Primary solver to analyze")
    parser.add_argument("--max-contacts", type=int, default=50,
                       help="Maximum number of contacts to analyze")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
