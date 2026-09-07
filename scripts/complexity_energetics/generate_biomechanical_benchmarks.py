#!/usr/bin/env python3
"""Generate biomechanical and robotic benchmarks for energy analysis.

This script calculates comprehensive biomechanical benchmarks including:
- Power density comparisons between biological and robotic systems
- Efficiency trade-offs and energy storage capacity
- Detailed hexapod platform energy analysis
- Automatic markdown generation with calculated values

Usage:
    python scripts/generate_biomechanical_benchmarks.py [--output-dir output_dir] [--update-markdown]

References:
- Biomechanical power density: https://doi.org/10.1126/science.273.5272.267
- Robotic actuator efficiency: https://doi.org/10.1109/TMECH.2019.2942671
- Energy storage comparisons: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_biomechanical_benchmarks import run


def main():
    """Main function for generating biomechanical benchmarks."""
    parser = argparse.ArgumentParser(description="Generate biomechanical and robotic benchmarks")
    parser.add_argument("--output-dir", default="biomechanical_output",
                       help="Output directory for generated files")
    parser.add_argument("--update-markdown", action="store_true",
                       help="Update markdown files with generated content")
    parser.add_argument("--paper-dir",
                       default=os.path.join(os.path.dirname(__file__), '..', '..', 'papers', 'complexity_energetics'),
                       help="Path to complexity_energetics paper directory")
    
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    main()
