#!/usr/bin/env python3
"""
Update Paper Sections with Key Numbers Integration

Automatically replaces hardcoded numbers in paper sections with key_numbers.json
placeholders for dynamic content generation.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.key_numbers_updater import run


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Update paper sections with key numbers integration")
    parser.add_argument("--paper", default="complexity_energetics",
                       help="Paper name to update (default: complexity_energetics)")
    parser.add_argument("--section", help="Specific section to update")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing placeholders instead of updating")

    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    main()
