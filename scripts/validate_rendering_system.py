#!/usr/bin/env python3
"""Comprehensive validation of the PDF rendering system.

This script validates all aspects of the rendering system:
- Paper configuration validation
- Cross-reference consistency
- Figure format compliance
- Math symbol formatting
- Hyperlink validation
- Build system integration

Usage:
    python3 scripts/validate_rendering_system.py [--paper PAPER_NAME] [--verbose]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.publishing.rendering_validator import run


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate PDF rendering system")
    parser.add_argument("--paper", help="Validate specific paper only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
