#!/usr/bin/env python3
"""Unified build and validation system for configured Ant Stack papers.

This legacy-compatible wrapper builds and validates the paper directories that
exist under ``papers/``.

Features:
- Pre-build validation and dependency checks
- Integrated figure generation and cross-reference validation
- Post-build quality assurance with detailed reporting
- Consistent formatting and documentation standards
- Comprehensive test integration
- Streamlined workflow for both development and production builds
"""

import sys
import argparse
from pathlib import Path

from antstack_core.orchestration.unified_build import run


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Build System for Ant Stack Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
	  uv run python scripts/common_pipeline/unified_build.py                    # Build all papers
	  uv run python scripts/common_pipeline/unified_build.py --paper ant_stack  # Build specific paper
	  uv run python scripts/common_pipeline/unified_build.py --no-tests         # Skip tests
	  uv run python scripts/common_pipeline/unified_build.py --validate-only    # Only validate, don't build
        """
    )
    
    parser.add_argument(
        "--paper", 
        choices=["ant_stack", "complexity_energetics", "documentation"],
        help="Build specific paper only"
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true", 
        help="Skip test execution"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't build"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
