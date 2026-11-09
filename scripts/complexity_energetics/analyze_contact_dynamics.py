#!/usr/bin/env python3
"""Contact dynamics analysis orchestrator.

Thin orchestrator for contact complexity analysis using ce.workloads methods.
Analyzes different solver algorithms and contact configurations.

Usage:
    python scripts/complexity_energetics/analyze_contact_dynamics.py --output NEW_DIRECTORY
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
    parser.add_argument("--output", required=True,
                       help="Explicit absent or empty output family directory")
    parser.add_argument("--max-contacts", type=int, default=50,
                       help="Maximum number of contacts to analyze")

    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
