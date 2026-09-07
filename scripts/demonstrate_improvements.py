#!/usr/bin/env python3
"""
AntStack Comprehensive Improvements Demonstration

This script demonstrates all the major improvements made to the AntStack project,
showcasing the enhanced functionality, better documentation, and improved methods.

Run this script to see the comprehensive improvements in action:
    python scripts/demonstrate_improvements.py
"""

import sys

from antstack_core.orchestration.demo import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
