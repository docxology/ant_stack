#!/usr/bin/env python3
"""Comprehensive formatting fix for the manuscript.

Fixes:
1. Naked URLs and file paths
2. Broken citations and references
3. Plaintext variables that should be LaTeX formatted
4. Ensures all citations are proper (Name, Year) format with hyperlinks
"""

import sys
from antstack_core.publishing.formatting_fixes import main

if __name__ == '__main__':
    main()
