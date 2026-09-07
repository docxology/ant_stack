#!/usr/bin/env python3
"""Generate all tables and numerical content for the manuscript.

Ensures all tables, numbers, and statistical results are generated from
the actual analysis code rather than being manually entered.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_tables_and_numbers import main

if __name__ == "__main__":
    main()
