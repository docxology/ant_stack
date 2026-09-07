#!/usr/bin/env python3
"""Generate multipanel figures for the manuscript.

Creates sophisticated multipanel figures with proper subfigure layouts,
statistical overlays, and publication-quality formatting.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_multipanel_figures import main

if __name__ == "__main__":
    main()
