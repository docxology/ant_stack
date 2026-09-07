#!/usr/bin/env python3
"""
Unified Configuration Validation Script

This script validates that all components of the Ant Stack complexity and energetics
analysis use the same configuration values, ensuring coherent maximal usage of
unified paper-level configuration.

Validates:
- Energy coefficients consistency across all sources
- Parameter values alignment between paper config and manifest
- Test configuration consistency
- Analysis pipeline configuration usage
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.config_validation import main


if __name__ == "__main__":
    main()
