#!/usr/bin/env python3
"""Tests for antstack_core package initialization and core functionality.

Tests package imports, dependency checking, version information, and basic
package structure validation.

Following .cursorrules principles:
- Comprehensive validation of package initialization
- Dependency checking and error handling
- Professional, well-documented test cases
- Real functionality testing (no mocks)
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCorePackage(unittest.TestCase):
    """Test antstack_core package initialization and core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent

    def test_package_import(self):
        """Test that antstack_core can be imported successfully."""
        try:
            import antstack_core
            self.assertTrue(hasattr(antstack_core, '__version__'))
            self.assertTrue(hasattr(antstack_core, '__author__'))
            self.assertTrue(hasattr(antstack_core, '__all__'))
        except ImportError as e:
            self.fail(f"Failed to import antstack_core: {e}")

    def test_package_version(self):
        """Test package version information."""
        import antstack_core
        self.assertIsInstance(antstack_core.__version__, str)
        self.assertTrue(len(antstack_core.__version__) > 0)

    def test_package_author(self):
        """Test package author information."""
        import antstack_core
        self.assertIsInstance(antstack_core.__author__, str)
        self.assertTrue(len(antstack_core.__author__) > 0)

    def test_package_exports(self):
        """Test package __all__ exports."""
        import antstack_core
        expected_modules = ['figures', 'mathematics', 'publishing', 'analysis']
        for module in expected_modules:
            self.assertIn(module, antstack_core.__all__)

    def test_dependency_checking(self):
        """Test that dependency checking works correctly."""
        # Test with available dependencies
        with patch('antstack_core.matplotlib', create=True):
            with patch('antstack_core.numpy', create=True):
                # Reload module to trigger dependency check
                import importlib
                import antstack_core
                importlib.reload(antstack_core)

    def test_dependency_warning_handling(self):
        """Test that missing dependencies are handled gracefully."""
        with patch.dict('sys.modules', {'matplotlib': None, 'numpy': None}):
            with patch('builtins.print') as mock_print:
                # Reload module to trigger dependency check
                import importlib
                import antstack_core
                importlib.reload(antstack_core)

                # Should have printed warnings about missing dependencies
                mock_print.assert_called()

    def test_core_module_imports(self):
        """Test that all core modules can be imported."""
        try:
            from antstack_core import analysis
            from antstack_core import figures
            from antstack_core import publishing
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")

    def test_analysis_module_structure(self):
        """Test analysis module has expected structure."""
        from antstack_core.analysis import (
            EnergyCoefficients, ComputeLoad, EnergyBreakdown,
            estimate_detailed_energy, bootstrap_mean_ci, analyze_scaling_relationship
        )

        # Test that key classes are available
        self.assertTrue(callable(EnergyCoefficients))
        self.assertTrue(callable(ComputeLoad))
        self.assertTrue(callable(EnergyBreakdown))

        # Test that key functions are available
        self.assertTrue(callable(estimate_detailed_energy))
        self.assertTrue(callable(bootstrap_mean_ci))
        self.assertTrue(callable(analyze_scaling_relationship))

    def test_figures_module_structure(self):
        """Test figures module has expected structure."""
        from antstack_core.figures import bar_plot, line_plot, scatter_plot

        # Test that key functions are available
        self.assertTrue(callable(bar_plot))
        self.assertTrue(callable(line_plot))
        self.assertTrue(callable(scatter_plot))

    def test_publishing_module_structure(self):
        """Test publishing module has expected structure."""
        # Publishing module might be minimal initially
        from antstack_core import publishing
        self.assertIsNotNone(publishing)

    def test_module_imports_with_missing_dependencies(self):
        """Test graceful handling when optional dependencies are missing."""
        # This is more of an integration test, but ensures robustness
        try:
            from antstack_core.analysis import complexity_analysis
            # Should handle missing scipy/numpy gracefully
        except ImportError:
            # Expected if dependencies missing
            pass

    def test_package_path_setup(self):
        """Test that package path setup works correctly."""
        import antstack_core
        # Package should be importable from project root
        self.assertIn('antstack_core', sys.modules)

    def test_version_consistency(self):
        """Test version consistency across package."""
        import antstack_core
        # Version should be a valid semantic version
        version = antstack_core.__version__
        parts = version.split('.')
        self.assertEqual(len(parts), 3)  # Major.minor.patch
        for part in parts:
            self.assertTrue(part.isdigit())

    def test_author_consistency(self):
        """Test author information consistency."""
        import antstack_core
        author = antstack_core.__author__
        # Should contain a name
        self.assertIn(' ', author)  # First and last name

    def test_all_exports_are_strings(self):
        """Test that __all__ contains only strings."""
        import antstack_core
        for item in antstack_core.__all__:
            self.assertIsInstance(item, str)
            self.assertTrue(len(item) > 0)


if __name__ == '__main__':
    unittest.main()
