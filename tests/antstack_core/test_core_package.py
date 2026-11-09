#!/usr/bin/env python3
"""Behavioral regressions for explicit dependencies and quiet package imports."""

import unittest
from unittest.mock import patch


class TestCorePackage(unittest.TestCase):
    """Dependency discovery and import-side-effect behavior."""

    def test_dependency_checking(self):
        """Test that explicit dependency checking reports importable packages."""
        import antstack_core
        missing = antstack_core.check_runtime_dependencies(("sys",))
        self.assertEqual(missing, ())

    def test_dependency_checking_missing_packages(self):
        """Test that missing dependencies are returned without import-time output."""
        import antstack_core
        with patch('antstack_core.find_spec', return_value=None):
            missing = antstack_core.check_runtime_dependencies(("matplotlib", "numpy"))

        self.assertEqual(missing, ("matplotlib", "numpy"))

    def test_package_import_has_no_stdout_side_effect(self):
        """Importing the package should not print dependency warnings."""
        import importlib
        import antstack_core
        with patch('builtins.print') as mock_print:
            importlib.reload(antstack_core)
        mock_print.assert_not_called()


if __name__ == '__main__':
    unittest.main()
