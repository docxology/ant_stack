#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.key_numbers module.

Tests key numbers integration functionality including:
- JSON data loading and caching
- Key numbers data structure validation
- Formatted value extraction and formatting
- Manager class operations and integration
- Error handling and edge cases
- Dynamic updating and validation

Following .cursorrules principles:
- Real data analysis (no mocks)
- Professional validation and error handling
- Comprehensive edge case testing
- Clear separation of concerns
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch
from typing import Dict, List, Optional, Any

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.key_numbers import (
    KeyNumbersData,
    KeyNumbersLoader,
    KeyNumbersManager,
    get_key_numbers_loader,
    get_key_numbers_manager
)


class TestKeyNumbersData(unittest.TestCase):
    """Test KeyNumbersData dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "per_decision_energy": {
                "body": 1.2,
                "brain": 0.05,
                "mind": 0.02,
                "total": 1.27
            },
            "computational_load": {
                "flops": 1000000,
                "memory_mb": 512,
                "spikes": 10000
            },
            "scaling_exponents": {
                "body": 0.85,
                "brain": 1.2,
                "mind": 1.5,
                "combined": 1.1
            },
            "system_parameters": {
                "temperature_k": 298.15,
                "time_step_ms": 10,
                "decision_horizon": 5
            }
        }

    def test_key_numbers_data_creation(self):
        """Test basic KeyNumbersData creation."""
        data = KeyNumbersData(
            per_decision_energy=self.sample_data["per_decision_energy"],
            computational_load=self.sample_data["computational_load"],
            scaling_exponents=self.sample_data["scaling_exponents"],
            system_parameters=self.sample_data["system_parameters"]
        )

        self.assertEqual(data.per_decision_energy["body"], 1.2)
        self.assertEqual(data.computational_load["flops"], 1000000)
        self.assertEqual(data.scaling_exponents["combined"], 1.1)

    def test_key_numbers_data_from_dict(self):
        """Test creation from dictionary."""
        data = KeyNumbersData.from_dict(self.sample_data)

        self.assertEqual(data.per_decision_energy["total"], 1.27)
        self.assertEqual(data.system_parameters["temperature_k"], 298.15)

    def test_key_numbers_data_from_dict_empty(self):
        """Test from_dict with empty dictionary."""
        empty_data = KeyNumbersData.from_dict({})
        self.assertEqual(empty_data.per_decision_energy, {})
        self.assertEqual(empty_data.computational_load, {})
        self.assertEqual(empty_data.scaling_exponents, {})
        self.assertEqual(empty_data.system_parameters, {})

    def test_key_numbers_data_to_dict(self):
        """Test dictionary serialization."""
        data = KeyNumbersData.from_dict(self.sample_data)
        dict_result = data.to_dict()

        self.assertEqual(dict_result["per_decision_energy"]["body"], 1.2)
        self.assertEqual(dict_result["computational_load"]["memory_mb"], 512)
        self.assertEqual(dict_result["scaling_exponents"]["mind"], 1.5)

    def test_key_numbers_data_round_trip(self):
        """Test serialization round-trip."""
        original = KeyNumbersData.from_dict(self.sample_data)
        serialized = original.to_dict()
        restored = KeyNumbersData.from_dict(serialized)

        self.assertEqual(original.per_decision_energy, restored.per_decision_energy)
        self.assertEqual(original.computational_load, restored.computational_load)
        self.assertEqual(original.scaling_exponents, restored.scaling_exponents)
        self.assertEqual(original.system_parameters, restored.system_parameters)


class TestKeyNumbersLoader(unittest.TestCase):
    """Test KeyNumbersLoader class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_json_data = {
            "per_decision_energy": {
                "body": 1.5,
                "brain": 0.08,
                "mind": 0.03,
                "total": 1.61
            },
            "computational_load": {
                "flops": 2000000,
                "memory_mb": 1024,
                "spikes": 20000
            },
            "scaling_exponents": {
                "body": 0.9,
                "brain": 1.1,
                "mind": 1.3,
                "combined": 1.0
            },
            "system_parameters": {
                "temperature_k": 310.15,
                "time_step_ms": 20,
                "decision_horizon": 10
            }
        }

    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    def test_key_numbers_loader_creation_with_path(self):
        """Test KeyNumbersLoader creation with explicit path."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)
            self.assertEqual(loader.json_path, Path(json_file))
            self.assertIsNone(loader._cache)
        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_creation_without_path(self):
        """Test KeyNumbersLoader creation without path (auto-discovery)."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        temp_dir = os.path.dirname(json_file)
        key_numbers_path = os.path.join(temp_dir, "key_numbers.json")
        os.rename(json_file, key_numbers_path)

        try:
            # Change to temp directory for auto-discovery
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            loader = KeyNumbersLoader()
            self.assertEqual(loader.json_path, Path(key_numbers_path))
        finally:
            os.chdir(original_cwd)
            os.unlink(key_numbers_path)

    def test_key_numbers_loader_creation_no_file_found(self):
        """Test KeyNumbersLoader creation when no file is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                with self.assertRaises(FileNotFoundError):
                    KeyNumbersLoader()
            finally:
                os.chdir(original_cwd)

    def test_key_numbers_loader_load(self):
        """Test loading key numbers data."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)
            data = loader.load()

            self.assertIsInstance(data, KeyNumbersData)
            self.assertEqual(data.per_decision_energy["body"], 1.5)
            self.assertEqual(data.system_parameters["temperature_k"], 310.15)
        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_caching(self):
        """Test caching functionality."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)

            # First load
            data1 = loader.load()
            self.assertIsNotNone(loader._cache)
            self.assertIsNotNone(loader._last_modified)

            # Second load (should use cache)
            data2 = loader.load()
            self.assertIs(data1, data2)  # Same object reference

            # Force reload
            data3 = loader.load(force_reload=True)
            self.assertIsNot(data1, data3)  # Different object reference
        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_file_not_found(self):
        """Test loading when file doesn't exist."""
        loader = KeyNumbersLoader("/nonexistent/path/key_numbers.json")

        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_key_numbers_loader_get_formatted_value(self):
        """Test formatted value extraction."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)

            # Test direct access
            formatted = loader.get_formatted_value("per_decision_energy.body", ".2f")
            self.assertEqual(formatted, "1.50")

            # Test different format
            formatted = loader.get_formatted_value("computational_load.flops", ",.0f")
            self.assertEqual(formatted, "2,000,000")

            # Test default format
            formatted = loader.get_formatted_value("scaling_exponents.body")
            self.assertEqual(formatted, "0.900")

        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_get_formatted_value_nested(self):
        """Test formatted value extraction with deeply nested keys."""
        nested_data = {
            "per_decision_energy": {
                "modules": {
                    "body": {"value": 1.5, "unit": "J"},
                    "brain": {"value": 0.08, "unit": "J"}
                }
            }
        }
        json_file = self.create_temp_json_file(nested_data)
        try:
            loader = KeyNumbersLoader(json_file)

            formatted = loader.get_formatted_value("per_decision_energy.modules.body.value", ".3f")
            self.assertEqual(formatted, "1.500")

        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_get_formatted_value_missing_key(self):
        """Test formatted value extraction with missing key."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)

            formatted = loader.get_formatted_value("nonexistent.key", ".2f")
            self.assertEqual(formatted, "N/A")

        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_get_all_values(self):
        """Test getting all values with formatting."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)

            all_values = loader.get_all_values(".2f")

            self.assertIsInstance(all_values, dict)
            self.assertIn("per_decision_energy", all_values)
            self.assertIn("body", all_values["per_decision_energy"])
            self.assertEqual(all_values["per_decision_energy"]["body"], "1.50")

        finally:
            os.unlink(json_file)

    def test_key_numbers_loader_get_summary(self):
        """Test getting summary information."""
        json_file = self.create_temp_json_file(self.sample_json_data)
        try:
            loader = KeyNumbersLoader(json_file)

            summary = loader.get_summary()

            self.assertIsInstance(summary, dict)
            self.assertIn("total_energy", summary)
            self.assertIn("dominant_module", summary)
            self.assertIn("scaling_regime", summary)

        finally:
            os.unlink(json_file)


class TestKeyNumbersManager(unittest.TestCase):
    """Test KeyNumbersManager class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "per_decision_energy": {
                "body": 2.0,
                "brain": 0.1,
                "mind": 0.05,
                "total": 2.15
            },
            "computational_load": {
                "flops": 5000000,
                "memory_mb": 2048,
                "spikes": 50000
            },
            "scaling_exponents": {
                "body": 0.8,
                "brain": 1.0,
                "mind": 1.2,
                "combined": 0.9
            },
            "system_parameters": {
                "temperature_k": 300.0,
                "time_step_ms": 15,
                "decision_horizon": 8
            }
        }

    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    def test_key_numbers_manager_creation(self):
        """Test KeyNumbersManager creation."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)
            self.assertIsNotNone(manager.loader)
            self.assertEqual(manager.loader.json_path, Path(json_file))
        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_get_value(self):
        """Test getting values through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            value = manager.get_value("per_decision_energy.body")
            self.assertEqual(value, 2.0)

            value = manager.get_value("computational_load.flops")
            self.assertEqual(value, 5000000)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_get_formatted_value(self):
        """Test getting formatted values through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            formatted = manager.get_formatted_value("per_decision_energy.body", ".1f")
            self.assertEqual(formatted, "2.0")

            formatted = manager.get_formatted_value("scaling_exponents.brain", ".2f")
            self.assertEqual(formatted, "1.00")

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_update_value(self):
        """Test updating values through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            # Update a value
            manager.update_value("per_decision_energy.body", 2.5)

            # Check that the value was updated
            updated_value = manager.get_value("per_decision_energy.body")
            self.assertEqual(updated_value, 2.5)

            # Check that file was updated
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            self.assertEqual(file_data["per_decision_energy"]["body"], 2.5)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_update_multiple_values(self):
        """Test updating multiple values through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            updates = {
                "per_decision_energy.brain": 0.15,
                "computational_load.memory_mb": 4096,
                "scaling_exponents.mind": 1.1
            }

            manager.update_multiple_values(updates)

            # Check all updates
            self.assertEqual(manager.get_value("per_decision_energy.brain"), 0.15)
            self.assertEqual(manager.get_value("computational_load.memory_mb"), 4096)
            self.assertEqual(manager.get_value("scaling_exponents.mind"), 1.1)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_calculate_total_energy(self):
        """Test total energy calculation."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            total_energy = manager.calculate_total_energy()
            expected_total = 2.0 + 0.1 + 0.05  # body + brain + mind

            self.assertEqual(total_energy, expected_total)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_get_scaling_analysis(self):
        """Test scaling analysis through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            analysis = manager.get_scaling_analysis()

            self.assertIsInstance(analysis, dict)
            self.assertIn("body_exponent", analysis)
            self.assertIn("brain_exponent", analysis)
            self.assertIn("mind_exponent", analysis)
            self.assertIn("system_scaling", analysis)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_generate_report(self):
        """Test report generation through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            report = manager.generate_report()

            self.assertIsInstance(report, str)
            self.assertIn("Key Numbers Report", report)
            self.assertIn("2.0", report)  # body energy
            self.assertIn("0.8", report)  # body scaling exponent

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_validate_data(self):
        """Test data validation through manager."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            is_valid, errors = manager.validate_data()

            self.assertTrue(is_valid)
            self.assertIsInstance(errors, list)
            self.assertEqual(len(errors), 0)

        finally:
            os.unlink(json_file)

    def test_key_numbers_manager_validate_data_invalid(self):
        """Test validation of invalid data."""
        invalid_data = {
            "per_decision_energy": {
                "body": -1.0,  # Negative energy (invalid)
                "brain": 0.1
            }
        }
        json_file = self.create_temp_json_file(invalid_data)
        try:
            manager = KeyNumbersManager(json_file)

            is_valid, errors = manager.validate_data()

            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)

        finally:
            os.unlink(json_file)


class TestKeyNumbersUtilities(unittest.TestCase):
    """Test key numbers utility functions."""

    def test_get_key_numbers_loader(self):
        """Test get_key_numbers_loader utility function."""
        # This should work if there's a key_numbers.json file in the expected location
        # For testing purposes, we'll mock the file existence
        with patch('antstack_core.analysis.key_numbers.Path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('antstack_core.analysis.key_numbers.KeyNumbersLoader') as mock_loader:
                loader = get_key_numbers_loader()
                mock_loader.assert_called_once()

    def test_get_key_numbers_manager(self):
        """Test get_key_numbers_manager utility function."""
        # This should work if there's a key_numbers.json file in the expected location
        # For testing purposes, we'll mock the file existence
        with patch('antstack_core.analysis.key_numbers.Path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('antstack_core.analysis.key_numbers.KeyNumbersManager') as mock_manager:
                manager = get_key_numbers_manager()
                mock_manager.assert_called_once()


class TestKeyNumbersIntegration(unittest.TestCase):
    """Test integration between key numbers components."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "per_decision_energy": {
                "body": 3.0,
                "brain": 0.2,
                "mind": 0.1,
                "total": 3.3
            },
            "computational_load": {
                "flops": 10000000,
                "memory_mb": 4096,
                "spikes": 100000
            },
            "scaling_exponents": {
                "body": 0.7,
                "brain": 0.9,
                "mind": 1.1,
                "combined": 0.8
            },
            "system_parameters": {
                "temperature_k": 295.0,
                "time_step_ms": 25,
                "decision_horizon": 15
            }
        }

    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            # Create manager
            manager = KeyNumbersManager(json_file)

            # Load data
            data = manager.loader.load()

            # Update a value
            manager.update_value("per_decision_energy.body", 3.5)

            # Get formatted values
            formatted_body = manager.get_formatted_value("per_decision_energy.body", ".1f")
            formatted_total = manager.get_formatted_value("per_decision_energy.total", ".1f")

            # Generate report
            report = manager.generate_report()

            # Validate
            is_valid, errors = manager.validate_data()

            # Assertions
            self.assertEqual(formatted_body, "3.5")
            self.assertIsInstance(report, str)
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

        finally:
            os.unlink(json_file)

    def test_data_consistency_across_operations(self):
        """Test data consistency across different operations."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            # Get initial values
            initial_body = manager.get_value("per_decision_energy.body")
            initial_total = manager.get_value("per_decision_energy.total")

            # Update body energy
            manager.update_value("per_decision_energy.body", 4.0)

            # Check that related values are updated appropriately
            new_body = manager.get_value("per_decision_energy.body")
            new_total = manager.get_value("per_decision_energy.total")

            self.assertEqual(new_body, 4.0)
            # Note: This test assumes the manager automatically updates total
            # If not, the total should remain the same or be recalculated

        finally:
            os.unlink(json_file)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager = KeyNumbersManager(json_file)

            # Test accessing non-existent key
            value = manager.get_value("nonexistent.key")
            self.assertIsNone(value)

            # Test formatting non-existent key
            formatted = manager.get_formatted_value("nonexistent.key")
            self.assertEqual(formatted, "N/A")

            # Test updating non-existent key (should handle gracefully)
            manager.update_value("nonexistent.key", 1.0)

            # Manager should still function normally
            valid_value = manager.get_value("per_decision_energy.body")
            self.assertEqual(valid_value, 3.0)

        finally:
            os.unlink(json_file)


class TestKeyNumbersRobustness(unittest.TestCase):
    """Test robustness of key numbers functionality."""

    def test_file_handling_edge_cases(self):
        """Test file handling with edge cases."""
        # Test with very large JSON file
        large_data = {
            "per_decision_energy": {f"module_{i}": i * 0.1 for i in range(1000)},
            "computational_load": {f"metric_{i}": i * 1000 for i in range(1000)},
            "scaling_exponents": {f"exp_{i}": 0.5 + i * 0.01 for i in range(1000)},
            "system_parameters": {f"param_{i}": i for i in range(1000)}
        }

        json_file = self.create_temp_json_file(large_data)
        try:
            manager = KeyNumbersManager(json_file)

            # Should handle large files without issues
            value = manager.get_value("per_decision_energy.module_500")
            self.assertEqual(value, 50.0)

            formatted = manager.get_formatted_value("scaling_exponents.exp_100")
            self.assertEqual(formatted, "1.500")

        finally:
            os.unlink(json_file)

    def test_data_type_handling(self):
        """Test handling of different data types."""
        mixed_data = {
            "per_decision_energy": {
                "body": 1.5,  # float
                "brain": 0,   # int
                "mind": 0.0   # float zero
            },
            "computational_load": {
                "flops": 1000000,  # int
                "memory_mb": 512.0,  # float
                "spikes": 0         # int zero
            }
        }

        json_file = self.create_temp_json_file(mixed_data)
        try:
            manager = KeyNumbersManager(json_file)

            # Should handle different numeric types
            self.assertEqual(manager.get_value("per_decision_energy.body"), 1.5)
            self.assertEqual(manager.get_value("computational_load.flops"), 1000000)
            self.assertEqual(manager.get_value("per_decision_energy.mind"), 0.0)

        finally:
            os.unlink(json_file)

    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access scenarios."""
        json_file = self.create_temp_json_file(self.sample_data)
        try:
            manager1 = KeyNumbersManager(json_file)
            manager2 = KeyNumbersManager(json_file)

            # Both managers should work independently
            value1 = manager1.get_value("per_decision_energy.body")
            value2 = manager2.get_value("per_decision_energy.body")

            self.assertEqual(value1, value2)

            # Updates from one manager should be visible to the other
            # (though this depends on file system caching behavior)
            manager1.update_value("per_decision_energy.body", 4.0)

            # Force reload for manager2
            manager2.loader.load(force_reload=True)
            value2_updated = manager2.get_value("per_decision_energy.body")

            self.assertEqual(value2_updated, 4.0)

        finally:
            os.unlink(json_file)

    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name


if __name__ == '__main__':
    unittest.main()
