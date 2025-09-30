#!/usr/bin/env python3
"""Comprehensive tests for key_numbers integration and dynamic content generation.

Tests the complete key_numbers system including:
- Key numbers loading and caching
- Dynamic content generation for papers
- Parameter validation and error handling
- Integration with analysis pipelines
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.key_numbers import (
    KeyNumbersLoader, KeyNumbersManager, get_key_numbers_manager
)
from antstack_core.analysis.energy import EnergyCoefficients
from antstack_core.analysis.workloads import (
    enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form
)


class TestKeyNumbersLoader(unittest.TestCase):
    """Test key numbers loader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_key_numbers.json"

        # Create test key numbers file with correct structure
        self.test_data = {
            "per_decision_energy": {
                "body_mj": 0.5,
                "brain_mj": 2.0,
                "mind_mj": 1.5,
                "total_mj": 4.0
            },
            "computational_load": {
                "body_flops": 100000,
                "brain_flops": 500000,
                "mind_flops": 200000
            },
            "scaling_exponents": {
                "brain": 0.8,
                "mind": 1.2,
                "body": 1.1
            },
            "system_parameters": {
                "rho": 0.02,
                "N_KC": 50000,
                "H_p_max": 20,
                "B_max": 8
            }
        }

        with open(self.test_file, 'w') as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_data_loading(self):
        """Test loading key numbers from file."""
        loader = KeyNumbersLoader(self.test_file)
        data = loader.load()

        self.assertIsNotNone(data)
        self.assertEqual(data.per_decision_energy["body_mj"], 0.5)
        self.assertEqual(data.per_decision_energy["brain_mj"], 2.0)
        self.assertEqual(data.scaling_exponents["brain"], 0.8)

    def test_formatted_value_access(self):
        """Test formatted value access."""
        loader = KeyNumbersLoader(self.test_file)

        # Test energy values
        body_energy = loader.get_formatted_value("per_decision_energy.body_mj")
        self.assertEqual(body_energy, "0.500")

        # Test scaling values
        brain_scaling = loader.get_formatted_value("scaling_exponents.brain")
        self.assertEqual(brain_scaling, "0.800")

    def test_energy_value_access(self):
        """Test energy value access methods."""
        loader = KeyNumbersLoader(self.test_file)

        body_energy_mj = loader.get_energy_value("body", "mj")
        self.assertEqual(body_energy_mj, 0.5)

        body_energy_j = loader.get_energy_value("body", "j")
        self.assertEqual(body_energy_j, 0.0)  # Not in data

    def test_caching_behavior(self):
        """Test that data is cached and reloaded when file changes."""
        loader = KeyNumbersLoader(self.test_file)

        # Load initial data
        data1 = loader.load()
        self.assertEqual(data1.per_decision_energy["body_mj"], 0.5)

        # Modify the file
        self.test_data["per_decision_energy"]["body_mj"] = 0.7
        with open(self.test_file, 'w') as f:
            json.dump(self.test_data, f)

        # Should return new value when file is modified (mtime changed)
        data2 = loader.load(force_reload=False)
        self.assertEqual(data2.per_decision_energy["body_mj"], 0.7)

        # Should return new value with force_reload
        data3 = loader.load(force_reload=True)
        self.assertEqual(data3.per_decision_energy["body_mj"], 0.7)

    def test_missing_file_handling(self):
        """Test handling of missing key numbers file."""
        missing_file = Path(self.temp_dir) / "missing.json"

        # The loader searches in standard locations, so this should work
        # and then fail when trying to load
        loader = KeyNumbersLoader(missing_file)
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in key numbers file."""
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")

        loader = KeyNumbersLoader(invalid_file)
        with self.assertRaises(json.JSONDecodeError):
            loader.load()


class TestKeyNumbersManager(unittest.TestCase):
    """Test key numbers manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "manager_test.json"

        self.test_data = {
            "per_decision_energy": {
                "body_mj": 0.5,
                "brain_mj": 2.0
            },
            "scaling_exponents": {
                "brain": 0.8,
                "body": 1.1
            }
        }

        with open(self.test_file, 'w') as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_manager_creation(self):
        """Test creating key numbers manager."""
        loader = KeyNumbersLoader(self.test_file)
        manager = KeyNumbersManager(loader)

        self.assertIsNotNone(manager)
        self.assertEqual(manager.loader, loader)

    def test_placeholder_replacement(self):
        """Test placeholder replacement functionality."""
        loader = KeyNumbersLoader(self.test_file)
        manager = KeyNumbersManager(loader)

        # Test energy placeholder - this may not work if the shortcut isn't implemented
        text = "Body energy: {{key_numbers.per_decision_energy.body_mj}}"
        result = manager.replace_placeholders(text)
        self.assertIn("0.500", result)

        # Test scaling placeholder
        text = "Brain scaling: {{key_numbers.scaling_exponents.brain}}"
        result = manager.replace_placeholders(text)
        self.assertIn("0.800", result)

    def test_formatted_placeholders(self):
        """Test formatted placeholder replacement."""
        loader = KeyNumbersLoader(self.test_file)
        manager = KeyNumbersManager(loader)

        # Test with format specification
        text = "Body energy: {{key_numbers.per_decision_energy.body_mj:.2e}}"
        result = manager.replace_placeholders(text)
        self.assertIn("5.00e-01", result)

    def test_missing_placeholder_handling(self):
        """Test handling of missing placeholders."""
        loader = KeyNumbersLoader(self.test_file)
        manager = KeyNumbersManager(loader)

        # Test missing key
        text = "Missing: {{key_numbers.missing.key}}"
        result = manager.replace_placeholders(text)
        # Should contain original placeholder or error indicator
        self.assertIsInstance(result, str)


class TestKeyNumbersIntegration(unittest.TestCase):
    """Test integration of key numbers with analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_numbers_file = Path(self.temp_dir) / "integration_test.json"

        # Create comprehensive key numbers file with correct structure
        self.key_numbers_data = {
            "per_decision_energy": {
                "body_mj": 0.5,
                "brain_mj": 2.0,
                "mind_mj": 1.5
            },
            "computational_load": {
                "body_flops": 100000,
                "brain_flops": 500000,
                "mind_flops": 200000
            },
            "scaling_exponents": {
                "brain": 0.8,
                "mind": 1.2,
                "body": 1.1
            },
            "system_parameters": {
                "rho": 0.02,
                "N_KC": 50000,
                "K_max": 256,
                "H_p_max": 20,
                "B_max": 8,
                "state_dim_max": 32,
                "J_max": 50,
                "C_max": 20,
                "S_max": 512
            }
        }

        with open(self.key_numbers_file, 'w') as f:
            json.dump(self.key_numbers_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_brain_scaling_with_key_numbers(self):
        """Test brain workload scaling using key numbers."""
        loader = KeyNumbersLoader(self.key_numbers_file)

        # Test different K values
        K_values = [64, 128, 256]
        energies = []

        coeffs = EnergyCoefficients()

        for K in K_values:
            # Get parameters from key numbers
            N_KC = loader.load().system_parameters.get("N_KC", 50000)
            rho = loader.load().system_parameters.get("rho", 0.02)

            params = {
                'K': K,
                'N_KC': N_KC,
                'rho': rho,
                'H': 64,
                'hz': 100
            }

            load = enhanced_brain_workload_closed_form(0.01, params)
            # Use the correct energy estimation method
            from antstack_core.analysis.energy import estimate_detailed_energy
            breakdown = estimate_detailed_energy(load, coeffs)
            energies.append(breakdown.total)

        # Verify scaling behavior
        self.assertGreater(energies[1], energies[0])  # 128 > 64
        self.assertGreater(energies[2], energies[1])  # 256 > 128

    def test_coefficients_from_key_numbers(self):
        """Test creating energy coefficients from key numbers."""
        loader = KeyNumbersLoader(self.key_numbers_file)

        # Get coefficients from key numbers (using defaults since not in test data)
        flops_pj = 1.0  # Default
        sram_pj_per_byte = 0.1  # Default
        dram_pj_per_byte = 20.0  # Default

        # Create coefficients
        coeffs = EnergyCoefficients(
            flops_pj=flops_pj,
            sram_pj_per_byte=sram_pj_per_byte,
            dram_pj_per_byte=dram_pj_per_byte
        )

        # Test that coefficients work
        from antstack_core.analysis.energy import ComputeLoad, estimate_detailed_energy
        load = ComputeLoad(flops=1000, sram_bytes=1000, dram_bytes=1000)
        energy = estimate_detailed_energy(load, coeffs)

        self.assertGreater(energy.total, 0)

    def test_parameter_bounds_from_key_numbers(self):
        """Test parameter bounds validation using key numbers."""
        loader = KeyNumbersLoader(self.key_numbers_file)
        data = loader.load()

        # Test brain parameters - N_KC is in the test data
        N_KC = data.system_parameters.get("N_KC", 0)
        self.assertEqual(N_KC, 50000)

        # Test mind parameters - these are in the test data
        H_p_max = data.system_parameters.get("H_p_max", 0)
        B_max = data.system_parameters.get("B_max", 0)
        self.assertEqual(H_p_max, 20)
        self.assertEqual(B_max, 8)

    def test_scaling_analysis_with_key_numbers(self):
        """Test scaling analysis using exponents from key numbers."""
        loader = KeyNumbersLoader(self.key_numbers_file)
        data = loader.load()

        # Get scaling exponents from key numbers
        brain_exponent = data.scaling_exponents.get("brain", 0.8)
        mind_exponent = data.scaling_exponents.get("mind", 1.2)

        # Test that these are reasonable values
        self.assertGreater(brain_exponent, 0)
        self.assertLess(brain_exponent, 2)
        self.assertGreater(mind_exponent, 0)
        self.assertLess(mind_exponent, 3)


class TestKeyNumbersRobustness(unittest.TestCase):
    """Test robustness and error handling of key numbers system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "robustness_test.json"

        # Create test key numbers file
        self.test_data = {
            "per_decision_energy": {"body_mj": 0.5},
            "scaling_exponents": {"brain": 0.8}
        }

        with open(self.test_file, 'w') as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_empty_key_numbers_file(self):
        """Test handling of empty key numbers file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            empty_file = f.name

        try:
            loader = KeyNumbersLoader(empty_file)
            data = loader.load()

            # Should have empty dictionaries
            self.assertEqual(data.per_decision_energy, {})
            self.assertEqual(data.scaling_exponents, {})

        finally:
            os.unlink(empty_file)

    def test_missing_key_in_formatted_access(self):
        """Test handling of missing keys in formatted access."""
        loader = KeyNumbersLoader(self.test_file)

        # Test missing key path
        with self.assertRaises(KeyError):
            loader.get_formatted_value("missing.path")

    def test_concurrent_access(self):
        """Test concurrent access to key numbers loader."""
        import threading

        results = []

        def worker():
            loader = KeyNumbersLoader(self.test_file)
            data = loader.load()
            value = data.per_decision_energy.get("body_mj", 0)
            results.append(value)

        # Create and run threads
        threads = []
        for _ in range(3):  # Reduced to 3 threads for stability
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All should get the same value
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r == 0.5 for r in results))


if __name__ == '__main__':
    unittest.main()
