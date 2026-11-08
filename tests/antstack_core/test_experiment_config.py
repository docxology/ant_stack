#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.experiment_config module.

Tests experiment configuration and manifest management including:
- Workload configuration and parameterization
- Energy coefficient configuration
- Scaling analysis parameters
- Power measurement configuration
- Experiment provenance tracking
- YAML/JSON serialization and validation

Following .cursorrules principles:
- Real configuration management (no mocks)
- Professional validation and error handling
- Comprehensive edge case testing
- Clear separation of concerns
"""

import unittest
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.experiment_config import (
    WorkloadConfig,
    EnergyCoefficientsConfig,
    ScalingConfig,
    MeterConfig,
    ExperimentManifest
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

class TestScientificManifestDomains(unittest.TestCase):
    @staticmethod
    def manifest(**kwargs):
        return ExperimentManifest(workloads={'body': WorkloadConfig(name='body')}, **kwargs)

    def test_missing_optional_experiment_name_has_meaningful_default(self):
        manifest = ExperimentManifest.from_dict({'workloads': {'body': {'name': 'body', 'mode': 'closed_form'}}})
        self.assertTrue(manifest.experiment_name.strip())
        self.assertEqual(manifest.validate(), (True, []))

    def test_all_coefficient_domains_and_zero_prices(self):
        for field in EnergyCoefficientsConfig.__dataclass_fields__:
            for bad in (-1, float('nan'), float('inf')):
                manifest = self.manifest(coefficients=EnergyCoefficientsConfig(**{field: bad}))
                with self.subTest(field=field, bad=bad):
                    self.assertFalse(manifest.validate()[0])
            zero = self.manifest(coefficients=EnergyCoefficientsConfig(**{field: 0}))
            self.assertEqual(zero.validate(), (True, []))

    def test_invalid_workload_duration_mode_names_and_parameters(self):
        for bad in (-1, 0, float('nan'), float('inf')):
            manifest = ExperimentManifest(workloads={'body': WorkloadConfig(name='body', duration_s=bad)})
            self.assertFalse(manifest.validate()[0])
        for bad in (0, -1, 1.5, float('nan'), float('inf')):
            manifest = ExperimentManifest(workloads={'body': WorkloadConfig(name='body', repeats=bad)})
            self.assertFalse(manifest.validate()[0])
        self.assertFalse(ExperimentManifest(workloads={'typo': WorkloadConfig(name='typo')}).validate()[0])
        self.assertFalse(ExperimentManifest(workloads={'body': WorkloadConfig(name='body', mode='typo')}).validate()[0])
        invalid = (
            ('body', {'J': -1}), ('body', {'contact_solver': 'lcp'}),
            ('brain', {'rho': 1.1}), ('brain', {'rho': float('nan')}),
            ('brain', {'connectivity_pattern': 'biological'}),
            ('mind', {'B': 0}), ('mind', {'branching': 4}), ('mind', {'max_policies': 0}),
        )
        for name, params in invalid:
            manifest = ExperimentManifest(workloads={name: WorkloadConfig(name=name, params=params)})
            with self.subTest(name=name, params=params):
                self.assertFalse(manifest.validate()[0])

    def test_invalid_physical_and_meter_values_are_not_valid(self):
        for field in ('mass_kg', 'distance_m'):
            for bad in (-1, 0, float('nan'), float('inf')):
                self.assertFalse(self.manifest(**{field: bad}).validate()[0])
        for config in ({'meter_type': 'hardware'}, {'device_index': -1}, {'device_index': True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MeterConfig.from_dict(config)



class TestWorkloadConfig(unittest.TestCase):
    """Test WorkloadConfig class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_params = {
            'joints': 6,
            'velocity': 1.0,
            'mass': 0.001,
            'channels': 100,
            'sparsity': 0.1,
            'horizon': 5,
            'branching': 3
        }

        self.config = WorkloadConfig(
            name="test_workload",
            duration_s=1.5,
            repeats=10,
            params=self.sample_params,
            mode="closed_form"
        )

    def test_workload_config_creation(self):
        """Test basic WorkloadConfig creation."""
        config = WorkloadConfig(name="simple_workload")
        self.assertEqual(config.name, "simple_workload")
        self.assertEqual(config.duration_s, 2.0)  # default
        self.assertEqual(config.repeats, 5)      # default
        self.assertEqual(config.params, {})      # default
        self.assertIsNone(config.mode)           # default

    def test_workload_config_with_params(self):
        """Test WorkloadConfig with custom parameters."""
        self.assertEqual(self.config.name, "test_workload")
        self.assertEqual(self.config.duration_s, 1.5)
        self.assertEqual(self.config.repeats, 10)
        self.assertEqual(self.config.params, self.sample_params)
        self.assertEqual(self.config.mode, "closed_form")

    def test_workload_config_post_init(self):
        """Test __post_init__ method."""
        config = WorkloadConfig(name="test")
        self.assertIsInstance(config.params, dict)
        self.assertEqual(config.params, {})

    def test_workload_config_to_dict(self):
        """Test dictionary serialization."""
        config_dict = self.config.to_dict()

        expected_keys = ['name', 'duration_s', 'repeats', 'params', 'mode']
        for key in expected_keys:
            self.assertIn(key, config_dict)

        self.assertEqual(config_dict['name'], "test_workload")
        self.assertEqual(config_dict['duration_s'], 1.5)
        self.assertEqual(config_dict['repeats'], 10)
        self.assertEqual(config_dict['mode'], "closed_form")

    def test_workload_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'name': 'restored_workload',
            'duration_s': 3.0,
            'repeats': 20,
            'params': {'test_param': 'test_value'},
            'mode': 'loop'
        }

        config = WorkloadConfig.from_dict(config_dict)

        self.assertEqual(config.name, 'restored_workload')
        self.assertEqual(config.duration_s, 3.0)
        self.assertEqual(config.repeats, 20)
        self.assertEqual(config.params, {'test_param': 'test_value'})
        self.assertEqual(config.mode, 'loop')

    def test_workload_config_from_dict_defaults(self):
        """Test from_dict with missing fields."""
        minimal_dict = {'name': 'minimal'}
        config = WorkloadConfig.from_dict(minimal_dict)

        self.assertEqual(config.name, 'minimal')
        self.assertEqual(config.duration_s, 2.0)  # default
        self.assertEqual(config.repeats, 5)      # default
        self.assertEqual(config.params, {})      # default
        self.assertIsNone(config.mode)           # default

    def test_workload_config_round_trip(self):
        """Test serialization round-trip."""
        original_config = self.config
        config_dict = original_config.to_dict()
        restored_config = WorkloadConfig.from_dict(config_dict)

        self.assertEqual(original_config.name, restored_config.name)
        self.assertEqual(original_config.duration_s, restored_config.duration_s)
        self.assertEqual(original_config.repeats, restored_config.repeats)
        self.assertEqual(original_config.params, restored_config.params)
        self.assertEqual(original_config.mode, restored_config.mode)


class TestEnergyCoefficientsConfig(unittest.TestCase):
    """Test EnergyCoefficientsConfig class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EnergyCoefficientsConfig(
            flops_pj=2.0,
            sram_pj_per_byte=0.2,
            dram_pj_per_byte=25.0,
            spike_aj=2.0,
            body_per_joint_w=3.0,
            body_sensor_w_per_channel=0.01,
            baseline_w=1.0
        )

    def test_energy_coefficients_config_creation(self):
        """Test basic EnergyCoefficientsConfig creation."""
        config = EnergyCoefficientsConfig()
        self.assertEqual(config.flops_pj, 1.0)  # default
        self.assertEqual(config.sram_pj_per_byte, 0.1)  # default
        self.assertEqual(config.dram_pj_per_byte, 20.0)  # default

    def test_energy_coefficients_config_custom_values(self):
        """Test EnergyCoefficientsConfig with custom values."""
        self.assertEqual(self.config.flops_pj, 2.0)
        self.assertEqual(self.config.sram_pj_per_byte, 0.2)
        self.assertEqual(self.config.dram_pj_per_byte, 25.0)
        self.assertEqual(self.config.spike_aj, 2.0)
        self.assertEqual(self.config.body_per_joint_w, 3.0)
        self.assertEqual(self.config.body_sensor_w_per_channel, 0.01)
        self.assertEqual(self.config.baseline_w, 1.0)

    def test_energy_coefficients_config_to_energy_coefficients(self):
        """Test conversion to EnergyCoefficients."""
        from antstack_core.analysis.energy import EnergyCoefficients

        energy_coeffs = self.config.to_energy_coefficients()

        self.assertIsInstance(energy_coeffs, EnergyCoefficients)
        self.assertEqual(energy_coeffs.flops_pj, 2.0)
        self.assertEqual(energy_coeffs.sram_pj_per_byte, 0.2)
        self.assertEqual(energy_coeffs.dram_pj_per_byte, 25.0)

    def test_energy_coefficients_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'flops_pj': 3.0,
            'sram_pj_per_byte': 0.3,
            'dram_pj_per_byte': 30.0,
            'spike_aj': 3.0,
            'body_per_joint_w': 4.0,
            'body_sensor_w_per_channel': 0.02,
            'baseline_w': 2.0
        }

        config = EnergyCoefficientsConfig.from_dict(config_dict)

        self.assertEqual(config.flops_pj, 3.0)
        self.assertEqual(config.sram_pj_per_byte, 0.3)
        self.assertEqual(config.baseline_w, 2.0)

    def test_energy_coefficients_config_from_dict_partial(self):
        """Test from_dict with partial data."""
        partial_dict = {'flops_pj': 5.0, 'baseline_w': 3.0}
        config = EnergyCoefficientsConfig.from_dict(partial_dict)

        self.assertEqual(config.flops_pj, 5.0)
        self.assertEqual(config.baseline_w, 3.0)
        # Other values should be defaults
        self.assertEqual(config.sram_pj_per_byte, 0.1)
        self.assertEqual(config.dram_pj_per_byte, 20.0)


class TestScalingConfig(unittest.TestCase):
    """Test ScalingConfig class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ScalingConfig(
            parameter_range=(1, 100),
            n_points=20,
            scaling_type="power_law",
            confidence_level=0.95,
            bootstrap_samples=1000
        )

    def test_scaling_config_creation(self):
        """Test basic ScalingConfig creation."""
        config = ScalingConfig(parameter_range=(1, 10))
        self.assertEqual(config.parameter_range, (1, 10))
        self.assertEqual(config.n_points, 10)  # default
        self.assertEqual(config.scaling_type, "power_law")  # default

    def test_scaling_config_custom_values(self):
        """Test ScalingConfig with custom values."""
        self.assertEqual(self.config.parameter_range, (1, 100))
        self.assertEqual(self.config.n_points, 20)
        self.assertEqual(self.config.scaling_type, "power_law")
        self.assertEqual(self.config.confidence_level, 0.95)
        self.assertEqual(self.config.bootstrap_samples, 1000)

    def test_scaling_config_generate_parameter_values(self):
        """Test parameter value generation."""
        params = self.config.generate_parameter_values()

        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 20)  # n_points
        self.assertEqual(params[0], 1)    # start of range
        self.assertEqual(params[-1], 100) # end of range

    def test_scaling_config_generate_parameter_values_logarithmic(self):
        """Test logarithmic parameter generation."""
        log_config = ScalingConfig(
            parameter_range=(1, 1000),
            n_points=5,
            scaling_type="logarithmic"
        )
        params = log_config.generate_parameter_values()

        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 5)
        # Should be logarithmically spaced
        self.assertAlmostEqual(params[0], 1.0, places=1)

    def test_scaling_config_to_dict(self):
        """Test dictionary serialization."""
        config_dict = self.config.to_dict()

        expected_keys = ['parameter_range', 'n_points', 'scaling_type',
                        'confidence_level', 'bootstrap_samples']
        for key in expected_keys:
            self.assertIn(key, config_dict)

    def test_scaling_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'parameter_range': [10, 1000],
            'n_points': 50,
            'scaling_type': 'exponential',
            'confidence_level': 0.99,
            'bootstrap_samples': 5000
        }

        config = ScalingConfig.from_dict(config_dict)

        self.assertEqual(config.parameter_range, (10, 1000))
        self.assertEqual(config.n_points, 50)
        self.assertEqual(config.scaling_type, 'exponential')
        self.assertEqual(config.confidence_level, 0.99)
        self.assertEqual(config.bootstrap_samples, 5000)


class TestMeterConfig(unittest.TestCase):
    """Unsupported acquisition promises and invalid device selection fail closed."""

    def test_unimplemented_controls_are_not_silently_accepted(self):
        controls = {"baseline_measurement": True, "log_measurements": True,
                    "calibration_file": "not-a-calibration.json", "sampling_rate_hz": 1000,
                    "measurement_duration_s": 5, "channels": ["cpu"]}
        for key, value in controls.items():
            with self.subTest(control=key), self.assertRaises(ValueError):
                MeterConfig.from_dict({"meter_type": "null", key: value})

    def test_invalid_device_and_inapplicable_options_are_refused(self):
        for config in ({"meter_type": "hardware"}, {"device_index": -1}, {"device_index": True},
                       {"device_index": "1"}, {"meter_type": "rapl", "device_index": 1},
                       {"meter_type": "null", "energy_path": "/unused"},
                       {"meter_type": "rapl", "energy_path": ""},
                       {"meter_type": "null", "type": "rapl"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MeterConfig.from_dict(config)




class TestExperimentManifest(unittest.TestCase):
    """Test ExperimentManifest class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.workload_configs = [
            WorkloadConfig(name="body", params={'J': 6}),
            WorkloadConfig(name="brain", params={'K': 100})
        ]

        self.energy_config = EnergyCoefficientsConfig(flops_pj=2.0)
        self.scaling_config = ScalingConfig(parameter_range=(1, 10), n_points=5)
        self.meter_config = MeterConfig(meter_type="null")

        self.manifest = ExperimentManifest(
            experiment_name="test_experiment",
            description="Test experiment for energy analysis",
            workload_configs=self.workload_configs,
            energy_coefficients=self.energy_config,
            scaling_config=self.scaling_config,
            meter_config=self.meter_config,
            output_directory="./results",
            random_seed=42
        )

    def test_experiment_manifest_creation(self):
        """Test basic ExperimentManifest creation."""
        manifest = ExperimentManifest(
            experiment_name="simple_experiment",
            workload_configs=[]
        )

        self.assertEqual(manifest.experiment_name, "simple_experiment")
        self.assertIsInstance(manifest.workload_configs, list)

    def test_experiment_manifest_full_config(self):
        """Test ExperimentManifest with full configuration."""
        self.assertEqual(self.manifest.experiment_name, "test_experiment")
        self.assertEqual(self.manifest.description, "Test experiment for energy analysis")
        self.assertEqual(len(self.manifest.workload_configs), 2)
        self.assertEqual(self.manifest.output_directory, "./results")
        self.assertEqual(self.manifest.random_seed, 42)

    def test_experiment_manifest_validate(self):
        """Test manifest validation."""
        is_valid, errors = self.manifest.validate()

        self.assertTrue(is_valid)
        self.assertIsInstance(errors, list)
        self.assertEqual(len(errors), 0)

    def test_experiment_manifest_validate_invalid(self):
        """Test validation of invalid manifest."""
        invalid_manifest = ExperimentManifest(
            experiment_name="",  # Empty name should be invalid
            workload_configs=[]
        )

        is_valid, errors = invalid_manifest.validate()

        self.assertFalse(is_valid)
        self.assertIsInstance(errors, list)
        self.assertGreater(len(errors), 0)

    @unittest.skipUnless(HAS_YAML, "PyYAML not available")
    def test_experiment_manifest_save_load_yaml(self):
        """Test YAML save and load functionality."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Save to YAML
            self.manifest.save_to_yaml(temp_path)

            # Load from YAML
            loaded_manifest = ExperimentManifest.load_from_yaml(temp_path)

            # Compare key attributes
            self.assertEqual(loaded_manifest.experiment_name, self.manifest.experiment_name)
            self.assertEqual(loaded_manifest.description, self.manifest.description)
            self.assertEqual(len(loaded_manifest.workload_configs), len(self.manifest.workload_configs))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_experiment_manifest_to_dict(self):
        """Test dictionary serialization."""
        manifest_dict = self.manifest.to_dict()

        expected_keys = ['experiment_name', 'description', 'workload_configs',
                        'energy_coefficients', 'scaling_config', 'meter_config',
                        'output_directory', 'random_seed']
        for key in expected_keys:
            self.assertIn(key, manifest_dict)


    def test_experiment_manifest_create_from_workloads(self):
        """Test creation from workload list."""
        workloads = [
            {"name": "workload1", "params": {"param1": 1}},
            {"name": "workload2", "params": {"param2": 2}}
        ]

        manifest = ExperimentManifest.create_from_workloads(
            "batch_experiment",
            workloads,
            description="Batch experiment"
        )

        self.assertEqual(manifest.experiment_name, "batch_experiment")
        self.assertEqual(len(manifest.workload_configs), 2)
        self.assertEqual(manifest.workload_configs[0].name, "workload1")


class TestExperimentConfigIntegration(unittest.TestCase):
    """Test integration between experiment configuration components."""




    def test_configuration_error_handling(self):
        """Test error handling in configuration classes."""
        # Test ExperimentManifest with invalid data
        invalid_manifest = ExperimentManifest(
            experiment_name="",  # Empty name
            workload_configs=[]  # No workloads
        )

        is_valid, errors = invalid_manifest.validate()
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)



if __name__ == '__main__':
    unittest.main()
