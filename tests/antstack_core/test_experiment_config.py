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
    """Test MeterConfig class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MeterConfig(
            meter_type="software",
            sampling_rate_hz=1000,
            measurement_duration_s=10.0,
            channels=["cpu", "memory", "disk"],
            calibration_file="calibration.json",
            baseline_measurement=True
        )

    def test_meter_config_creation(self):
        """Test basic MeterConfig creation."""
        config = MeterConfig(meter_type="hardware")
        self.assertEqual(config.meter_type, "hardware")
        self.assertEqual(config.sampling_rate_hz, 1000)  # default
        self.assertEqual(config.measurement_duration_s, 5.0)  # default

    def test_meter_config_custom_values(self):
        """Test MeterConfig with custom values."""
        self.assertEqual(self.config.meter_type, "software")
        self.assertEqual(self.config.sampling_rate_hz, 1000)
        self.assertEqual(self.config.measurement_duration_s, 10.0)
        self.assertEqual(self.config.channels, ["cpu", "memory", "disk"])
        self.assertEqual(self.config.calibration_file, "calibration.json")
        self.assertTrue(self.config.baseline_measurement)

    def test_meter_config_calculate_buffer_size(self):
        """Test buffer size calculation."""
        buffer_size = self.config.calculate_buffer_size()

        expected_size = int(1000 * 10.0)  # sampling_rate * duration
        self.assertEqual(buffer_size, expected_size)

    def test_meter_config_validate_channels(self):
        """Test channel validation."""
        valid_channels = ["cpu", "memory", "network", "disk"]
        result = self.config.validate_channels(valid_channels)
        self.assertTrue(result)

        invalid_channels = ["invalid_channel"]
        result = self.config.validate_channels(invalid_channels)
        self.assertFalse(result)

    def test_meter_config_to_dict(self):
        """Test dictionary serialization."""
        config_dict = self.config.to_dict()

        expected_keys = ['meter_type', 'sampling_rate_hz', 'measurement_duration_s',
                        'channels', 'calibration_file', 'baseline_measurement']
        for key in expected_keys:
            self.assertIn(key, config_dict)

    def test_meter_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'meter_type': 'external',
            'sampling_rate_hz': 2000,
            'measurement_duration_s': 20.0,
            'channels': ['temperature', 'voltage'],
            'calibration_file': 'external_cal.json',
            'baseline_measurement': False
        }

        config = MeterConfig.from_dict(config_dict)

        self.assertEqual(config.meter_type, 'external')
        self.assertEqual(config.sampling_rate_hz, 2000)
        self.assertEqual(config.measurement_duration_s, 20.0)
        self.assertEqual(config.channels, ['temperature', 'voltage'])


class TestExperimentManifest(unittest.TestCase):
    """Test ExperimentManifest class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.workload_configs = [
            WorkloadConfig(name="body_workload", params={'joints': 6}),
            WorkloadConfig(name="brain_workload", params={'channels': 100})
        ]

        self.energy_config = EnergyCoefficientsConfig(flops_pj=2.0)
        self.scaling_config = ScalingConfig(parameter_range=(1, 10), n_points=5)
        self.meter_config = MeterConfig(meter_type="software")

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

    def test_experiment_manifest_from_dict(self):
        """Test creation from dictionary."""
        manifest_dict = {
            'experiment_name': 'dict_experiment',
            'description': 'Created from dict',
            'workload_configs': [WorkloadConfig(name="test").to_dict()],
            'energy_coefficients': EnergyCoefficientsConfig().to_dict(),
            'scaling_config': ScalingConfig(parameter_range=(1, 5)).to_dict(),
            'meter_config': MeterConfig(meter_type="hardware").to_dict(),
            'output_directory': './dict_results',
            'random_seed': 123
        }

        manifest = ExperimentManifest.from_dict(manifest_dict)

        self.assertEqual(manifest.experiment_name, 'dict_experiment')
        self.assertEqual(manifest.description, 'Created from dict')
        self.assertEqual(manifest.output_directory, './dict_results')
        self.assertEqual(manifest.random_seed, 123)

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

    def test_complete_experiment_setup(self):
        """Test complete experiment setup workflow."""
        # Create all configuration components
        workload = WorkloadConfig(
            name="comprehensive_test",
            duration_s=5.0,
            repeats=3,
            params={'joints': 8, 'channels': 200, 'horizon': 10},
            mode="closed_form"
        )

        energy_coeffs = EnergyCoefficientsConfig(
            flops_pj=1.5,
            baseline_w=0.8
        )

        scaling = ScalingConfig(
            parameter_range=(1, 50),
            n_points=10,
            scaling_type="power_law"
        )

        meter = MeterConfig(
            meter_type="software",
            sampling_rate_hz=2000,
            channels=["cpu", "memory", "power"]
        )

        # Create experiment manifest
        manifest = ExperimentManifest(
            experiment_name="integration_test",
            description="Complete integration test",
            workload_configs=[workload],
            energy_coefficients=energy_coeffs,
            scaling_config=scaling,
            meter_config=meter,
            output_directory="./integration_results",
            random_seed=42
        )

        # Validate the complete setup
        is_valid, errors = manifest.validate()

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Test serialization
        manifest_dict = manifest.to_dict()
        self.assertIsInstance(manifest_dict, dict)

        # Test round-trip
        restored_manifest = ExperimentManifest.from_dict(manifest_dict)
        self.assertEqual(restored_manifest.experiment_name, manifest.experiment_name)

    def test_configuration_parameter_ranges(self):
        """Test that configuration parameters handle valid ranges."""
        # Test WorkloadConfig parameter validation
        valid_workload = WorkloadConfig(
            name="valid",
            duration_s=1.0,  # Positive duration
            repeats=1,       # At least 1 repeat
            params={}
        )
        self.assertIsInstance(valid_workload, WorkloadConfig)

        # Test ScalingConfig parameter validation
        valid_scaling = ScalingConfig(
            parameter_range=(0.1, 100.0),  # Valid positive range
            n_points=5  # Reasonable number of points
        )
        self.assertIsInstance(valid_scaling, ScalingConfig)

        # Test MeterConfig parameter validation
        valid_meter = MeterConfig(
            meter_type="software",
            sampling_rate_hz=100,  # Reasonable sampling rate
            measurement_duration_s=1.0  # Reasonable duration
        )
        self.assertIsInstance(valid_meter, MeterConfig)

    def test_configuration_edge_cases(self):
        """Test configuration handling of edge cases."""
        # Test with minimal configurations
        minimal_workload = WorkloadConfig(name="minimal")
        self.assertEqual(minimal_workload.duration_s, 2.0)  # default
        self.assertEqual(minimal_workload.repeats, 5)      # default

        minimal_manifest = ExperimentManifest(
            experiment_name="minimal",
            workload_configs=[minimal_workload]
        )
        is_valid, errors = minimal_manifest.validate()
        self.assertTrue(is_valid)

        # Test with extreme values
        extreme_workload = WorkloadConfig(
            name="extreme",
            duration_s=0.001,  # Very short duration
            repeats=1000      # Many repeats
        )
        self.assertIsInstance(extreme_workload, WorkloadConfig)

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

        # Test WorkloadConfig with invalid data
        # These should still create valid objects (validation is separate)
        invalid_workload = WorkloadConfig(
            name="",
            duration_s=-1.0,  # Negative duration
            repeats=-1       # Negative repeats
        )
        self.assertIsInstance(invalid_workload, WorkloadConfig)


if __name__ == '__main__':
    unittest.main()
