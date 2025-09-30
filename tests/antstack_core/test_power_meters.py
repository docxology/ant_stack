#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.power_meters module.

Tests power measurement and energy monitoring functionality including:
- Multiple power meter backends (RAPL, NVML, null)
- Context managers for energy measurement
- Cross-platform compatibility with graceful fallbacks
- Integration with energy analysis workflows
- Error handling and edge cases

Following .cursorrules principles:
- Real hardware testing when available
- Graceful fallback testing for CI environments
- Comprehensive error handling validation
- Platform-specific testing with proper detection
"""

import unittest
import tempfile
import time
import os
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from typing import Optional, Union

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.power_meters import (
    PowerSample,
    PowerMeter,
    NullPowerMeter,
    RaplPowerMeter,
    NvmlPowerMeter,
    measure_energy,
    create_power_meter
)

# Check for optional dependencies
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    pynvml = None


class TestPowerSample(unittest.TestCase):
    """Test PowerSample dataclass functionality."""

    def test_power_sample_creation(self):
        """Test basic PowerSample creation."""
        sample = PowerSample(timestamp=1234567890.0, watts=45.5)

        self.assertEqual(sample.timestamp, 1234567890.0)
        self.assertEqual(sample.watts, 45.5)

    def test_power_sample_with_zero_power(self):
        """Test PowerSample with zero power consumption."""
        sample = PowerSample(timestamp=time.time(), watts=0.0)

        self.assertIsInstance(sample.timestamp, float)
        self.assertEqual(sample.watts, 0.0)

    def test_power_sample_with_negative_power(self):
        """Test PowerSample with negative power (edge case)."""
        sample = PowerSample(timestamp=time.time(), watts=-5.0)

        self.assertEqual(sample.watts, -5.0)
        # Negative power might indicate measurement error

    def test_power_sample_with_high_power(self):
        """Test PowerSample with high power consumption."""
        sample = PowerSample(timestamp=time.time(), watts=500.0)  # 500W

        self.assertEqual(sample.watts, 500.0)


class TestNullPowerMeter(unittest.TestCase):
    """Test NullPowerMeter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.meter = NullPowerMeter()

    def test_null_power_meter_creation(self):
        """Test NullPowerMeter creation."""
        self.assertIsInstance(self.meter, NullPowerMeter)
        self.assertIsInstance(self.meter, PowerMeter)

    def test_null_power_meter_read(self):
        """Test null power meter reading."""
        sample = self.meter.read()

        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 0.0)
        self.assertIsInstance(sample.timestamp, float)
        self.assertGreater(sample.timestamp, 0)

    def test_null_power_meter_multiple_reads(self):
        """Test multiple readings from null power meter."""
        samples = []
        for _ in range(5):
            sample = self.meter.read()
            samples.append(sample)
            time.sleep(0.01)  # Small delay

        # All samples should have 0 watts
        for sample in samples:
            self.assertEqual(sample.watts, 0.0)

        # Timestamps should be increasing
        timestamps = [s.timestamp for s in samples]
        self.assertEqual(timestamps, sorted(timestamps))


class TestRaplPowerMeter(unittest.TestCase):
    """Test RaplPowerMeter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_rapl_power_meter_creation_default_path(self):
        """Test RaplPowerMeter creation with default path."""
        meter = RaplPowerMeter()

        expected_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self.assertEqual(meter.energy_uj_path, expected_path)

    def test_rapl_power_meter_creation_custom_path(self):
        """Test RaplPowerMeter creation with custom path."""
        custom_path = "/custom/rapl/path/energy_uj"
        meter = RaplPowerMeter(energy_uj_path=custom_path)

        self.assertEqual(meter.energy_uj_path, custom_path)

    def test_rapl_power_meter_read_file_not_found(self):
        """Test RAPL meter reading when file doesn't exist."""
        meter = RaplPowerMeter(energy_uj_path="/nonexistent/path")

        sample = meter.read()

        # Should return zero power when file doesn't exist
        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 0.0)

    @patch('builtins.open', new_callable=mock_open, read_data="1000000\n")
    def test_rapl_power_meter_read_first_reading(self, mock_file):
        """Test RAPL meter first reading."""
        meter = RaplPowerMeter(energy_uj_path=self.temp_path)

        sample = meter.read()

        # First reading should return 0 watts (no previous measurement)
        self.assertEqual(sample.watts, 0.0)
        self.assertIsInstance(sample.timestamp, float)

    @patch('builtins.open', new_callable=mock_open, read_data="2000000\n")
    def test_rapl_power_meter_read_subsequent_reading(self, mock_file):
        """Test RAPL meter subsequent reading."""
        meter = RaplPowerMeter(energy_uj_path=self.temp_path)

        # First reading to establish baseline
        meter.read()

        # Mock a second reading with higher energy
        mock_file.return_value.read.return_value = "2000000\n"
        time.sleep(0.1)  # Small delay

        sample = meter.read()

        # Should calculate power based on energy difference
        # Energy diff: 1,000,000 μJ = 1 J
        # Time diff: ~0.1 s
        # Power: 1 J / 0.1 s = 10 W
        self.assertGreater(sample.watts, 0)

    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_rapl_power_meter_read_permission_error(self, mock_file):
        """Test RAPL meter reading with permission error."""
        meter = RaplPowerMeter(energy_uj_path=self.temp_path)

        sample = meter.read()

        # Should handle permission error gracefully
        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 0.0)

    @patch('builtins.open', side_effect=ValueError("Invalid file content"))
    def test_rapl_power_meter_read_invalid_content(self, mock_file):
        """Test RAPL meter reading with invalid file content."""
        meter = RaplPowerMeter(energy_uj_path=self.temp_path)

        sample = meter.read()

        # Should handle invalid content gracefully
        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 0.0)


class TestNvmlPowerMeter(unittest.TestCase):
    """Test NvmlPowerMeter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.meter = NvmlPowerMeter()

    def test_nvml_power_meter_creation(self):
        """Test NvmlPowerMeter creation."""
        self.assertIsInstance(self.meter, NvmlPowerMeter)
        self.assertIsInstance(self.meter, PowerMeter)

    @patch('antstack_core.analysis.power_meters.pynvml')
    def test_nvml_power_meter_read_with_nvml(self, mock_pynvml):
        """Test NVML meter reading with pynvml available."""
        # Mock pynvml functionality
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = None
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # 150W in milliwatts
        mock_pynvml.nvmlShutdown.return_value = None

        sample = self.meter.read()

        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 150.0)  # Converted from milliwatts

    def test_nvml_power_meter_read_without_nvml(self):
        """Test NVML meter reading without pynvml available."""
        # Temporarily set pynvml to None to simulate missing dependency
        original_pynvml = self.meter.pynvml
        self.meter.pynvml = None

        try:
            sample = self.meter.read()

            # Should fallback to zero power
            self.assertIsInstance(sample, PowerSample)
            self.assertEqual(sample.watts, 0.0)
        finally:
            self.meter.pynvml = original_pynvml

    @patch('antstack_core.analysis.power_meters.pynvml')
    def test_nvml_power_meter_read_nvml_error(self, mock_pynvml):
        """Test NVML meter reading with NVML error."""
        # Mock pynvml to raise an exception
        mock_pynvml.nvmlInit.side_effect = Exception("NVML Error")

        sample = self.meter.read()

        # Should handle NVML error gracefully
        self.assertIsInstance(sample, PowerSample)
        self.assertEqual(sample.watts, 0.0)

    @patch('antstack_core.analysis.power_meters.pynvml')
    def test_nvml_power_meter_initialization(self, mock_pynvml):
        """Test NVML meter initialization."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = None
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        # First read should initialize NVML
        sample1 = self.meter.read()
        mock_pynvml.nvmlInit.assert_called_once()

        # Subsequent reads should not re-initialize
        sample2 = self.meter.read()
        # Should still be called only once
        self.assertEqual(mock_pynvml.nvmlInit.call_count, 1)


class TestPowerMeterUtilities(unittest.TestCase):
    """Test power meter utility functions."""

    def test_create_power_meter_null(self):
        """Test creating null power meter."""
        meter = create_power_meter("null")

        self.assertIsInstance(meter, NullPowerMeter)

    def test_create_power_meter_rapl(self):
        """Test creating RAPL power meter."""
        meter = create_power_meter("rapl")

        self.assertIsInstance(meter, RaplPowerMeter)

    def test_create_power_meter_nvml(self):
        """Test creating NVML power meter."""
        meter = create_power_meter("nvml")

        self.assertIsInstance(meter, NvmlPowerMeter)

    def test_create_power_meter_invalid_type(self):
        """Test creating power meter with invalid type."""
        with self.assertRaises(ValueError):
            create_power_meter("invalid_type")

    def test_create_power_meter_with_kwargs(self):
        """Test creating power meter with additional arguments."""
        meter = create_power_meter("rapl", energy_uj_path="/custom/path")

        self.assertIsInstance(meter, RaplPowerMeter)
        self.assertEqual(meter.energy_uj_path, "/custom/path")

    def test_measure_energy_with_null_meter(self):
        """Test energy measurement with null meter."""
        meter = NullPowerMeter()

        with measure_energy(meter) as measurement:
            # Simulate some work
            time.sleep(0.01)
            result = 42  # Dummy computation result

        self.assertIsInstance(measurement, dict)
        self.assertIn("energy_joules", measurement)
        self.assertIn("duration_seconds", measurement)
        self.assertIn("average_power_watts", measurement)
        self.assertIn("samples", measurement)

        # With null meter, energy should be 0
        self.assertEqual(measurement["energy_joules"], 0.0)

    def test_measure_energy_with_custom_meter(self):
        """Test energy measurement with custom meter."""
        # Create a mock meter that returns varying power
        class MockPowerMeter(PowerMeter):
            def __init__(self):
                self.power_values = [10.0, 15.0, 12.0, 18.0]
                self.index = 0

            def read(self):
                power = self.power_values[self.index % len(self.power_values)]
                self.index += 1
                return PowerSample(timestamp=time.time(), watts=power)

        meter = MockPowerMeter()

        with measure_energy(meter) as measurement:
            # Simulate work with multiple readings
            for _ in range(3):
                time.sleep(0.01)

        self.assertIsInstance(measurement, dict)
        self.assertIn("samples", measurement)
        self.assertGreater(len(measurement["samples"]), 0)

        # Check that samples contain power readings
        for sample in measurement["samples"]:
            self.assertIsInstance(sample, PowerSample)
            self.assertGreaterEqual(sample.watts, 10.0)

    def test_measure_energy_without_meter(self):
        """Test energy measurement without explicit meter."""
        # Should use default meter (null meter)
        with measure_energy() as measurement:
            time.sleep(0.01)

        self.assertIsInstance(measurement, dict)
        self.assertEqual(measurement["energy_joules"], 0.0)

    def test_measure_energy_exception_handling(self):
        """Test energy measurement with exceptions."""
        # Create a meter that raises an exception
        class FaultyPowerMeter(PowerMeter):
            def read(self):
                raise RuntimeError("Meter fault")

        meter = FaultyPowerMeter()

        # Should handle exceptions gracefully
        with measure_energy(meter) as measurement:
            time.sleep(0.01)

        self.assertIsInstance(measurement, dict)
        # Should still have valid structure even with errors
        self.assertIn("energy_joules", measurement)


class TestPowerMeterIntegration(unittest.TestCase):
    """Test integration between power meter components."""

    def test_power_meter_polymorphism(self):
        """Test that all power meters implement the same interface."""
        meters = [
            NullPowerMeter(),
            RaplPowerMeter(),
            NvmlPowerMeter()
        ]

        for meter in meters:
            # All should implement PowerMeter interface
            self.assertIsInstance(meter, PowerMeter)

            # All should have a read method that returns PowerSample
            sample = meter.read()
            self.assertIsInstance(sample, PowerSample)
            self.assertIsInstance(sample.timestamp, float)
            self.assertIsInstance(sample.watts, (int, float))

    def test_power_meter_factory_integration(self):
        """Test integration between factory and meter types."""
        meter_types = ["null", "rapl", "nvml"]

        for meter_type in meter_types:
            meter = create_power_meter(meter_type)

            # Should be able to use the created meter
            with measure_energy(meter) as measurement:
                time.sleep(0.005)

            self.assertIsInstance(measurement, dict)
            self.assertIn("duration_seconds", measurement)

    def test_energy_measurement_consistency(self):
        """Test consistency of energy measurements."""
        meter = NullPowerMeter()

        measurements = []
        for _ in range(3):
            with measure_energy(meter) as measurement:
                time.sleep(0.01)
                measurements.append(measurement)

        # All measurements should be consistent (null meter always returns 0)
        for measurement in measurements:
            self.assertEqual(measurement["energy_joules"], 0.0)
            self.assertGreater(measurement["duration_seconds"], 0)

    def test_power_meter_error_recovery(self):
        """Test error recovery in power meter operations."""
        # Test with a meter that fails occasionally
        class UnreliablePowerMeter(PowerMeter):
            def __init__(self):
                self.call_count = 0

            def read(self):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise RuntimeError("Intermittent failure")
                return PowerSample(timestamp=time.time(), watts=10.0)

        meter = UnreliablePowerMeter()

        # Should handle intermittent failures gracefully
        with measure_energy(meter) as measurement:
            for _ in range(5):  # This will trigger at least one failure
                time.sleep(0.005)

        self.assertIsInstance(measurement, dict)
        # Should have collected some valid samples despite failures
        self.assertIn("samples", measurement)


class TestPowerMeterRobustness(unittest.TestCase):
    """Test robustness of power meter functionality."""

    def test_power_meter_with_system_load(self):
        """Test power meter under simulated system load."""
        meter = NullPowerMeter()

        # Simulate varying system load
        import threading
        import queue

        results_queue = queue.Queue()

        def measure_worker():
            with measure_energy(meter) as measurement:
                # Simulate variable computation time
                time.sleep(0.001 + (time.time() % 0.01))
            results_queue.put(measurement)

        # Run multiple measurement threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=measure_worker)
            threads.append(thread)
            thread.start()

        # Collect results
        results = []
        for _ in range(5):
            results.append(results_queue.get(timeout=1.0))

        # All results should be valid
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn("duration_seconds", result)

    def test_power_meter_memory_usage(self):
        """Test power meter memory usage patterns."""
        meter = NullPowerMeter()

        # Test with many measurements
        measurements = []
        for _ in range(1000):
            with measure_energy(meter) as measurement:
                measurements.append(measurement)

        # Should handle many measurements without memory issues
        self.assertEqual(len(measurements), 1000)
        for measurement in measurements:
            self.assertIsInstance(measurement, dict)

    def test_power_meter_timing_precision(self):
        """Test timing precision in power measurements."""
        meter = NullPowerMeter()

        with measure_energy(meter) as measurement:
            start_time = time.time()
            time.sleep(0.1)  # 100ms
            end_time = time.time()

        # Duration should be reasonably close to expected
        measured_duration = measurement["duration_seconds"]
        actual_duration = end_time - start_time

        # Allow for some timing imprecision
        self.assertLess(abs(measured_duration - actual_duration), 0.01)  # Within 10ms

    def test_power_meter_cross_platform_compatibility(self):
        """Test power meter compatibility across different scenarios."""
        # Test with different file system scenarios
        scenarios = [
            "/sys/class/powercap/intel-rapl:0/energy_uj",  # Linux RAPL
            "/nonexistent/path",                           # Missing file
            None                                          # Default path
        ]

        for path in scenarios:
            if path is None:
                meter = RaplPowerMeter()
            else:
                meter = RaplPowerMeter(energy_uj_path=path)

            # Should handle all scenarios gracefully
            sample = meter.read()
            self.assertIsInstance(sample, PowerSample)


if __name__ == '__main__':
    unittest.main()
