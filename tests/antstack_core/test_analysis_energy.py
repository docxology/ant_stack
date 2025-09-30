#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.energy module.

Tests energy estimation, coefficient management, unit conversions, and
energy efficiency analysis with statistical validation.

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
import math
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.energy import (
    EnergyCoefficients, ComputeLoad, EnergyBreakdown,
    estimate_detailed_energy, pj_to_j, aj_to_j, j_to_mj, j_to_kj, j_to_wh, wh_to_j,
    w_to_mw, mw_to_w, s_to_ms, ms_to_s, integrate_power_to_energy,
    estimate_compute_energy, add_baseline_energy, cost_of_transport,
    calculate_landauer_limit, GRAVITY_M_S2
)


class TestEnergyCoefficients(unittest.TestCase):
    """Test EnergyCoefficients class functionality."""

    def test_default_initialization(self):
        """Test default coefficient initialization."""
        coeffs = EnergyCoefficients()
        self.assertEqual(coeffs.flops_pj, 1.0)
        self.assertEqual(coeffs.sram_pj_per_byte, 0.1)
        self.assertEqual(coeffs.dram_pj_per_byte, 20.0)
        self.assertEqual(coeffs.spike_aj, 1.0)
        self.assertEqual(coeffs.body_per_joint_w, 2.0)
        self.assertEqual(coeffs.body_sensor_w_per_channel, 0.005)
        self.assertEqual(coeffs.baseline_w, 0.5)  # Updated to match paper config

    def test_custom_initialization(self):
        """Test custom coefficient initialization."""
        coeffs = EnergyCoefficients(
            flops_pj=2.0,
            sram_pj_per_byte=0.2,
            dram_pj_per_byte=30.0,
            spike_aj=2.0,
            body_per_joint_w=3.0,
            body_sensor_w_per_channel=0.01,
            baseline_w=0.5
        )
        self.assertEqual(coeffs.flops_pj, 2.0)
        self.assertEqual(coeffs.baseline_w, 0.5)

    def test_technology_scaling(self):
        """Test technology node scaling."""
        coeffs = EnergyCoefficients()
        scaled = coeffs.scale_by_technology(14.0)  # 14nm node (larger than 7nm)

        # Should maintain physical parameters
        self.assertEqual(scaled.body_per_joint_w, coeffs.body_per_joint_w)
        self.assertEqual(scaled.baseline_w, coeffs.baseline_w)

        # Should scale computational parameters (larger node = higher energy)
        self.assertGreater(scaled.flops_pj, coeffs.flops_pj)

    def test_to_dict_conversion(self):
        """Test dictionary conversion."""
        coeffs = EnergyCoefficients()
        data = coeffs.to_dict()

        self.assertIsInstance(data, dict)
        self.assertIn('flops_pj', data)
        self.assertIn('baseline_w', data)
        self.assertEqual(data['flops_pj'], 1.0)

    def test_immutability(self):
        """Test that EnergyCoefficients is effectively immutable."""
        coeffs = EnergyCoefficients()
        # Should be frozen dataclass
        with self.assertRaises(AttributeError):
            coeffs.flops_pj = 2.0


class TestComputeLoad(unittest.TestCase):
    """Test ComputeLoad class functionality."""

    def test_default_initialization(self):
        """Test default ComputeLoad initialization."""
        load = ComputeLoad()
        self.assertEqual(load.flops, 0.0)
        self.assertEqual(load.sram_bytes, 0.0)
        self.assertEqual(load.dram_bytes, 0.0)
        self.assertEqual(load.spikes, 0.0)

    def test_custom_initialization(self):
        """Test custom ComputeLoad initialization."""
        load = ComputeLoad(
            flops=1e9,
            sram_bytes=1024,
            dram_bytes=2048,
            spikes=1000
        )
        self.assertEqual(load.flops, 1e9)
        self.assertEqual(load.sram_bytes, 1024)

    def test_scaling(self):
        """Test load scaling."""
        load = ComputeLoad(flops=100, sram_bytes=50, dram_bytes=25, spikes=10)
        scaled = load.scale(2.0)

        self.assertEqual(scaled.flops, 200)
        self.assertEqual(scaled.sram_bytes, 100)
        self.assertEqual(scaled.dram_bytes, 50)
        self.assertEqual(scaled.spikes, 20)

    def test_to_dict_conversion(self):
        """Test dictionary conversion."""
        load = ComputeLoad(flops=1000, sram_bytes=500)
        data = load.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data['flops'], 1000)
        self.assertEqual(data['sram_bytes'], 500)


class TestEnergyBreakdown(unittest.TestCase):
    """Test EnergyBreakdown class functionality."""

    def test_initialization(self):
        """Test EnergyBreakdown initialization."""
        breakdown = EnergyBreakdown(
            actuation=1.0,
            sensing=0.5,
            compute_flops=0.1,
            compute_memory=0.05,
            compute_spikes=0.01,
            baseline=0.2
        )

        self.assertEqual(breakdown.actuation, 1.0)
        self.assertAlmostEqual(breakdown.total_compute, 0.16, places=10)  # 0.1 + 0.05 + 0.01
        self.assertEqual(breakdown.total, 1.86)  # 1.0 + 0.5 + 0.16 + 0.2

    def test_compute_fraction(self):
        """Test compute fraction calculation."""
        breakdown = EnergyBreakdown(
            actuation=1.0,
            sensing=1.0,
            compute_flops=1.0,
            compute_memory=1.0,
            compute_spikes=1.0,
            baseline=1.0
        )

        # Total = 6.0, compute = 3.0, fraction = 0.5
        self.assertEqual(breakdown.compute_fraction, 0.5)

    def test_dominant_component(self):
        """Test dominant component identification."""
        breakdown = EnergyBreakdown(actuation=5.0, sensing=1.0)
        self.assertEqual(breakdown.dominant_component(), 'Actuation')

    def test_to_dict_conversion(self):
        """Test dictionary conversion for plotting."""
        breakdown = EnergyBreakdown(actuation=1.0, compute_flops=0.5)
        data = breakdown.to_dict()

        self.assertIn('Actuation', data)
        self.assertIn('Compute (FLOPs)', data)
        self.assertEqual(data['Actuation'], 1.0)


class TestEnergyEstimation(unittest.TestCase):
    """Test energy estimation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.coeffs = EnergyCoefficients()
        self.load = ComputeLoad(
            flops=1e9,  # 1 billion FLOPs
            sram_bytes=1024,
            dram_bytes=2048,
            spikes=1000
        )

    def test_estimate_detailed_energy_basic(self):
        """Test basic detailed energy estimation."""
        breakdown = estimate_detailed_energy(self.load, self.coeffs)

        self.assertIsInstance(breakdown, EnergyBreakdown)
        self.assertGreater(breakdown.compute_flops, 0)
        self.assertGreater(breakdown.compute_memory, 0)
        self.assertGreater(breakdown.compute_spikes, 0)

    def test_estimate_detailed_energy_with_actuation(self):
        """Test energy estimation with actuation and sensing."""
        breakdown = estimate_detailed_energy(
            self.load, self.coeffs,
            actuation_energy=1.0,
            sensing_energy=0.5
        )

        self.assertEqual(breakdown.actuation, 1.0)
        self.assertEqual(breakdown.sensing, 0.5)

    def test_estimate_detailed_energy_with_baseline(self):
        """Test energy estimation with baseline power."""
        coeffs_with_baseline = EnergyCoefficients(baseline_w=1.0)
        breakdown = estimate_detailed_energy(self.load, coeffs_with_baseline, duration_s=2.0)

        self.assertEqual(breakdown.baseline, 2.0)  # 1.0 W * 2.0 s

    def test_estimate_compute_energy(self):
        """Test compute energy estimation."""
        energy = estimate_compute_energy(self.load, self.coeffs)

        # Should be sum of all compute components
        expected = (
            pj_to_j(self.coeffs.flops_pj) * self.load.flops +
            pj_to_j(self.coeffs.sram_pj_per_byte) * self.load.sram_bytes +
            pj_to_j(self.coeffs.dram_pj_per_byte) * self.load.dram_bytes +
            aj_to_j(self.coeffs.spike_aj) * self.load.spikes
        )

        self.assertAlmostEqual(energy, expected, places=10)

    def test_add_baseline_energy(self):
        """Test baseline energy addition."""
        base_energy = 1.0
        duration = 5.0
        coeffs = EnergyCoefficients(baseline_w=0.5)

        total_energy = add_baseline_energy(base_energy, duration, coeffs)
        expected = base_energy + (0.5 * 5.0)  # 1.0 + 2.5 = 3.5

        self.assertEqual(total_energy, expected)

    def test_add_baseline_energy_zero_baseline(self):
        """Test baseline energy addition with zero baseline."""
        base_energy = 1.0
        coeffs = EnergyCoefficients(baseline_w=0.0)

        total_energy = add_baseline_energy(base_energy, 10.0, coeffs)
        self.assertEqual(total_energy, base_energy)


class TestUnitConversions(unittest.TestCase):
    """Test unit conversion functions."""

    def test_pj_to_j(self):
        """Test picojoule to joule conversion."""
        self.assertEqual(pj_to_j(1e12), 1.0)  # 1 PJ = 1 J
        self.assertEqual(pj_to_j(1e9), 0.001)  # 1 GJ = 0.001 J

    def test_aj_to_j(self):
        """Test attojoule to joule conversion."""
        self.assertEqual(aj_to_j(1e18), 1.0)  # 1 aJ = 1e-18 J
        self.assertEqual(aj_to_j(1e15), 1e-3)  # 1 fJ = 1e-15 J -> 1e-3 J

    def test_j_to_mj(self):
        """Test joule to millijoule conversion."""
        self.assertEqual(j_to_mj(1.0), 1000.0)
        self.assertEqual(j_to_mj(0.001), 1.0)

    def test_j_to_kj(self):
        """Test joule to kilojoule conversion."""
        self.assertEqual(j_to_kj(1000.0), 1.0)
        self.assertEqual(j_to_kj(1.0), 0.001)

    def test_j_to_wh(self):
        """Test joule to watt-hour conversion."""
        self.assertEqual(j_to_wh(3600.0), 1.0)  # 1 Wh = 3600 J
        self.assertEqual(j_to_wh(1800.0), 0.5)

    def test_wh_to_j(self):
        """Test watt-hour to joule conversion."""
        self.assertEqual(wh_to_j(1.0), 3600.0)
        self.assertEqual(wh_to_j(0.5), 1800.0)

    def test_w_to_mw(self):
        """Test watt to milliwatt conversion."""
        self.assertEqual(w_to_mw(1.0), 1000.0)
        self.assertEqual(w_to_mw(0.001), 1.0)

    def test_mw_to_w(self):
        """Test milliwatt to watt conversion."""
        self.assertEqual(mw_to_w(1000.0), 1.0)
        self.assertEqual(mw_to_w(1.0), 0.001)

    def test_s_to_ms(self):
        """Test second to millisecond conversion."""
        self.assertEqual(s_to_ms(1.0), 1000.0)
        self.assertEqual(s_to_ms(0.001), 1.0)

    def test_ms_to_s(self):
        """Test millisecond to second conversion."""
        self.assertEqual(ms_to_s(1000.0), 1.0)
        self.assertEqual(ms_to_s(1.0), 0.001)


class TestPowerIntegration(unittest.TestCase):
    """Test power integration functions."""

    def test_integrate_power_to_energy_simple(self):
        """Test simple trapezoidal power integration."""
        power_watts = [1.0, 2.0, 3.0]
        timestamps_s = [0.0, 1.0, 2.0]

        energy = integrate_power_to_energy(power_watts, timestamps_s)

        # Expected: (1+2)/2 * 1 + (2+3)/2 * 1 = 1.5 + 2.5 = 4.0
        self.assertEqual(energy, 4.0)

    def test_integrate_power_to_energy_edge_cases(self):
        """Test edge cases for power integration."""
        # Empty inputs
        self.assertEqual(integrate_power_to_energy([], []), 0.0)

        # Single point
        self.assertEqual(integrate_power_to_energy([1.0], [0.0]), 0.0)

        # Mismatched lengths
        self.assertEqual(integrate_power_to_energy([1.0], [0.0, 1.0]), 0.0)

        # Negative timestamps (should still work)
        energy = integrate_power_to_energy([1.0, 2.0], [1.0, 0.0])
        self.assertEqual(energy, 0.0)  # dt negative, skipped


class TestCostOfTransport(unittest.TestCase):
    """Test cost of transport calculations."""

    def test_cost_of_transport_basic(self):
        """Test basic cost of transport calculation."""
        energy_j = 10.0
        mass_kg = 1.0
        distance_m = 1.0

        cot = cost_of_transport(energy_j, mass_kg, distance_m)
        expected = 10.0 / (1.0 * GRAVITY_M_S2 * 1.0)  # 10 / (1 * 9.81 * 1)

        self.assertAlmostEqual(cot, expected, places=5)

    def test_cost_of_transport_edge_cases(self):
        """Test edge cases for cost of transport."""
        # Zero energy
        self.assertEqual(cost_of_transport(0.0, 1.0, 1.0), 0.0)

        # Zero mass
        self.assertEqual(cost_of_transport(1.0, 0.0, 1.0), 0.0)

        # Zero distance
        self.assertEqual(cost_of_transport(1.0, 1.0, 0.0), 0.0)

        # Negative values
        self.assertEqual(cost_of_transport(-1.0, 1.0, 1.0), 0.0)


class TestLandauerLimit(unittest.TestCase):
    """Test Landauer limit calculations."""

    def test_landauer_limit_basic(self):
        """Test basic Landauer limit calculation."""
        bits = 1.0
        temp_k = 300.0

        limit = calculate_landauer_limit(bits, temp_k)
        expected = bits * 1.380649e-23 * temp_k * math.log(2)

        self.assertAlmostEqual(limit, expected, places=10)

    def test_landauer_limit_room_temperature(self):
        """Test Landauer limit at room temperature."""
        bits = 1000.0
        limit = calculate_landauer_limit(bits)

        # Should be positive and proportional to bits
        self.assertGreater(limit, 0)
        self.assertAlmostEqual(limit / bits, calculate_landauer_limit(1.0), places=10)


class TestConstants(unittest.TestCase):
    """Test physical constants."""

    def test_gravity_constant(self):
        """Test gravity constant value."""
        self.assertAlmostEqual(GRAVITY_M_S2, 9.80665, places=5)


if __name__ == '__main__':
    unittest.main()
