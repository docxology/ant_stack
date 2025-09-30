#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.statistics module.

Tests statistical methods including bootstrap analysis, scaling relationships,
energy efficiency metrics, and theoretical limit calculations.

Following .cursorrules principles:
- Real statistical validation of scientific methods
- Bootstrap confidence intervals and significance testing
- Comprehensive edge case testing
- Professional documentation with statistical references
"""

import unittest
import math
import random
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.statistics import (
    bootstrap_mean_ci, analyze_scaling_relationship,
    calculate_energy_efficiency_metrics, estimate_theoretical_limits
)


class TestBootstrapAnalysis(unittest.TestCase):
    """Test bootstrap statistical methods."""

    def setUp(self):
        """Set up test fixtures with reproducible random seed."""
        random.seed(42)
        self.test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.expected_mean = 3.0

    def test_bootstrap_mean_ci_basic(self):
        """Test basic bootstrap confidence interval calculation."""
        mean, lower, upper = bootstrap_mean_ci(self.test_data, seed=42)

        self.assertAlmostEqual(mean, self.expected_mean, places=1)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        self.assertLessEqual(lower, upper)

    def test_bootstrap_mean_ci_custom_alpha(self):
        """Test bootstrap with custom confidence level."""
        mean1, lower1, upper1 = bootstrap_mean_ci(self.test_data, alpha=0.05, seed=42)
        mean2, lower2, upper2 = bootstrap_mean_ci(self.test_data, alpha=0.10, seed=42)

        # 90% CI should be narrower than 95% CI
        self.assertLess(upper2 - lower2, upper1 - lower1)

    def test_bootstrap_mean_ci_custom_samples(self):
        """Test bootstrap with custom number of samples."""
        mean1, _, _ = bootstrap_mean_ci(self.test_data, num_samples=100, seed=42)
        mean2, _, _ = bootstrap_mean_ci(self.test_data, num_samples=1000, seed=42)

        # Should be close but not necessarily identical due to randomness
        self.assertAlmostEqual(mean1, mean2, places=1)

    def test_bootstrap_mean_ci_empty_data(self):
        """Test bootstrap with empty data raises error."""
        with self.assertRaises(ValueError):
            bootstrap_mean_ci([])

    def test_bootstrap_mean_ci_single_value(self):
        """Test bootstrap with single value."""
        mean, lower, upper = bootstrap_mean_ci([5.0])
        self.assertEqual(mean, 5.0)
        self.assertEqual(lower, 5.0)
        self.assertEqual(upper, 5.0)

    def test_bootstrap_mean_ci_reproducibility(self):
        """Test bootstrap reproducibility with fixed seed."""
        result1 = bootstrap_mean_ci(self.test_data, seed=123)
        result2 = bootstrap_mean_ci(self.test_data, seed=123)

        self.assertEqual(result1, result2)


class TestScalingAnalysis(unittest.TestCase):
    """Test scaling relationship analysis."""

    def setUp(self):
        """Set up test fixtures for scaling analysis."""
        # Perfect linear relationship: y = 2x + 1
        self.x_linear = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_linear = [3.0, 5.0, 7.0, 9.0, 11.0]  # 2x + 1

        # Perfect quadratic relationship: y = x^2
        self.x_quadratic = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_quadratic = [1.0, 4.0, 9.0, 16.0, 25.0]

        # Noisy linear relationship
        self.x_noisy = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_noisy = [2.8, 5.2, 6.9, 9.1, 10.8]  # y = 2x + 1 ± 0.2

    def test_analyze_scaling_linear_perfect(self):
        """Test scaling analysis on linear relationship in log space."""
        # Use data that follows a true power law: y = 2 * x
        x_power = [1.0, 2.0, 4.0, 8.0, 16.0]
        y_power = [2.0, 4.0, 8.0, 16.0, 32.0]  # y = 2 * x

        result = analyze_scaling_relationship(x_power, y_power)

        self.assertIn("scaling_exponent", result)
        self.assertIn("intercept", result)
        self.assertIn("r_squared", result)
        self.assertIn("scaling_regime", result)

        self.assertAlmostEqual(result["scaling_exponent"], 1.0, places=2)
        self.assertAlmostEqual(result["intercept"], 2.0, places=1)
        self.assertGreater(result["r_squared"], 0.99)
        self.assertEqual(result["scaling_regime"], "linear")

    def test_analyze_scaling_quadratic(self):
        """Test scaling analysis on quadratic relationship."""
        result = analyze_scaling_relationship(self.x_quadratic, self.y_quadratic)

        self.assertAlmostEqual(result["scaling_exponent"], 2.0, places=2)
        self.assertEqual(result["scaling_regime"], "quadratic")

    def test_analyze_scaling_noisy_linear(self):
        """Test scaling analysis on noisy linear relationship."""
        result = analyze_scaling_relationship(self.x_noisy, self.y_noisy)

        # Should be approximately linear (close to 1.0) with high R²
        self.assertAlmostEqual(result["scaling_exponent"], 1.0, delta=0.2)
        self.assertGreater(result["r_squared"], 0.99)

    def test_analyze_scaling_insufficient_data(self):
        """Test scaling analysis with insufficient data."""
        result = analyze_scaling_relationship([1.0, 2.0], [1.0, 2.0])

        self.assertIn("error", result)
        self.assertIn("Insufficient data", result["error"])

    def test_analyze_scaling_mismatched_lengths(self):
        """Test scaling analysis with mismatched data lengths."""
        result = analyze_scaling_relationship([1.0, 2.0, 3.0], [1.0, 2.0])

        self.assertIn("error", result)
        self.assertIn("Mismatched data lengths", result["error"])

    def test_analyze_scaling_zero_values(self):
        """Test scaling analysis with zero values (log issues)."""
        result = analyze_scaling_relationship([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        self.assertIn("error", result)
        self.assertIn("non-positive values", result["error"])

    def test_analyze_scaling_negative_values(self):
        """Test scaling analysis with negative values."""
        result = analyze_scaling_relationship([-1.0, 0.0, 1.0], [1.0, 1.0, 1.0])

        self.assertIn("error", result)
        self.assertIn("non-positive values", result["error"])

    def test_analyze_scaling_edge_case_single_point(self):
        """Test scaling analysis with single point."""
        result = analyze_scaling_relationship([1.0], [1.0])

        self.assertIn("error", result)
        self.assertIn("Insufficient data", result["error"])


class TestEnergyEfficiency(unittest.TestCase):
    """Test energy efficiency metrics calculation."""

    def setUp(self):
        """Set up test fixtures for efficiency analysis."""
        self.energy_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Joules
        self.performance_values = [10.0, 20.0, 30.0, 40.0, 50.0]  # Operations/s

    def test_calculate_energy_efficiency_basic(self):
        """Test basic energy efficiency calculation."""
        result = calculate_energy_efficiency_metrics(
            self.energy_values, self.performance_values
        )

        self.assertIn("average_energy_j", result)
        self.assertIn("average_performance", result)
        self.assertIn("performance_per_joule", result)
        self.assertIn("energy_per_performance", result)
        self.assertIn("energy_delay_product", result)
        self.assertIn("efficiency_score", result)

        # Check calculations
        expected_avg_energy = 3.0  # mean of [1,2,3,4,5]
        expected_avg_perf = 30.0   # mean of [10,20,30,40,50]
        expected_perf_per_joule = expected_avg_perf / expected_avg_energy

        self.assertEqual(result["average_energy_j"], expected_avg_energy)
        self.assertEqual(result["average_performance"], expected_avg_perf)
        self.assertEqual(result["performance_per_joule"], expected_perf_per_joule)

    def test_calculate_energy_efficiency_empty_data(self):
        """Test efficiency calculation with empty data."""
        result = calculate_energy_efficiency_metrics([], [])

        self.assertIn("error", result)
        self.assertIn("Empty input data", result["error"])

    def test_calculate_energy_efficiency_mismatched_lengths(self):
        """Test efficiency calculation with mismatched data lengths."""
        result = calculate_energy_efficiency_metrics([1.0, 2.0], [1.0])

        self.assertIn("error", result)
        self.assertIn("Mismatched data lengths", result["error"])

    def test_calculate_energy_efficiency_zero_energy(self):
        """Test efficiency calculation with zero energy values."""
        result = calculate_energy_efficiency_metrics([0.0, 0.0], [1.0, 2.0])

        # Should handle zero energy gracefully (returns 0 to avoid inf)
        self.assertEqual(result["performance_per_joule"], 0.0)
        self.assertEqual(result["energy_per_performance"], 0.0)

    def test_calculate_energy_efficiency_zero_performance(self):
        """Test efficiency calculation with zero performance values."""
        result = calculate_energy_efficiency_metrics([1.0, 2.0], [0.0, 0.0])

        self.assertEqual(result["performance_per_joule"], 0.0)
        self.assertEqual(result["energy_per_performance"], float('inf'))


class TestTheoreticalLimits(unittest.TestCase):
    """Test theoretical limit calculations."""

    def setUp(self):
        """Set up test fixtures for theoretical limit analysis."""
        self.basic_workload = {"flops": 1e9}
        self.temperature_k = 300.0

    def test_estimate_theoretical_limits_basic(self):
        """Test basic theoretical limit estimation."""
        limits = estimate_theoretical_limits(self.basic_workload, self.temperature_k)

        self.assertIn("landauer_limit_j", limits)
        self.assertIn("thermal_energy_j", limits)
        self.assertIn("total_theoretical_j", limits)

        # Landauer limit should be positive
        self.assertGreater(limits["landauer_limit_j"], 0)

        # Thermal energy should be small but positive
        self.assertGreater(limits["thermal_energy_j"], 0)
        self.assertLess(limits["thermal_energy_j"], 1e-20)  # Very small

    def test_estimate_theoretical_limits_with_erasure(self):
        """Test theoretical limits with explicit bit erasure."""
        workload = {"bits_erased": 1000.0}
        limits = estimate_theoretical_limits(workload, self.temperature_k)

        # Should calculate Landauer limit based on bits erased
        expected_landauer = (1000.0 * 1.380649e-23 * self.temperature_k *
                           math.log(2))
        self.assertAlmostEqual(limits["landauer_limit_j"], expected_landauer, places=10)

    def test_estimate_theoretical_limits_with_mechanical_work(self):
        """Test theoretical limits with mechanical work."""
        workload = {
            "bits_erased": 100.0,
            "mechanical_work_j": 0.001
        }
        limits = estimate_theoretical_limits(workload, self.temperature_k)

        # Total should include both Landauer limit and mechanical work
        expected_total = (limits["landauer_limit_j"] + 0.001)
        self.assertEqual(limits["total_theoretical_j"], expected_total)

    def test_estimate_theoretical_limits_with_temperature_range(self):
        """Test theoretical limits with hot/cold reservoirs."""
        workload = {
            "bits_erased": 10.0,
            "hot_temperature_k": 400.0,
            "mechanical_work_j": 0.001  # Required for Carnot efficiency
        }
        limits = estimate_theoretical_limits(workload, self.temperature_k)

        # Should include Carnot efficiency
        self.assertIn("carnot_efficiency", limits)
        self.assertGreater(limits["carnot_efficiency"], 0)
        self.assertLess(limits["carnot_efficiency"], 1)

    def test_estimate_theoretical_limits_empty_workload(self):
        """Test theoretical limits with empty workload."""
        limits = estimate_theoretical_limits({}, self.temperature_k)

        # Should still have thermal energy
        self.assertIn("thermal_energy_j", limits)
        self.assertEqual(limits["total_theoretical_j"], limits["thermal_energy_j"])

    def test_estimate_theoretical_limits_quantum_case(self):
        """Test theoretical limits with quantum gates."""
        workload = {"quantum_gates": 1000.0}
        limits = estimate_theoretical_limits(workload, self.temperature_k)

        # Should include quantum limit
        self.assertIn("quantum_limit_j", limits)
        self.assertGreater(limits["quantum_limit_j"], 0)


class TestStatisticalEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in statistical methods."""

    def test_bootstrap_with_extreme_values(self):
        """Test bootstrap with extreme values."""
        data = [1e-10, 1e10, 1.0, 1.0, 1.0]
        mean, lower, upper = bootstrap_mean_ci(data, seed=42)

        self.assertTrue(math.isfinite(mean))
        self.assertTrue(math.isfinite(lower))
        self.assertTrue(math.isfinite(upper))

    def test_scaling_with_large_exponents(self):
        """Test scaling analysis with large exponents."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 16.0, 81.0, 256.0, 625.0]  # x^4

        result = analyze_scaling_relationship(x, y)
        self.assertAlmostEqual(result["scaling_exponent"], 4.0, places=1)

    def test_efficiency_with_negative_values(self):
        """Test efficiency calculation with negative values."""
        # Should handle negative energy gracefully
        result = calculate_energy_efficiency_metrics([-1.0, 1.0], [1.0, 2.0])

        # Implementation should handle this case
        self.assertIn("average_energy_j", result)

    def test_theoretical_limits_with_invalid_temperature(self):
        """Test theoretical limits with invalid temperature."""
        workload = {"bits_erased": 1.0}

        # Very low temperature
        limits_cold = estimate_theoretical_limits(workload, 0.1)
        self.assertGreater(limits_cold["landauer_limit_j"], 0)

        # Very high temperature
        limits_hot = estimate_theoretical_limits(workload, 1000.0)
        self.assertGreater(limits_hot["landauer_limit_j"],
                          limits_cold["landauer_limit_j"])


class TestStatisticalReproducibility(unittest.TestCase):
    """Test statistical method reproducibility."""

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap results are reproducible with same seed."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        result1 = bootstrap_mean_ci(data, num_samples=100, seed=12345)
        result2 = bootstrap_mean_ci(data, num_samples=100, seed=12345)

        self.assertEqual(result1, result2)

    def test_scaling_reproducibility(self):
        """Test that scaling analysis is deterministic."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.1, 4.2, 6.0, 8.1, 10.2]  # Noisy y = 2x + 0.1

        result1 = analyze_scaling_relationship(x, y)
        result2 = analyze_scaling_relationship(x, y)

        self.assertEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
