#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.scaling_analysis module.

Tests scaling analysis functionality with comprehensive coverage including:
- Single parameter scaling analysis
- Multi-parameter scaling analysis
- Regime detection and classification
- Comparative analysis across multiple results
- Report generation
- Edge cases and error conditions

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.scaling_analysis import (
    ScalingResult,
    MultiParameterScaling,
    ScalingAnalyzer,
    HAS_NUMPY,
    HAS_SCIPY
)

try:
    import numpy as np
    HAS_NUMPY_REAL = True
except ImportError:
    HAS_NUMPY_REAL = False


class TestScalingResult(unittest.TestCase):
    """Test ScalingResult dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "parameter_name": "complexity",
            "parameter_values": [1.0, 2.0, 4.0, 8.0, 16.0],
            "response_values": [1.0, 1.5, 2.0, 2.5, 3.0],
            "scaling_exponent": 0.8,
            "intercept": 0.5,
            "r_squared": 0.95,
            "regime": "sub-linear",
            "confidence_interval": (0.7, 0.9),
            "p_value": 0.001,
            "valid": True
        }

    def test_scaling_result_creation(self):
        """Test basic ScalingResult creation."""
        result = ScalingResult(**self.sample_data)

        self.assertEqual(result.parameter_name, "complexity")
        self.assertEqual(len(result.parameter_values), 5)
        self.assertEqual(len(result.response_values), 5)
        self.assertEqual(result.scaling_exponent, 0.8)
        self.assertEqual(result.intercept, 0.5)
        self.assertEqual(result.r_squared, 0.95)
        self.assertEqual(result.regime, "sub-linear")
        self.assertEqual(result.confidence_interval, (0.7, 0.9))
        self.assertEqual(result.p_value, 0.001)
        self.assertTrue(result.valid)

    def test_scaling_result_defaults(self):
        """Test ScalingResult with minimal data."""
        minimal_data = {
            "parameter_name": "test",
            "parameter_values": [1.0, 2.0],
            "response_values": [1.0, 2.0]
        }

        result = ScalingResult(**minimal_data)
        self.assertEqual(result.parameter_name, "test")
        self.assertEqual(result.scaling_exponent, None)
        self.assertEqual(result.intercept, None)
        self.assertEqual(result.r_squared, None)
        self.assertEqual(result.regime, None)
        self.assertEqual(result.confidence_interval, None)
        self.assertEqual(result.p_value, None)
        self.assertFalse(result.valid)


class TestMultiParameterScaling(unittest.TestCase):
    """Test MultiParameterScaling dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.primary_param = "complexity"
        self.secondary_params = {
            "memory": [100, 200, 400],
            "bandwidth": [10, 20, 40]
        }

        # Create sample scaling results
        self.scaling_results = {
            "complexity": ScalingResult(
                parameter_name="complexity",
                parameter_values=[1, 2, 4],
                response_values=[1, 1.5, 2.0],
                scaling_exponent=0.8,
                valid=True
            ),
            "memory": ScalingResult(
                parameter_name="memory",
                parameter_values=[100, 200, 400],
                response_values=[1, 1.5, 2.0],
                scaling_exponent=0.5,
                valid=True
            )
        }

        self.interaction_effects = {"memory": 0.4, "bandwidth": 0.2}

    def test_multi_parameter_scaling_creation(self):
        """Test MultiParameterScaling creation."""
        multi_scaling = MultiParameterScaling(
            primary_parameter=self.primary_param,
            secondary_parameters=self.secondary_params,
            scaling_results=self.scaling_results,
            interaction_effects=self.interaction_effects
        )

        self.assertEqual(multi_scaling.primary_parameter, "complexity")
        self.assertEqual(len(multi_scaling.secondary_parameters), 2)
        self.assertEqual(len(multi_scaling.scaling_results), 2)
        self.assertEqual(multi_scaling.interaction_effects, self.interaction_effects)

    def test_multi_parameter_scaling_defaults(self):
        """Test MultiParameterScaling with defaults."""
        multi_scaling = MultiParameterScaling(
            primary_parameter="test",
            secondary_parameters={},
            scaling_results={}
        )

        self.assertEqual(multi_scaling.primary_parameter, "test")
        self.assertEqual(len(multi_scaling.secondary_parameters), 0)
        self.assertEqual(len(multi_scaling.scaling_results), 0)
        self.assertIsNone(multi_scaling.interaction_effects)


class TestScalingAnalyzer(unittest.TestCase):
    """Test ScalingAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ScalingAnalyzer()

        # Test data with known scaling relationships (all positive values)
        self.x_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        self.y_linear = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Linear scaling
        self.y_sublinear = [1.0, 1.8, 2.8, 3.9, 5.0, 6.2]  # Sub-linear scaling
        self.y_superlinear = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]  # Super-linear scaling

    def test_analyzer_initialization(self):
        """Test ScalingAnalyzer initialization."""
        analyzer = ScalingAnalyzer()
        self.assertIsInstance(analyzer, ScalingAnalyzer)

    def test_analyze_single_parameter_scaling_linear(self):
        """Test single parameter scaling analysis with linear relationship."""
        result = self.analyzer.analyze_single_parameter_scaling(
            self.x_values, self.y_linear, "complexity", "performance"
        )

        self.assertIsInstance(result, ScalingResult)
        self.assertEqual(result.parameter_name, "complexity")
        self.assertEqual(len(result.parameter_values), 6)
        self.assertEqual(len(result.response_values), 6)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.scaling_exponent)
        self.assertIsNotNone(result.r_squared)
        self.assertIsNotNone(result.regime)

        # Linear relationship should have exponent close to 1.0
        if result.scaling_exponent is not None:
            self.assertAlmostEqual(result.scaling_exponent, 1.0, delta=0.1)

    def test_analyze_single_parameter_scaling_insufficient_data(self):
        """Test single parameter scaling with insufficient data."""
        x_short = [1.0, 2.0]  # Only 2 points
        y_short = [1.0, 2.0]

        result = self.analyzer.analyze_single_parameter_scaling(
            x_short, y_short, "complexity", "performance"
        )

        self.assertIsInstance(result, ScalingResult)
        self.assertFalse(result.valid)
        self.assertIsNone(result.scaling_exponent)
        self.assertIsNone(result.intercept)
        self.assertIsNone(result.r_squared)
        self.assertIsNone(result.regime)

    def test_analyze_single_parameter_scaling_sublinear(self):
        """Test single parameter scaling analysis with sub-linear relationship."""
        result = self.analyzer.analyze_single_parameter_scaling(
            self.x_values, self.y_sublinear, "complexity", "performance"
        )

        self.assertIsInstance(result, ScalingResult)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.scaling_exponent)
        self.assertIsNotNone(result.regime)

        # Sub-linear relationship should have exponent between 0 and 1
        if result.scaling_exponent is not None:
            self.assertGreater(result.scaling_exponent, 0)
            self.assertLess(result.scaling_exponent, 1)

    def test_analyze_single_parameter_scaling_superlinear(self):
        """Test single parameter scaling analysis with super-linear relationship."""
        result = self.analyzer.analyze_single_parameter_scaling(
            self.x_values, self.y_superlinear, "complexity", "performance"
        )

        self.assertIsInstance(result, ScalingResult)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.scaling_exponent)
        self.assertIsNotNone(result.regime)

        # Super-linear relationship should have exponent greater than 1
        if result.scaling_exponent is not None:
            self.assertGreater(result.scaling_exponent, 1)

    def test_analyze_multi_parameter_scaling(self):
        """Test multi-parameter scaling analysis."""
        primary_values = [1, 2, 4, 8]
        secondary_params = {
            "memory": [100, 200, 400, 800],
            "bandwidth": [10, 20, 40, 80]
        }
        response_values = [1, 1.5, 2.0, 2.5]

        result = self.analyzer.analyze_multi_parameter_scaling(
            "complexity", primary_values, secondary_params, response_values
        )

        self.assertIsInstance(result, MultiParameterScaling)
        self.assertEqual(result.primary_parameter, "complexity")
        self.assertEqual(len(result.secondary_parameters), 2)
        self.assertEqual(len(result.scaling_results), 3)  # primary + 2 secondary

        # Check that all results are ScalingResult instances
        for scaling_result in result.scaling_results.values():
            self.assertIsInstance(scaling_result, ScalingResult)

    def test_analyze_multi_parameter_scaling_empty_secondary(self):
        """Test multi-parameter scaling with no secondary parameters."""
        primary_values = [1, 2, 4, 8]
        secondary_params = {}
        response_values = [1, 1.5, 2.0, 2.5]

        result = self.analyzer.analyze_multi_parameter_scaling(
            "complexity", primary_values, secondary_params, response_values
        )

        self.assertIsInstance(result, MultiParameterScaling)
        self.assertEqual(result.primary_parameter, "complexity")
        self.assertEqual(len(result.secondary_parameters), 0)
        self.assertEqual(len(result.scaling_results), 1)  # Only primary

    def test_detect_scaling_regime(self):
        """Test scaling regime detection."""
        # Test linear regime
        linear_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 2, 4],
            scaling_exponent=1.0,
            r_squared=0.95,
            valid=True
        )

        regime_info = self.analyzer.detect_scaling_regime(linear_result)

        self.assertIsInstance(regime_info, dict)
        self.assertIn("regime", regime_info)
        self.assertIn("confidence", regime_info)
        self.assertEqual(regime_info["regime"], "linear")

        # Test constant regime
        constant_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 1, 1],
            scaling_exponent=0.05,
            r_squared=0.95,
            valid=True
        )

        regime_info = self.analyzer.detect_scaling_regime(constant_result)
        self.assertEqual(regime_info["regime"], "constant")

        # Test invalid result
        invalid_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 2, 4],
            scaling_exponent=None,
            valid=False
        )

        regime_info = self.analyzer.detect_scaling_regime(invalid_result)
        self.assertEqual(regime_info["regime"], "invalid")

    def test_detect_scaling_regime_edge_cases(self):
        """Test scaling regime detection edge cases."""
        # Test with None values
        none_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 2, 4],
            scaling_exponent=None,
            r_squared=None,
            valid=True
        )

        regime_info = self.analyzer.detect_scaling_regime(none_result)
        self.assertEqual(regime_info["regime"], "invalid")
        self.assertIsNone(regime_info["confidence"])

    def test_compare_scaling_regimes(self):
        """Test comparison of scaling regimes."""
        scaling_results = {
            "complexity": ScalingResult(
                parameter_name="complexity",
                parameter_values=[1, 2, 4],
                response_values=[1, 2, 4],
                scaling_exponent=1.0,
                r_squared=0.95,
                regime="linear",
                valid=True
            ),
            "memory": ScalingResult(
                parameter_name="memory",
                parameter_values=[100, 200, 400],
                response_values=[1, 1.5, 2.0],
                scaling_exponent=0.8,
                r_squared=0.90,
                regime="sub-linear",
                valid=True
            )
        }

        comparison = self.analyzer.compare_scaling_regimes(scaling_results)

        self.assertIsInstance(comparison, dict)
        self.assertIn("num_valid_results", comparison)
        self.assertIn("exponent_range", comparison)
        self.assertIn("regime_distribution", comparison)
        self.assertIn("most_challenging", comparison)

        self.assertEqual(comparison["num_valid_results"], 2)
        self.assertIsInstance(comparison["regime_distribution"], dict)

    def test_compare_scaling_regimes_empty(self):
        """Test comparison with empty scaling results."""
        comparison = self.analyzer.compare_scaling_regimes({})

        self.assertIsInstance(comparison, dict)
        self.assertIn("error", comparison)
        self.assertEqual(comparison["error"], "No scaling results to compare")

    def test_compare_scaling_regimes_invalid(self):
        """Test comparison with only invalid scaling results."""
        invalid_results = {
            "param1": ScalingResult(
                parameter_name="param1",
                parameter_values=[1, 2],
                response_values=[1, 2],
                valid=False
            )
        }

        comparison = self.analyzer.compare_scaling_regimes(invalid_results)

        self.assertIsInstance(comparison, dict)
        self.assertIn("error", comparison)
        self.assertEqual(comparison["error"], "No valid scaling results")


    def test_paired_bootstrap_is_required_and_not_an_inferential_p_value(self):
        analyzer = ScalingAnalyzer(bootstrap_samples=100, random_seed=7)
        result = analyzer.analyze_single_parameter_scaling([1, 10, 100], [1, 100, 10000])
        self.assertTrue(result.valid)
        self.assertGreater(result.bootstrap_valid_samples, 0)
        self.assertEqual(result.confidence_interval, (2, 2))
        self.assertIsNone(result.p_value)
        more = ScalingAnalyzer(bootstrap_samples=1000, random_seed=7).analyze_single_parameter_scaling([1, 10, 100], [1, 100, 10000])
        self.assertEqual(more.confidence_interval, (2, 2))
        self.assertIsNone(more.p_value)

    def test_scaling_analysis_with_numpy_fallback(self):
        """Test scaling analysis when NumPy operations might fail."""
        # This test ensures the code handles edge cases gracefully
        # when NumPy operations might fail

        # Test with minimal data that might cause numerical issues
        x_vals = [1.0, 1.0, 1.0]  # All same values
        y_vals = [1.0, 1.0, 1.0]

        result = self.analyzer.analyze_single_parameter_scaling(
            x_vals, y_vals, "constant_param", "constant_response"
        )

        # Result should be created even if analysis fails
        self.assertIsInstance(result, ScalingResult)
        self.assertEqual(result.parameter_name, "constant_param")


    def test_constant_and_polynomial_regimes_remain_aligned(self):
        analyzer = ScalingAnalyzer(bootstrap_samples=100)
        constant = analyzer.analyze_single_parameter_scaling([1, 10, 100], [3, 3, 3], "constant")
        quadratic = analyzer.analyze_single_parameter_scaling([1, 10, 100], [1, 100, 10000], "quadratic")
        cubic = analyzer.analyze_single_parameter_scaling([1, 10, 100], [1, 1000, 1000000], "cubic")
        self.assertEqual(constant.confidence_interval, (0, 0))
        self.assertEqual(constant.regime, "constant")
        self.assertEqual(cubic.scaling_exponent, 3)
        self.assertEqual(cubic.regime, "cubic")
        self.assertEqual(analyzer.detect_scaling_regime(cubic)["regime"], "cubic")
        self.assertIsNone(analyzer.detect_scaling_regime(cubic)["confidence"])
        comparison = analyzer.compare_scaling_regimes({"constant": constant, "quadratic": quadratic})
        self.assertEqual(comparison["exponent_range"], (0, 2))
        self.assertEqual(comparison["most_challenging"]["parameter"], "quadratic")
        self.assertEqual(comparison["most_challenging"]["exponent"], 2)
        only_constant = analyzer.compare_scaling_regimes({"constant": constant})
        self.assertEqual(only_constant["exponent_range"], (0, 0))

    def test_marginal_correlation_is_not_an_interaction(self):
        result = ScalingAnalyzer(bootstrap_samples=50).analyze_multi_parameter_scaling(
            "x", [1, 2, 4, 8], {"z": [2, 4, 8, 16]}, [1, 4, 16, 64]
        )
        self.assertIsNone(result.interaction_effects)
        self.assertAlmostEqual(result.parameter_correlations["z"], 1)
        self.assertNotIn("x", result.parameter_correlations)

    def test_scaling_local_rng_and_invalid_domains(self):
        state = random.getstate()
        numpy_state = np.random.get_state()
        analyzer = ScalingAnalyzer(bootstrap_samples=30, random_seed=7)
        first = analyzer.analyze_single_parameter_scaling([1, 2, 4, 8], [1, 3, 5, 10])
        second = analyzer.analyze_single_parameter_scaling([1, 2, 4, 8], [1, 3, 5, 10])
        self.assertEqual(first.confidence_interval, second.confidence_interval)
        self.assertEqual(random.getstate(), state)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(numpy_state[1], after[1]))
        self.assertEqual(numpy_state[2:], after[2:])
        for bad in [float("nan"), float("inf"), -1, 0]:
            with self.subTest(value=bad):
                result = analyzer.analyze_single_parameter_scaling([1, 2, bad], [1, 2, 3])
                self.assertFalse(result.valid)
                self.assertIsNotNone(result.error)
        for count in [0, -1, 2.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                ScalingAnalyzer(bootstrap_samples=count)
        for confidence in [0, 1, float("nan")]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                ScalingAnalyzer(confidence_level=confidence)


if __name__ == '__main__':
    unittest.main()
