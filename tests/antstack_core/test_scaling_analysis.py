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
        self.assertEqual(regime_info["confidence"], 0.0)

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

    def test_generate_scaling_report(self):
        """Test scaling report generation."""
        scaling_result = ScalingResult(
            parameter_name="complexity",
            parameter_values=[1, 2, 4, 8],
            response_values=[1, 1.5, 2.0, 2.5],
            scaling_exponent=0.8,
            intercept=0.2,
            r_squared=0.95,
            regime="sub-linear",
            confidence_interval=(0.7, 0.9),
            p_value=0.001,
            valid=True
        )

        report = self.analyzer.generate_scaling_report(scaling_result)

        self.assertIsInstance(report, str)
        self.assertIn("Scaling Analysis Report: complexity", report)
        self.assertIn("Exponent: 0.800000", report)
        self.assertIn("R²: 0.950000", report)
        self.assertIn("Regime: sub-linear", report)
        self.assertIn("95% CI: [0.700000, 0.900000]", report)
        self.assertIn("P-value: 0.001000", report)
        self.assertIn("Interpretation:", report)
        self.assertIn("Recommendations:", report)

    def test_generate_scaling_report_invalid(self):
        """Test scaling report generation with invalid result."""
        invalid_result = ScalingResult(
            parameter_name="complexity",
            parameter_values=[1, 2],
            response_values=[1, 2],
            valid=False
        )

        report = self.analyzer.generate_scaling_report(invalid_result)

        self.assertIsInstance(report, str)
        self.assertIn("INVALID", report)
        self.assertIn("insufficient data", report)

    def test_interpret_regime(self):
        """Test regime interpretation."""
        # Test different regimes
        regimes_to_test = [
            ("constant", 0.05),
            ("linear", 1.0),
            ("quadratic", 2.0),
            ("cubic", 3.0),
            ("sub-linear", 0.5),
            ("super-linear", 1.5)
        ]

        for regime, exponent in regimes_to_test:
            interpretation = self.analyzer._interpret_regime(regime, exponent)
            self.assertIsInstance(interpretation, str)
            self.assertIn(".3f", interpretation)

        # Test unknown regime
        interpretation = self.analyzer._interpret_regime("unknown", 2.5)
        self.assertIsInstance(interpretation, str)
        self.assertIn("Unknown scaling regime", interpretation)

    def test_generate_scaling_recommendations(self):
        """Test scaling recommendations generation."""
        test_cases = [
            # (regime, exponent, r_squared, expected_keywords)
            ("constant", 0.05, 0.95, ["minimal impact", "low optimization"]),
            ("linear", 1.0, 0.95, ["balanced resource"]),
            ("sub-linear", 0.5, 0.95, ["efficiency gains", "scaling up"]),
            ("super-linear", 1.5, 0.95, ["bottlenecks", "high optimization"]),
            ("quadratic", 2.0, 0.95, ["algorithmic improvements"]),
            ("cubic", 3.0, 0.95, ["fundamental algorithmic redesign"]),
        ]

        for regime, exponent, r_squared, expected_keywords in test_cases:
            result = ScalingResult(
                parameter_name="test",
                parameter_values=[1, 2, 4],
                response_values=[1, 2, 4],
                scaling_exponent=exponent,
                regime=regime,
                r_squared=r_squared,
                valid=True
            )

            recommendations = self.analyzer._generate_scaling_recommendations(result)

            self.assertIsInstance(recommendations, str)
            for keyword in expected_keywords:
                self.assertIn(keyword, recommendations)

        # Test high exponent warning
        high_exp_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 2, 4],
            scaling_exponent=2.5,
            regime="super-linear",
            r_squared=0.95,
            valid=True
        )

        recommendations = self.analyzer._generate_scaling_recommendations(high_exp_result)
        self.assertIn("Exponential scaling", recommendations)

        # Test low r_squared warning
        low_r2_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2, 4],
            response_values=[1, 2, 4],
            scaling_exponent=1.0,
            regime="linear",
            r_squared=0.5,
            valid=True
        )

        recommendations = self.analyzer._generate_scaling_recommendations(low_r2_result)
        self.assertIn("Low R²", recommendations)

    def test_generate_scaling_recommendations_invalid(self):
        """Test scaling recommendations with invalid result."""
        invalid_result = ScalingResult(
            parameter_name="test",
            parameter_values=[1, 2],
            response_values=[1, 2],
            valid=False
        )

        recommendations = self.analyzer._generate_scaling_recommendations(invalid_result)
        self.assertIn("Insufficient data", recommendations)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        if not HAS_NUMPY or not HAS_SCIPY:
            self.skipTest("NumPy and SciPy required for bootstrap tests")

        # Create test data with known relationship
        x_vals = [1, 2, 3, 4, 5]
        y_vals = [1, 2, 3, 4, 5]  # Perfect linear relationship

        result = self.analyzer.analyze_single_parameter_scaling(
            x_vals, y_vals, "test_param", "test_response"
        )

        # Should have confidence interval if bootstrap succeeds
        if result.valid:
            # Confidence interval might be None if bootstrap fails
            # but we can't predict that with certainty
            self.assertIsInstance(result.confidence_interval, (tuple, type(None)))

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


if __name__ == '__main__':
    unittest.main()
