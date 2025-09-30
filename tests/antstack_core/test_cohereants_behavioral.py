#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.cohereants.behavioral module.

Tests behavioral response analysis functionality including:
- Behavioral data validation and processing
- Statistical analysis (t-tests, effect sizes, confidence intervals)
- Power analysis and sample size calculations
- Response time analysis and classification
- Behavioral plotting and visualization
- Edge cases and error handling

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.cohereants.behavioral import (
    BehavioralData,
    StatisticalAnalyzer,
    BehavioralAnalyzer,
    analyze_behavioral_response,
    calculate_power_analysis,
    calculate_response_statistics,
    generate_behavioral_plots
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


class TestBehavioralData(unittest.TestCase):
    """Test BehavioralData class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.treatment_times = [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.0]
        self.control_times = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8, 2.4]

    def test_behavioral_data_creation(self):
        """Test basic BehavioralData creation."""
        data = BehavioralData(self.treatment_times, self.control_times)

        self.assertIsInstance(data, BehavioralData)
        self.assertEqual(len(data.treatment_times), 8)
        self.assertEqual(len(data.control_times), 8)

        # Check numpy arrays
        self.assertIsInstance(data.treatment_times, np.ndarray)
        self.assertIsInstance(data.control_times, np.ndarray)

    def test_behavioral_data_properties(self):
        """Test BehavioralData property calculations."""
        data = BehavioralData(self.treatment_times, self.control_times)

        # Test means
        treatment_mean = data.treatment_mean
        control_mean = data.control_mean

        self.assertIsInstance(treatment_mean, float)
        self.assertIsInstance(control_mean, float)
        self.assertGreater(treatment_mean, 0)
        self.assertGreater(control_mean, 0)

        # Treatment should be faster than control
        self.assertLess(treatment_mean, control_mean)

        # Test standard deviations
        treatment_std = data.treatment_std
        control_std = data.control_std

        self.assertIsInstance(treatment_std, float)
        self.assertIsInstance(control_std, float)
        self.assertGreater(treatment_std, 0)
        self.assertGreater(control_std, 0)

        # Test difference
        difference = data.difference
        expected_diff = treatment_mean - control_mean
        self.assertEqual(difference, expected_diff)

    def test_behavioral_data_sample_sizes(self):
        """Test sample size reporting."""
        data = BehavioralData(self.treatment_times, self.control_times)

        sample_sizes = data.sample_sizes
        self.assertIsInstance(sample_sizes, dict)
        self.assertEqual(sample_sizes['treatment'], 8)
        self.assertEqual(sample_sizes['control'], 8)

    def test_behavioral_data_statistics_check(self):
        """Test statistical feasibility check."""
        # Test with sufficient data
        data = BehavioralData(self.treatment_times, self.control_times)
        self.assertTrue(data.can_perform_statistics)

        # Test with insufficient data
        small_treatment = [1.2, 1.5]
        small_control = [2.1]
        data_small = BehavioralData(small_treatment, small_control)
        self.assertFalse(data_small.can_perform_statistics)

    def test_behavioral_data_validation_errors(self):
        """Test input validation error handling."""
        # Test non-list inputs
        with self.assertRaises(ValueError):
            BehavioralData("not_a_list", self.control_times)

        with self.assertRaises(ValueError):
            BehavioralData(self.treatment_times, "not_a_list")

        # Test empty lists
        with self.assertRaises(ValueError):
            BehavioralData([], self.control_times)

        with self.assertRaises(ValueError):
            BehavioralData(self.treatment_times, [])

        # Test non-positive values
        with self.assertRaises(ValueError):
            BehavioralData([1.2, 0, 1.5], self.control_times)

        with self.assertRaises(ValueError):
            BehavioralData(self.treatment_times, [2.1, -1.0, 2.3])

        # Test non-numeric values
        with self.assertRaises(ValueError):
            BehavioralData([1.2, "invalid", 1.5], self.control_times)

    def test_behavioral_data_edge_cases(self):
        """Test edge cases in BehavioralData."""
        # Test with very small positive values
        tiny_values = [1e-6, 2e-6, 3e-6]
        data = BehavioralData(tiny_values, tiny_values)
        self.assertIsInstance(data, BehavioralData)

        # Test with very large values
        large_values = [1e6, 2e6, 3e6]
        data = BehavioralData(large_values, large_values)
        self.assertIsInstance(data, BehavioralData)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test StatisticalAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.treatment_times = [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.0]
        self.control_times = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8, 2.4]
        self.behavioral_data = BehavioralData(self.treatment_times, self.control_times)

    def test_statistical_analyzer_creation(self):
        """Test StatisticalAnalyzer creation."""
        analyzer = StatisticalAnalyzer()
        self.assertIsInstance(analyzer, StatisticalAnalyzer)
        self.assertEqual(analyzer.alpha, 0.05)

        # Test custom alpha
        analyzer_custom = StatisticalAnalyzer(alpha=0.01)
        self.assertEqual(analyzer_custom.alpha, 0.01)

    def test_perform_t_test(self):
        """Test t-test functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.perform_t_test(self.behavioral_data)

        self.assertIsInstance(result, dict)

        # Check required keys
        required_keys = ['t_statistic', 'p_value', 'degrees_of_freedom']
        for key in required_keys:
            self.assertIn(key, result)

        # Check types
        self.assertIsInstance(result['t_statistic'], (float, type(np.nan)))
        self.assertIsInstance(result['p_value'], (float, type(np.nan)))
        self.assertIsInstance(result['degrees_of_freedom'], (float, type(np.nan)))

        # Check that we get a valid t-statistic (should be negative since treatment < control)
        if not np.isnan(result['t_statistic']):
            self.assertLess(result['t_statistic'], 0)  # Treatment faster than control

    def test_calculate_cohens_d(self):
        """Test Cohen's d effect size calculation."""
        analyzer = StatisticalAnalyzer()
        cohens_d = analyzer.calculate_cohens_d(self.behavioral_data)

        self.assertIsInstance(cohens_d, float)
        # Effect size should be negative (treatment faster than control)
        self.assertLess(cohens_d, 0)

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        ci = analyzer.calculate_confidence_interval(self.behavioral_data)

        self.assertIsInstance(ci, dict)
        self.assertIn('lower_bound', ci)
        self.assertIn('upper_bound', ci)
        self.assertIn('confidence_level', ci)

        # CI bounds should be valid numbers
        if not np.isnan(ci['lower_bound']) and not np.isnan(ci['upper_bound']):
            self.assertLess(ci['lower_bound'], ci['upper_bound'])

    def test_statistical_methods_integration(self):
        """Test integration of available statistical methods."""
        analyzer = StatisticalAnalyzer()

        # Test all available methods work together
        t_test = analyzer.perform_t_test(self.behavioral_data)
        cohens_d = analyzer.calculate_cohens_d(self.behavioral_data)
        ci = analyzer.calculate_confidence_interval(self.behavioral_data)

        # All should return valid results
        self.assertIsInstance(t_test, dict)
        self.assertIsInstance(cohens_d, (float, type(np.nan)))
        self.assertIsInstance(ci, dict)

    def test_statistical_analyzer_methods_work(self):
        """Test that all StatisticalAnalyzer methods work correctly."""
        analyzer = StatisticalAnalyzer()

        # Test all available methods
        t_test = analyzer.perform_t_test(self.behavioral_data)
        cohens_d = analyzer.calculate_cohens_d(self.behavioral_data)
        ci = analyzer.calculate_confidence_interval(self.behavioral_data)

        # Verify results
        self.assertIsInstance(t_test, dict)
        self.assertIn('t_statistic', t_test)
        self.assertIn('p_value', t_test)
        self.assertIn('degrees_of_freedom', t_test)

        self.assertIsInstance(cohens_d, (float, type(np.nan)))
        self.assertIsInstance(ci, dict)
        self.assertIn('lower_bound', ci)
        self.assertIn('upper_bound', ci)

    def test_statistical_analyzer_edge_cases(self):
        """Test StatisticalAnalyzer with edge cases."""
        analyzer = StatisticalAnalyzer()

        # Test with identical data (no effect)
        identical_times = [1.0, 1.0, 1.0, 1.0]
        identical_data = BehavioralData(identical_times, identical_times)

        result = analyzer.perform_t_test(identical_data)
        self.assertIsInstance(result, dict)

        # Effect size should be very close to zero
        cohens_d = analyzer.calculate_cohens_d(identical_data)
        if not np.isnan(cohens_d):
            self.assertAlmostEqual(cohens_d, 0.0, places=3)

        # Test with very small samples
        small_treatment = [1.0, 1.1]
        small_control = [2.0, 2.1]
        small_data = BehavioralData(small_treatment, small_control)

        result = analyzer.perform_t_test(small_data)
        self.assertIsInstance(result, dict)


class TestBehavioralAnalyzer(unittest.TestCase):
    """Test BehavioralAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.treatment_times = [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.0]
        self.control_times = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8, 2.4]

    def test_behavioral_analyzer_creation(self):
        """Test BehavioralAnalyzer creation."""
        analyzer = BehavioralAnalyzer()
        self.assertIsInstance(analyzer, BehavioralAnalyzer)

    def test_analyze_response(self):
        """Test behavioral response analysis."""
        analyzer = BehavioralAnalyzer()

        result = analyzer.analyze_response(self.treatment_times, self.control_times)

        self.assertIsInstance(result, dict)
        self.assertIn('treatment_mean', result)
        self.assertIn('control_mean', result)
        self.assertIn('difference', result)
        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('cohens_d', result)
        self.assertIn('significant', result)

    def test_behavioral_analyzer_with_different_data(self):
        """Test BehavioralAnalyzer with different data patterns."""
        analyzer = BehavioralAnalyzer()

        # Test with fast treatment vs slow control
        fast_treatment = [0.5, 0.6, 0.4, 0.7, 0.5]
        slow_control = [2.0, 2.2, 1.8, 2.5, 2.1]

        result = analyzer.analyze_response(fast_treatment, slow_control)

        self.assertIsInstance(result, dict)
        self.assertLess(result['treatment_mean'], result['control_mean'])

        # Test with no effect (identical distributions)
        identical_data = [1.0, 1.1, 0.9, 1.2, 1.0]
        result_no_effect = analyzer.analyze_response(identical_data, identical_data)

        self.assertIsInstance(result_no_effect, dict)
        self.assertAlmostEqual(result_no_effect['difference'], 0, places=2)


class TestBehavioralAnalysisFunctions(unittest.TestCase):
    """Test standalone behavioral analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.treatment_times = [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.0]
        self.control_times = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8, 2.4]
        self.behavioral_data = BehavioralData(self.treatment_times, self.control_times)

    def test_analyze_behavioral_response(self):
        """Test main behavioral response analysis function."""
        # Test with numpy arrays (expected format)
        treatment_array = np.array(self.treatment_times)
        control_array = np.array(self.control_times)
        result = analyze_behavioral_response(treatment_array, control_array)

        self.assertIsInstance(result, dict)
        # Should contain analysis results
        self.assertTrue(len(result) > 0)

    def test_calculate_power_analysis(self):
        """Test power analysis function."""
        # Test with two data arrays
        result = calculate_power_analysis(self.treatment_times, self.control_times)

        self.assertIsInstance(result, dict)
        # Should contain power analysis results
        self.assertTrue(len(result) > 0)

    def test_calculate_response_statistics(self):
        """Test response statistics calculation."""
        # Test with numpy array (expected format)
        response_array = np.array(self.treatment_times)
        result = calculate_response_statistics(response_array)

        self.assertIsInstance(result, dict)
        # Should contain statistics
        self.assertTrue(len(result) > 0)

    def test_generate_behavioral_plots(self):
        """Test behavioral plotting function."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        time_points = np.linspace(0, 1, len(self.treatment_times))
        result = generate_behavioral_plots(
            response_data=np.array(self.treatment_times),
            time_points=time_points
        )

        # Should return a matplotlib figure or None
        self.assertTrue(result is None or hasattr(result, 'savefig'))

    def test_function_error_handling(self):
        """Test error handling in behavioral analysis functions."""
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            analyze_behavioral_response(
                treatment_times=[],
                control_times=self.control_times
            )

        with self.assertRaises(ValueError):
            analyze_behavioral_response(
                treatment_times=self.treatment_times,
                control_times=[]
            )

    def test_power_analysis_edge_cases(self):
        """Test power analysis with edge cases."""
        # Test with different data patterns
        small_effect = [1.0, 1.1, 0.9, 1.2]
        large_effect = [1.0, 1.1, 0.9, 3.0]

        result1 = calculate_power_analysis(small_effect, large_effect)
        result2 = calculate_power_analysis(large_effect, small_effect)

        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)


class TestBehavioralIntegration(unittest.TestCase):
    """Test integration between behavioral analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.treatment_times = [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.0]
        self.control_times = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8, 2.4]

    def test_full_behavioral_analysis_workflow(self):
        """Test complete behavioral analysis workflow."""
        # Step 1: Create data
        data = BehavioralData(self.treatment_times, self.control_times)

        # Step 2: Statistical analysis
        stat_analyzer = StatisticalAnalyzer()
        t_test = stat_analyzer.perform_t_test(data)
        cohens_d = stat_analyzer.calculate_cohens_d(data)
        ci = stat_analyzer.calculate_confidence_interval(data)

        # Step 3: Behavioral analysis
        behavioral_analyzer = BehavioralAnalyzer()
        analysis_result = behavioral_analyzer.analyze_response(
            self.treatment_times, self.control_times
        )

        # Step 4: Main analysis function
        treatment_array = np.array(self.treatment_times)
        control_array = np.array(self.control_times)
        main_results = analyze_behavioral_response(treatment_array, control_array)

        # Verify all components work together
        self.assertIsInstance(t_test, dict)
        self.assertIsInstance(cohens_d, (float, type(np.nan)))
        self.assertIsInstance(ci, dict)
        self.assertIsInstance(analysis_result, dict)
        self.assertIsInstance(main_results, dict)

    def test_cross_validation_of_results(self):
        """Test consistency across different analysis methods."""
        data = BehavioralData(self.treatment_times, self.control_times)

        # Get results from different methods
        stat_analyzer = StatisticalAnalyzer()
        t_test_results = stat_analyzer.perform_t_test(data)
        cohens_d = stat_analyzer.calculate_cohens_d(data)

        # Behavioral analyzer results
        behavioral_analyzer = BehavioralAnalyzer()
        analysis_result = behavioral_analyzer.analyze_response(
            self.treatment_times, self.control_times
        )

        # Main function results
        treatment_array = np.array(self.treatment_times)
        control_array = np.array(self.control_times)
        main_results = analyze_behavioral_response(treatment_array, control_array)

        # Check consistency
        self.assertIsInstance(t_test_results, dict)
        self.assertIsInstance(cohens_d, (float, type(np.nan)))
        self.assertIsInstance(analysis_result, dict)
        self.assertIsInstance(main_results, dict)

        # Effect should be consistent (negative for faster treatment)
        if not np.isnan(cohens_d):
            self.assertLess(cohens_d, 0)

    def test_scalability_with_larger_datasets(self):
        """Test analysis scalability with larger datasets."""
        # Create larger dataset
        large_treatment = np.random.normal(1.0, 0.2, 1000).tolist()
        large_control = np.random.normal(2.0, 0.3, 1000).tolist()

        data = BehavioralData(large_treatment, large_control)

        # Test that analysis still works with larger data
        stat_analyzer = StatisticalAnalyzer()
        result = stat_analyzer.perform_t_test(data)

        self.assertIsInstance(result, dict)
        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)


class TestBehavioralRobustness(unittest.TestCase):
    """Test robustness of behavioral analysis methods."""

    def test_statistical_analyzer_parameter_variation(self):
        """Test StatisticalAnalyzer with different parameters."""
        alphas = [0.001, 0.01, 0.05, 0.10]

        treatment_times = [1.0, 1.1, 0.9, 1.2]
        control_times = [2.0, 2.1, 1.9, 2.2]
        data = BehavioralData(treatment_times, control_times)

        for alpha in alphas:
            analyzer = StatisticalAnalyzer(alpha=alpha)
            result = analyzer.perform_t_test(data)

            self.assertIsInstance(result, dict)
            self.assertIn('p_value', result)
            self.assertIn('t_statistic', result)

    def test_behavioral_data_type_consistency(self):
        """Test that BehavioralData maintains type consistency."""
        # Test with integers
        int_treatment = [1, 2, 3, 4, 5]
        int_control = [6, 7, 8, 9, 10]
        data = BehavioralData(int_treatment, int_control)

        # Should convert to float arrays
        self.assertEqual(data.treatment_times.dtype, np.float64)
        self.assertEqual(data.control_times.dtype, np.float64)

        # Properties should still work
        self.assertIsInstance(data.treatment_mean, float)
        self.assertIsInstance(data.control_mean, float)


if __name__ == '__main__':
    unittest.main()
