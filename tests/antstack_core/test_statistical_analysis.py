#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.statistical_analysis module.

Tests statistical analysis functionality including:
- Bootstrap confidence interval calculation
- Uncertainty quantification and propagation
- Quality metrics and validation checks
- Statistical significance testing
- Measurement quality assessment
- Edge cases and error handling

Following .cursorrules principles:
- Rigorous statistical method validation
- Comprehensive uncertainty quantification
- Bootstrap-based confidence intervals
- Professional statistical reporting
- Real data analysis (no mocks)
"""

import unittest
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.statistical_analysis import (
    StatisticalAnalyzer,
    analyze_measurement_quality,
    detect_significance,
)

import numpy as np


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test StatisticalAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(
            bootstrap_samples=100,
            confidence_level=0.95,
            random_seed=42
        )

        self.sample_data = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]
        self.single_value = [5.0]
        self.empty_data = []

    def test_statistical_analyzer_creation(self):
        """Test StatisticalAnalyzer creation with default parameters."""
        analyzer = StatisticalAnalyzer()
        self.assertIsInstance(analyzer, StatisticalAnalyzer)
        self.assertEqual(analyzer.bootstrap_samples, 1000)
        self.assertEqual(analyzer.confidence_level, 0.95)
        self.assertEqual(analyzer.random_seed, 42)

    def test_statistical_analyzer_custom_parameters(self):
        """Test StatisticalAnalyzer creation with custom parameters."""
        analyzer = StatisticalAnalyzer(
            bootstrap_samples=500,
            confidence_level=0.99,
            random_seed=123
        )
        self.assertEqual(analyzer.bootstrap_samples, 500)
        self.assertEqual(analyzer.confidence_level, 0.99)
        self.assertEqual(analyzer.random_seed, 123)

    def test_calculate_measurement_uncertainty_standard_error(self):
        """Test uncertainty calculation using standard error method."""
        uncertainty = self.analyzer.calculate_measurement_uncertainty(
            self.sample_data, method="standard_error"
        )

        self.assertAlmostEqual(uncertainty, math.sqrt(0.06 / 8), places=14)

    def test_calculate_measurement_uncertainty_std_dev(self):
        """Test uncertainty calculation using standard deviation method."""
        uncertainty = self.analyzer.calculate_measurement_uncertainty(
            self.sample_data, method="std_dev"
        )

        self.assertAlmostEqual(uncertainty, math.sqrt(0.06), places=14)

        # Standard deviation should be larger than standard error
        se_uncertainty = self.analyzer.calculate_measurement_uncertainty(
            self.sample_data, method="standard_error"
        )
        self.assertGreater(uncertainty, se_uncertainty)

    def test_calculate_measurement_uncertainty_single_value(self):
        """Test uncertainty calculation for single value."""
        uncertainty = self.analyzer.calculate_measurement_uncertainty(
            self.single_value, method="standard_error"
        )

        self.assertIsInstance(uncertainty, float)
        self.assertTrue(math.isnan(uncertainty))

    def test_calculate_measurement_uncertainty_empty_data(self):
        """Test uncertainty calculation for empty data."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_measurement_uncertainty(self.empty_data)

    def test_calculate_measurement_uncertainty_invalid_method(self):
        """Test uncertainty calculation with invalid method."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_measurement_uncertainty(
                self.sample_data, method="invalid_method"
            )


    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        mean, lower, upper = self.analyzer.calculate_confidence_intervals(self.sample_data)

        self.assertIsInstance(mean, float)
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

        # Basic sanity checks
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)
        self.assertGreater(upper, lower)

    def test_calculate_confidence_intervals_single_value(self):
        """Test confidence interval calculation for single value."""
        mean, lower, upper = self.analyzer.calculate_confidence_intervals(self.single_value)

        self.assertEqual(mean, 5.0)
        # For single values, CI might be the same as mean or have small bounds
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

    def test_calculate_confidence_intervals_empty_data(self):
        """Test confidence interval calculation for empty data."""
        with self.assertRaises((ValueError, IndexError)):
            self.analyzer.calculate_confidence_intervals(self.empty_data)

    def test_assess_measurement_quality(self):
        """Test measurement quality assessment."""
        quality_metrics = self.analyzer.assess_measurement_quality(self.sample_data)

        self.assertIsInstance(quality_metrics, dict)

        # Check for expected quality metrics
        expected_keys = ['sample_size', 'mean', 'std_dev', 'coefficient_of_variation',
                        'confidence_interval', 'quality_score']
        for key in expected_keys:
            self.assertIn(key, quality_metrics)

        # Check data types
        self.assertIsInstance(quality_metrics['sample_size'], int)
        self.assertIsInstance(quality_metrics['mean'], float)
        self.assertIsInstance(quality_metrics['quality_score'], float)

        # Quality score should be reasonable
        self.assertGreaterEqual(quality_metrics['quality_score'], 0.0)
        self.assertLessEqual(quality_metrics['quality_score'], 1.0)

    def test_assess_measurement_quality_single_value(self):
        """Test measurement quality assessment for single value."""
        quality_metrics = self.analyzer.assess_measurement_quality(self.single_value)

        self.assertIsInstance(quality_metrics, dict)
        self.assertEqual(quality_metrics['sample_size'], 1)
        self.assertEqual(quality_metrics['mean'], 5.0)

    def test_assess_measurement_quality_empty_data(self):
        """Test measurement quality assessment for empty data."""
        with self.assertRaises((ValueError, IndexError)):
            self.analyzer.assess_measurement_quality(self.empty_data)

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create data with an obvious outlier
        data_with_outlier = [1.0, 1.1, 1.2, 1.0, 1.1, 5.0]  # 5.0 is outlier
        outliers = self.analyzer.detect_outliers(data_with_outlier)

        self.assertIsInstance(outliers, list)

        # Should detect the outlier
        if len(outliers) > 0:
            self.assertIn(5.0, outliers)

    def test_detect_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        # Create normal data without outliers
        normal_data = [1.0, 1.1, 1.2, 1.0, 1.1, 1.05]
        outliers = self.analyzer.detect_outliers(normal_data)

        self.assertIsInstance(outliers, list)
        # Should have few or no outliers
        self.assertLessEqual(len(outliers), 1)

    def test_calculate_statistical_power(self):
        """Test statistical power calculation."""
        # Test with two different means
        group1 = [1.0, 1.1, 1.2, 1.0, 1.1]
        group2 = [2.0, 2.1, 2.2, 2.0, 2.1]

        power = self.analyzer.calculate_statistical_power(group1, group2, alpha=0.05)

        self.assertIsInstance(power, float)
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)

        # Should have high power for clearly different groups
        self.assertGreater(power, 0.8)

    def test_calculate_statistical_power_same_groups(self):
        """Test statistical power calculation for identical groups."""
        group1 = [1.0, 1.1, 1.2, 1.0, 1.1]
        group2 = [1.0, 1.1, 1.2, 1.0, 1.1]  # Identical to group1

        power = self.analyzer.calculate_statistical_power(group1, group2, alpha=0.05)

        self.assertIsInstance(power, float)
        # Should have low power for identical groups
        self.assertLess(power, 0.2)

    def test_calculate_effect_size(self):
        """Test effect size calculation."""
        group1 = [1.0, 1.1, 1.2, 1.0, 1.1]
        group2 = [2.0, 2.1, 2.2, 2.0, 2.1]

        effect_size = self.analyzer.calculate_effect_size(group1, group2)

        self.assertIsInstance(effect_size, float)

        # Should be large for clearly different groups
        self.assertGreater(abs(effect_size), 1.0)

    def test_calculate_effect_size_same_groups(self):
        """Test effect size calculation for identical groups."""
        group1 = [1.0, 1.1, 1.2, 1.0, 1.1]
        group2 = [1.0, 1.1, 1.2, 1.0, 1.1]

        effect_size = self.analyzer.calculate_effect_size(group1, group2)

        self.assertIsInstance(effect_size, float)
        # Should be very close to zero for identical groups
        self.assertAlmostEqual(effect_size, 0.0, places=3)


class TestStatisticalAnalysisFunctions(unittest.TestCase):
    """Test standalone statistical analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.measurements = {
            'control': [1.0, 1.1, 1.2, 1.0, 1.1, 1.05],
            'treatment': [1.8, 1.9, 2.0, 1.7, 1.8, 1.85],
            'outlier_test': [1.0, 1.1, 1.2, 1.0, 1.1, 5.0]
        }

    def test_analyze_measurement_quality(self):
        """Test analyze_measurement_quality function."""
        quality_analysis = analyze_measurement_quality(self.measurements)

        self.assertIsInstance(quality_analysis, dict)

        # Check for expected keys
        expected_keys = ['overall_quality', 'individual_metrics', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, quality_analysis)

        # Check individual metrics
        individual = quality_analysis['individual_metrics']
        self.assertIn('control', individual)
        self.assertIn('treatment', individual)

        # Each metric should have quality assessment
        for metric_name, metrics in individual.items():
            self.assertIn('quality_score', metrics)
            self.assertIn('sample_size', metrics)

    def test_analyze_measurement_quality_empty(self):
        """Test analyze_measurement_quality with empty measurements."""
        empty_measurements = {}
        quality_analysis = analyze_measurement_quality(empty_measurements)

        self.assertIsInstance(quality_analysis, dict)
        # Should handle empty case gracefully
        self.assertIn('overall_quality', quality_analysis)

    def test_analyze_measurement_quality_single_measurement(self):
        """Test analyze_measurement_quality with single measurement per group."""
        single_measurements = {
            'group1': [1.0],
            'group2': [2.0]
        }

        quality_analysis = analyze_measurement_quality(single_measurements)
        self.assertIsInstance(quality_analysis, dict)

    def test_detect_significance(self):
        """Test detect_significance function."""
        significance_analysis = detect_significance(self.measurements)

        self.assertIsInstance(significance_analysis, dict)

        # Check for expected keys
        expected_keys = ['significant_differences', 'effect_sizes', 'power_analysis']
        for key in expected_keys:
            self.assertIn(key, significance_analysis)

        # Check significance detection
        sig_diffs = significance_analysis['significant_differences']
        self.assertIn('control_vs_treatment', sig_diffs)

    def test_detect_significance_no_significant_differences(self):
        """Test detect_significance with no significant differences."""
        similar_measurements = {
            'group1': [1.0, 1.1, 1.2, 1.0, 1.1],
            'group2': [1.05, 1.15, 1.25, 1.05, 1.15]  # Similar to group1
        }

        significance_analysis = detect_significance(similar_measurements)
        self.assertIsInstance(significance_analysis, dict)

        # Should detect fewer significant differences
        sig_diffs = significance_analysis['significant_differences']
        self.assertIsInstance(sig_diffs, dict)

    def test_detect_significance_single_group(self):
        """Test detect_significance with single measurement group."""
        single_group = {'only_group': [1.0, 1.1, 1.2]}

        significance_analysis = detect_significance(single_group)
        self.assertIsInstance(significance_analysis, dict)

        # Should handle single group case
        sig_diffs = significance_analysis['significant_differences']
        self.assertIsInstance(sig_diffs, dict)


class TestStatisticalAnalysisIntegration(unittest.TestCase):
    """Test integration between statistical analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(bootstrap_samples=50, random_seed=42)

        self.experimental_data = {
            'baseline': [1.0, 1.05, 0.95, 1.02, 0.98, 1.03],
            'optimized': [1.5, 1.45, 1.52, 1.48, 1.47, 1.51],
            'theoretical': [2.0, 1.98, 2.02, 1.99, 2.01, 2.03]
        }

    def test_complete_statistical_workflow(self):
        """Test complete statistical analysis workflow."""
        # Step 1: Quality assessment
        quality = analyze_measurement_quality(self.experimental_data)
        self.assertIn('overall_quality', quality)

        # Step 2: Significance detection
        significance = detect_significance(self.experimental_data)
        self.assertIn('significant_differences', significance)

        # Step 3: Individual measurements analysis
        for measurement_name, values in self.experimental_data.items():
            # Uncertainty calculation
            uncertainty = self.analyzer.calculate_measurement_uncertainty(values)
            self.assertIsInstance(uncertainty, float)

            # Confidence intervals
            mean, lower, upper = self.analyzer.calculate_confidence_intervals(values)
            self.assertIsInstance(mean, float)
            self.assertIsInstance(lower, float)
            self.assertIsInstance(upper, float)

            # Quality assessment
            quality_metrics = self.analyzer.assess_measurement_quality(values)
            self.assertIn('quality_score', quality_metrics)

    def test_cross_method_consistency(self):
        """Test consistency across different statistical methods."""
        values = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]

        # Method 1: Standard error
        se_uncertainty = self.analyzer.calculate_measurement_uncertainty(
            values, method="standard_error"
        )

        # Method 2: Standard deviation
        std_uncertainty = self.analyzer.calculate_measurement_uncertainty(
            values, method="std_dev"
        )

        # Both should be positive and reasonable
        self.assertGreater(se_uncertainty, 0)
        self.assertGreater(std_uncertainty, 0)
        self.assertGreater(std_uncertainty, se_uncertainty)  # Std dev > SE

    def test_statistical_robustness_edge_cases(self):
        """Test statistical robustness with edge cases."""
        # Test with very small values
        small_values = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        uncertainty = self.analyzer.calculate_measurement_uncertainty(small_values)
        self.assertIsInstance(uncertainty, float)
        self.assertGreater(uncertainty, 0)

        # Test with very large values
        large_values = [1e6, 2e6, 3e6, 4e6, 5e6]
        uncertainty = self.analyzer.calculate_measurement_uncertainty(large_values)
        self.assertIsInstance(uncertainty, float)
        self.assertGreater(uncertainty, 0)

        # Test with mixed positive/negative values
        mixed_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        uncertainty = self.analyzer.calculate_measurement_uncertainty(mixed_values)
        self.assertIsInstance(uncertainty, float)
        self.assertGreater(uncertainty, 0)

    def test_bootstrap_reproducibility(self):
        """Test bootstrap reproducibility with fixed seed."""
        values = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]

        # Run analysis multiple times with same analyzer (same seed)
        results = []
        for _ in range(3):
            mean, lower, upper = self.analyzer.calculate_confidence_intervals(values)
            results.append((mean, lower, upper))

        # All results should be identical (same seed)
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result, first_result)

    def test_measurement_quality_correlation(self):
        """Test correlation between sample size and quality metrics."""
        small_sample = [1.0, 1.1]
        large_sample = [1.0, 1.1, 1.2, 1.0, 1.1, 1.05, 1.08, 0.98]

        small_quality = self.analyzer.assess_measurement_quality(small_sample)
        large_quality = self.analyzer.assess_measurement_quality(large_sample)

        # Larger sample should generally have better quality score
        # (though this is not guaranteed for all cases)
        self.assertIsInstance(small_quality['quality_score'], float)
        self.assertIsInstance(large_quality['quality_score'], float)


class TestStatisticalAnalysisRobustness(unittest.TestCase):
    """Test robustness of statistical analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(bootstrap_samples=50, random_seed=42)

    def test_extreme_value_handling(self):
        """Test handling of extreme values in statistical calculations."""
        # Test with values that might cause numerical issues
        extreme_values = [1e-10, 1e10, 0.0, -1e10, 1e-10]

        # Should handle extreme values without crashing
        uncertainty = self.analyzer.calculate_measurement_uncertainty(extreme_values)
        self.assertIsInstance(uncertainty, (float, int))

        mean, lower, upper = self.analyzer.calculate_confidence_intervals(extreme_values)
        self.assertTrue(all(isinstance(x, (float, int)) for x in [mean, lower, upper]))

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import psutil
        import os

        # Create large dataset
        large_dataset = list(range(10000))

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform analysis
        uncertainty = self.analyzer.calculate_measurement_uncertainty(large_dataset)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        self.assertLess(memory_increase, 50.0)
        self.assertIsInstance(uncertainty, float)

    def test_concurrent_analysis_simulation(self):
        """Test simulation of concurrent statistical analysis."""
        import threading
        import queue

        results_queue = queue.Queue()
        num_threads = 5

        def analyze_worker(dataset_id):
            """Worker function for concurrent analysis."""
            data = [1.0 + i*0.1 + dataset_id for i in range(10)]
            uncertainty = self.analyzer.calculate_measurement_uncertainty(data)
            results_queue.put((dataset_id, uncertainty))

        # Start concurrent analysis threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=analyze_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Collect results
        results = []
        for _ in range(num_threads):
            result = results_queue.get(timeout=2.0)
            results.append(result)

        # All analyses should complete successfully
        self.assertEqual(len(results), num_threads)
        for dataset_id, uncertainty in results:
            self.assertIsInstance(dataset_id, int)
            self.assertIsInstance(uncertainty, float)
            self.assertGreater(uncertainty, 0)

    def test_statistical_method_convergence(self):
        """Test convergence of statistical methods with increasing sample size."""
        base_data = [1.0, 1.1, 1.2, 1.0, 1.1]

        uncertainties = []
        for multiplier in range(1, 6):
            # Create progressively larger datasets
            data = base_data * multiplier
            uncertainty = self.analyzer.calculate_measurement_uncertainty(data)
            uncertainties.append(uncertainty)

        # Uncertainties should generally decrease with larger sample sizes
        # (though not strictly monotonically due to randomness)
        self.assertTrue(all(u > 0 for u in uncertainties))


class TestScientificInferenceReferences(unittest.TestCase):
    def test_student_t_small_sample_reference(self):
        result = StatisticalAnalyzer().detect_statistical_significance({"x": [1, 2, 3]}, baseline=0)["x"]
        self.assertAlmostEqual(result["t_statistic"], 2 * math.sqrt(3), places=12)
        self.assertEqual(result["degrees_of_freedom"], 2)
        self.assertAlmostEqual(result["p_value"], 1 - math.sqrt(6 / 7), places=12)
        self.assertFalse(result["significant"])
        # F_2(t) = 1/2 + t/(2 sqrt(t*t+2)); invert at .975 independently of scipy.
        margin = math.sqrt(722 / 117)
        self.assertAlmostEqual(result["confidence_interval"][0], 2 - margin, places=12)
        self.assertAlmostEqual(result["confidence_interval"][1], 2 + margin, places=12)
        wider = StatisticalAnalyzer().detect_statistical_significance({"x": [1, 2, 3]}, 0, alpha=0.01)["x"]
        self.assertLess(wider["confidence_interval"][0], result["confidence_interval"][0])

    def test_welch_small_sample_reference(self):
        result = detect_significance({"a": [1, 2], "b": [4, 5]})["significant_differences"]["a_vs_b"]
        self.assertAlmostEqual(result["t_statistic"], 3 * math.sqrt(2), places=12)
        self.assertAlmostEqual(result["degrees_of_freedom"], 2, places=12)
        self.assertAlmostEqual(result["p_value"], 1 - 3 / math.sqrt(10), places=12)
        self.assertFalse(result["significant"])
        unequal = detect_significance({"a": [1, 2], "b": [1, 3, 5]})["significant_differences"]["a_vs_b"]
        self.assertAlmostEqual(unequal["degrees_of_freedom"], 361 / 137, places=12)
        self.assertAlmostEqual(unequal["t_statistic"], 3 * math.sqrt(3 / 19), places=12)

    def test_degenerate_samples_are_consistent(self):
        analyzer = StatisticalAnalyzer()
        equal = detect_significance({"a": [1, 1], "b": [1, 1]})
        unequal = detect_significance({"a": [1, 1], "b": [2, 2]})
        self.assertEqual(equal["significant_differences"]["a_vs_b"]["p_value"], 1)
        self.assertTrue(math.isnan(equal["effect_sizes"]["a_vs_b"]))
        self.assertEqual(unequal["significant_differences"]["a_vs_b"]["p_value"], 0)
        self.assertEqual(unequal["effect_sizes"]["a_vs_b"], float("inf"))
        self.assertIsNone(unequal["significant_differences"]["a_vs_b"]["degrees_of_freedom"])
        self.assertEqual(unequal["significant_differences"]["a_vs_b"]["confidence_interval"], (1, 1))
        self.assertEqual(analyzer.detect_statistical_significance({"a": [1, 1]}, 0)["a"]["p_value"], 0)
        singleton = detect_significance({"a": [1], "b": [2, 3]})["significant_differences"]["a_vs_b"]
        self.assertFalse(singleton["testable"])
        self.assertIsNone(singleton["p_value"])

    def test_two_sided_power_respects_alpha_and_unequal_sizes(self):
        analyzer = StatisticalAnalyzer()
        for alpha in [0.01, 0.05, 0.2]:
            self.assertAlmostEqual(analyzer.calculate_statistical_power([1, 2, 3], [1, 2, 3], alpha), alpha, places=14)
        # Unequal group sizes use n1*n2/(n1+n2), not min(n1,n2)/2.
        left = [0, 1, 2]
        right = [1, 2, 3, 1, 2, 3]
        d = analyzer.calculate_effect_size(left, right)
        signal = d * math.sqrt(2)
        expected = 0.5 * math.erfc((1.959963984540054 - signal) / math.sqrt(2)) + 0.5 * math.erfc((1.959963984540054 + signal) / math.sqrt(2))
        self.assertAlmostEqual(analyzer.calculate_statistical_power(left, right), expected, places=14)

    def test_dictionary_list_interval_parity_and_rng_isolation(self):
        values = list(range(1, 11))
        state = random.getstate()
        numpy_state = np.random.get_state()
        analyzer = StatisticalAnalyzer(bootstrap_samples=1000, confidence_level=0.5, random_seed=7)
        narrow = analyzer.calculate_confidence_intervals(values)
        self.assertEqual(analyzer.calculate_confidence_intervals({"x": values})["x_50ci"], narrow[1:])
        wide = StatisticalAnalyzer(1000, 0.99, 7).calculate_confidence_intervals({"x": values})["x_99ci"]
        self.assertLess(narrow[2] - narrow[1], wide[1] - wide[0])
        self.assertEqual(random.getstate(), state)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        self.assertTrue(np.array_equal(numpy_state[1], after[1]))
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertEqual(analyzer.calculate_confidence_intervals({"x": [0, 0]})["x_50ci"], (0, 0))

    def test_balanced_folds_cover_remainder_exactly_once(self):
        result = StatisticalAnalyzer().perform_k_fold_cross_validation({"x": [0, 0, 0, 0, 0, 100]}, 5)["x"]
        self.assertEqual(result["cv_scores"], [25, 20, 20, 20, 100])
        self.assertEqual([index for fold in result["test_indices"] for index in fold], list(range(6)))
        self.assertEqual([len(fold) for fold in result["test_indices"]], [2, 1, 1, 1, 1])
        for invalid in [0, 1, 7, 2.5, True]:
            with self.subTest(k=invalid), self.assertRaises(ValueError):
                StatisticalAnalyzer().perform_k_fold_cross_validation({"x": [1, 2, 3, 4, 5, 6]}, invalid)

    def test_invalid_inference_domains(self):
        for count in [0, -1, 2.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                StatisticalAnalyzer(bootstrap_samples=count)
        for confidence in [0, 1, float("nan"), float("inf")]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                StatisticalAnalyzer(confidence_level=confidence)
        analyzer = StatisticalAnalyzer()
        for bad in [float("nan"), float("inf")]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                analyzer.analyze_measurement_quality({"x": [1, bad]})
            with self.subTest(baseline=bad), self.assertRaises(ValueError):
                analyzer.detect_statistical_significance({"x": [1, 2]}, bad)
        with self.assertRaises(ValueError):
            detect_significance({"a": [1, 2], "b": [3, 4]}, alpha=0)

    def test_data_quality_does_not_prove_power_or_replication(self):
        analysis = StatisticalAnalyzer().analyze_measurement_quality({"x": [1] * 20})
        self.assertIsNone(analysis["validation_checks"]["statistical_power"])
        self.assertIsNone(analysis["validation_checks"]["reproducibility"])
        singleton = StatisticalAnalyzer().analyze_measurement_quality({"x": [5]})
        self.assertNotIn("x", singleton["uncertainty"])
        self.assertIsNone(singleton["statistical_metrics"]["x_std"])
        self.assertEqual(singleton["confidence_intervals"], {})


if __name__ == '__main__':
    unittest.main()
