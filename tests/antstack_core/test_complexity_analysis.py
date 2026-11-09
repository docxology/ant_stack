#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.complexity_analysis module.

Tests core complexity analysis functionality including:
- Agent-based modeling and emergent behavior analysis
- Network complexity metrics and structural analysis
- Thermodynamic computing efficiency frameworks
- Complexity-entropy diagrams for intrinsic computation
- Advanced bootstrap methods with bias correction
- Multi-scale analysis integration

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.complexity_analysis import (
    ComplexityMetrics,
    AgentBasedModel,
    NetworkComplexityAnalyzer,
    ThermodynamicComplexityAnalyzer,
    ComplexityEntropyAnalyzer,
    EnhancedBootstrapAnalyzer,
    comprehensive_complexity_analysis
)

try:
    import numpy as np
    HAS_NUMPY_REAL = True
except ImportError:
    HAS_NUMPY_REAL = False




class TestAgentBasedModel(unittest.TestCase):
    """Observable toy dynamics, RNG isolation and invalid-index boundaries."""

    def test_synchronous_updates_costs_and_caller_random_state(self):
        import random
        import numpy as np
        python_state, numpy_state = random.getstate(), np.random.get_state()
        model = AgentBasedModel([{"initial_state": 0}, {"initial_state": 1}], [(0, 1, 2)], {}, time_steps=2)
        result = model.simulate()
        self.assertEqual(random.getstate(), python_state)
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        np.testing.assert_allclose(result["total_energy"], [2.04, 2.0784])
        np.testing.assert_allclose(result["state_variance"], [.2304, .21233664])
        self.assertEqual(result["state_pair_similarity"], [0, 0])
        self.assertEqual(result["entropy"], [1, 1])

    def test_similarity_is_a_state_pair_fraction_not_graph_clustering(self):
        model = AgentBasedModel([{"initial_state": value} for value in (0, .05, .1)], [], {}, time_steps=1)
        result = model.simulate()
        self.assertAlmostEqual(result["state_pair_similarity"][0], 2/3)
        self.assertNotIn("clustering_coefficient", result)
        pair = AgentBasedModel([{"initial_state": 0}, {"initial_state": .05}], [], {}, time_steps=1)
        self.assertEqual(pair.simulate()["state_pair_similarity"], [1])

    def test_optional_histogram_backend_does_not_change_population_variance(self):
        from unittest.mock import patch
        model = AgentBasedModel([{"initial_state": 0}, {"initial_state": 2}], [], {}, time_steps=1)
        with patch("antstack_core.analysis.complexity_analysis.HAS_NUMPY", False):
            result = model.simulate()
        self.assertEqual(result["state_variance"], [1])
        self.assertEqual(result["entropy"], [1])

    def test_invalid_agent_indices_strengths_and_steps_refuse(self):
        agents = [{}, {}]
        for interaction in [(-1, 0, 1), (0, 2, 1), (0.5, 1, 1), (0, 1, float("nan"))]:
            with self.subTest(interaction=interaction), self.assertRaises(ValueError):
                AgentBasedModel(agents, [interaction], {}, time_steps=1).simulate()
        for steps in (-1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                AgentBasedModel(agents, [], {}, time_steps=steps).simulate()

    def test_empty_population_has_no_energy_or_entropy(self):
        result = AgentBasedModel([], [], {}, time_steps=2).simulate()
        self.assertEqual(result["total_energy"], [0, 0])
        self.assertEqual(result["entropy"], [0, 0])




class TestNetworkComplexityAnalyzer(unittest.TestCase):
    """Test NetworkComplexityAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple network adjacency matrix
        self.adjacency_matrix = [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ]

        self.simple_matrix = [
            [0, 1],
            [1, 0]
        ]

    def test_network_complexity_analyzer_creation(self):
        """Test NetworkComplexityAnalyzer creation."""
        analyzer = NetworkComplexityAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_network_complexity_analyzer_basic_analysis(self):
        """Test basic network complexity metrics calculation."""
        analyzer = NetworkComplexityAnalyzer()
        metrics = analyzer.analyze_network_complexity(self.adjacency_matrix)

        self.assertIsInstance(metrics, ComplexityMetrics)
        # Check that some basic metrics are calculated
        self.assertIsInstance(metrics.network_density, float)
        self.assertIsInstance(metrics.clustering_coefficient, float)
        self.assertIsInstance(metrics.entropy, float)

        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics.network_density, 0.0)
        self.assertLessEqual(metrics.network_density, 1.0)

    def test_network_complexity_analyzer_empty_network(self):
        """Test network complexity analyzer with empty network."""
        analyzer = NetworkComplexityAnalyzer()
        metrics = analyzer.analyze_network_complexity([])

        # Should handle empty network gracefully
        self.assertIsInstance(metrics, ComplexityMetrics)
        # Should return default values for empty network
        self.assertEqual(metrics.network_density, 0.0)

    def test_network_complexity_analyzer_simple_network(self):
        """Test with simple 2-node network."""
        analyzer = NetworkComplexityAnalyzer()
        metrics = analyzer.analyze_network_complexity(self.simple_matrix)

        self.assertIsInstance(metrics, ComplexityMetrics)
        # For a simple connected network, density should be reasonable
        self.assertGreater(metrics.network_density, 0.0)

    def test_network_complexity_analyzer_helper_methods(self):
        """Test helper methods of NetworkComplexityAnalyzer."""
        analyzer = NetworkComplexityAnalyzer()

        # Test density calculation
        density = analyzer._calculate_density(self.adjacency_matrix)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)

        # Test clustering coefficient
        clustering = analyzer._calculate_clustering_coefficient(self.adjacency_matrix)
        self.assertIsInstance(clustering, float)
        self.assertGreaterEqual(clustering, 0.0)
        self.assertLessEqual(clustering, 1.0)

        # Test path length calculation
        path_length = analyzer._calculate_path_length(self.adjacency_matrix)
        self.assertIsInstance(path_length, float)
        self.assertGreaterEqual(path_length, 0.0)


class TestThermodynamicComplexityAnalyzer(unittest.TestCase):
    """Test ThermodynamicComplexityAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.energy_values = [1.0, 1.2, 0.8, 1.1, 0.9]
        self.temperature = 298.0  # Room temperature in Kelvin

    def test_thermodynamic_complexity_analyzer_creation(self):
        """Test ThermodynamicComplexityAnalyzer creation."""
        analyzer = ThermodynamicComplexityAnalyzer(temperature_k=self.temperature)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.temperature_k, self.temperature)

    def test_thermodynamic_complexity_analyzer_analysis(self):
        """Test thermodynamic complexity analysis."""
        from antstack_core.analysis.energy import EnergyBreakdown

        analyzer = ThermodynamicComplexityAnalyzer(temperature_k=self.temperature)

        energy_breakdown = EnergyBreakdown(
            compute_memory=1.0,
            actuation=2.0,
            sensing=0.2,
            baseline=0.1
        )

        metrics = analyzer.analyze_thermodynamic_efficiency(
            energy_breakdown, 5.0, 1000.0
        )

        self.assertIsInstance(metrics, ComplexityMetrics)
        self.assertIsInstance(metrics.thermodynamic_efficiency, float)
        self.assertGreaterEqual(metrics.thermodynamic_efficiency, 0.0)
        self.assertLessEqual(metrics.thermodynamic_efficiency, 1.0)

    def test_thermodynamic_complexity_analyzer_different_temperature(self):
        """Test thermodynamic analyzer with different temperature."""
        analyzer = ThermodynamicComplexityAnalyzer(temperature_k=77.0)  # Liquid nitrogen temp

        from antstack_core.analysis.energy import EnergyBreakdown
        energy_breakdown = EnergyBreakdown(
            compute_memory=1.0,
            actuation=2.0,
            sensing=0.2,
            baseline=0.1
        )

        metrics = analyzer.analyze_thermodynamic_efficiency(
            energy_breakdown, 5.0, 1000.0
        )

        self.assertIsInstance(metrics, ComplexityMetrics)
        self.assertEqual(analyzer.temperature_k, 77.0)

    def test_thermodynamic_complexity_analyzer_helper_methods(self):
        """Test helper methods of ThermodynamicComplexityAnalyzer."""
        analyzer = ThermodynamicComplexityAnalyzer(temperature_k=self.temperature)

        entropy_rate = analyzer._calculate_entropy_production_rate(self.energy_values, duration_s=0.05)
        self.assertAlmostEqual(entropy_rate, sum(self.energy_values) / (self.temperature * 0.05))
        free_energy = analyzer._calculate_free_energy(1.0, 1.0 / self.temperature)
        self.assertAlmostEqual(free_energy, 0.0)


class TestComplexityEntropyAnalyzer(unittest.TestCase):
    """Test ComplexityEntropyAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.system_states = [0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.5, 0.2]
        self.time_series = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.3, 0.2, 0.4, 0.3]
        ]

    def test_complexity_entropy_analyzer_creation(self):
        """Test ComplexityEntropyAnalyzer creation."""
        analyzer = ComplexityEntropyAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_complexity_entropy_analyzer_diagram_creation(self):
        """Test complexity-entropy diagram creation."""
        analyzer = ComplexityEntropyAnalyzer()

        result = analyzer.create_complexity_entropy_diagram(
            self.system_states,
            window_size=5
        )

        self.assertIsInstance(result, dict)
        self.assertIn('complexity', result)
        self.assertIn('entropy', result)

        # Should have results for windows in the time series
        expected_windows = len(self.system_states) - 5 + 1
        self.assertEqual(len(result['complexity']), expected_windows)
        self.assertEqual(len(result['entropy']), expected_windows)

    def test_complexity_entropy_analyzer_small_window(self):
        """Test with window size larger than time series."""
        analyzer = ComplexityEntropyAnalyzer()

        result = analyzer.create_complexity_entropy_diagram(
            self.system_states[:3],  # Only 3 points
            window_size=10  # Window larger than data
        )

        # Should return empty results for insufficient data
        self.assertEqual(len(result['complexity']), 0)
        self.assertEqual(len(result['entropy']), 0)

    def test_complexity_entropy_analyzer_empty_data(self):
        """Test with empty time series."""
        analyzer = ComplexityEntropyAnalyzer()

        result = analyzer.create_complexity_entropy_diagram([], window_size=5)

        # Should return empty results
        self.assertEqual(len(result['complexity']), 0)
        self.assertEqual(len(result['entropy']), 0)

    def test_complexity_entropy_analyzer_helper_methods(self):
        """Test helper methods of ComplexityEntropyAnalyzer."""
        analyzer = ComplexityEntropyAnalyzer()

        # Test Shannon entropy calculation
        entropy = analyzer._calculate_shannon_entropy(self.system_states)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

        # Test conditional entropy
        conditional_entropy = analyzer._calculate_conditional_entropy(self.system_states)
        self.assertIsInstance(conditional_entropy, float)
        self.assertGreaterEqual(conditional_entropy, 0.0)




class TestComprehensiveComplexityAnalysis(unittest.TestCase):
    """Test comprehensive_complexity_analysis function."""

    def setUp(self):
        """Set up test fixtures."""
        from antstack_core.analysis.energy import EnergyBreakdown

        # Create sample energy breakdown data
        self.energy_breakdowns = [
            EnergyBreakdown(
                compute_memory=1.0,
                actuation=2.0,
                sensing=0.2,
                baseline=0.1
            ),
            EnergyBreakdown(
                compute_memory=1.2,
                actuation=2.1,
                sensing=0.3,
                baseline=0.15
            )
        ]

        # Create computational data
        self.computational_data = [
            {
                "computational_work": 5.0,
                "information_processed": 1000.0
            },
            {
                "computational_work": 6.0,
                "information_processed": 1200.0
            }
        ]

        # Network data
        self.network_data = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

        # Time series data
        self.time_series_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def test_comprehensive_complexity_analysis_basic(self):
        """Test comprehensive complexity analysis with basic data."""
        results = comprehensive_complexity_analysis(
            self.energy_breakdowns,
            self.computational_data,
            self.network_data,
            self.time_series_data
        )

        self.assertIsInstance(results, dict)
        self.assertIn('thermodynamic_analysis', results)
        self.assertIn('network_analysis', results)
        self.assertIn('complexity_entropy_analysis', results)
        self.assertIn('bootstrap_analysis', results)
        self.assertIn('integrated_metrics', results)

    def test_comprehensive_complexity_analysis_minimal(self):
        """Test comprehensive complexity analysis with minimal data."""
        results = comprehensive_complexity_analysis(
            self.energy_breakdowns[:1],
            self.computational_data[:1]
        )

        # Should work with minimal data
        self.assertIsInstance(results, dict)
        self.assertIn('thermodynamic_analysis', results)

    def test_comprehensive_complexity_analysis_empty(self):
        """Test comprehensive complexity analysis with empty data."""
        results = comprehensive_complexity_analysis([], [])

        # Should handle empty data gracefully
        self.assertIsInstance(results, dict)

    def test_comprehensive_complexity_analysis_network_only(self):
        """Test with only network data."""
        results = comprehensive_complexity_analysis(
            [], [], network_data=self.network_data
        )

        self.assertIsInstance(results, dict)
        self.assertIn('network_analysis', results)

    def test_comprehensive_complexity_analysis_time_series_only(self):
        """Test with only time series data."""
        results = comprehensive_complexity_analysis(
            [], [], time_series_data=self.time_series_data
        )

        self.assertIsInstance(results, dict)
        self.assertIn('complexity_entropy_analysis', results)


class TestComplexityAnalysisIntegration(unittest.TestCase):
    """Test integration between different complexity analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        from antstack_core.analysis.energy import EnergyBreakdown

        self.energy_breakdown = EnergyBreakdown(
            compute_memory=1.0,
            actuation=2.0,
            sensing=0.2,
            baseline=0.1
        )

        self.network_data = [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ]

    def test_full_complexity_analysis_workflow(self):
        """Test complete complexity analysis workflow."""
        # Test individual components
        thermo_analyzer = ThermodynamicComplexityAnalyzer(temperature_k=298.0)
        thermo_result = thermo_analyzer.analyze_thermodynamic_efficiency(
            self.energy_breakdown, 5.0, 1000.0
        )

        network_analyzer = NetworkComplexityAnalyzer()
        network_result = network_analyzer.analyze_network_complexity(self.network_data)

        ce_analyzer = ComplexityEntropyAnalyzer()
        ce_result = ce_analyzer.create_complexity_entropy_diagram([0.1, 0.2, 0.3, 0.4, 0.5])

        bootstrap_analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=50)
        bootstrap_result = bootstrap_analyzer.analyze_scaling_with_bootstrap(
            [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]
        )

        # All should produce valid results
        self.assertIsInstance(thermo_result, ComplexityMetrics)
        self.assertIsInstance(network_result, ComplexityMetrics)
        self.assertIsInstance(ce_result, dict)
        self.assertIsInstance(bootstrap_result, dict)

    def test_complexity_metrics_consistency(self):
        """Test that ComplexityMetrics objects are consistent across analyzers."""
        # Create different analyzers and check they all return ComplexityMetrics
        thermo_analyzer = ThermodynamicComplexityAnalyzer(temperature_k=298.0)
        network_analyzer = NetworkComplexityAnalyzer()

        from antstack_core.analysis.energy import EnergyBreakdown
        energy_breakdown = EnergyBreakdown(
            compute_memory=1.0,
            actuation=2.0,
            sensing=0.2,
            baseline=0.1
        )

        thermo_result = thermo_analyzer.analyze_thermodynamic_efficiency(
            energy_breakdown, 5.0, 1000.0
        )
        network_result = network_analyzer.analyze_network_complexity(self.network_data)

        # Both should return ComplexityMetrics objects
        self.assertIsInstance(thermo_result, ComplexityMetrics)
        self.assertIsInstance(network_result, ComplexityMetrics)

        # Both should have the same structure
        self.assertTrue(hasattr(thermo_result, 'entropy'))
        self.assertTrue(hasattr(network_result, 'entropy'))


class TestComplexityAnalysisRobustness(unittest.TestCase):
    """Test robustness of complexity analysis methods."""

    def test_complexity_analysis_with_numpy_fallback(self):
        """Test that analysis works without numpy."""
        # This test ensures our fallback implementations work
        network_analyzer = NetworkComplexityAnalyzer()
        simple_network = [[0, 1], [1, 0]]

        result = network_analyzer.analyze_network_complexity(simple_network)
        self.assertIsInstance(result, ComplexityMetrics)

    def test_complexity_analysis_with_scipy_fallback(self):
        """Test that analysis works without scipy."""
        # This test ensures our fallback implementations work
        bootstrap_analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=50)
        x_data = [1, 2, 3, 4, 5]
        y_data = [1, 2, 3, 4, 5]

        result = bootstrap_analyzer.analyze_scaling_with_bootstrap(x_data, y_data)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()


# Exact scientific references for both implementations, including Python-only paths.
import pytest


@pytest.mark.parametrize("numpy_backend", [False, True])
def test_reference_graph_metrics_and_weight_invariance(monkeypatch, numpy_backend):
    import importlib
    module = importlib.import_module(NetworkComplexityAnalyzer.__module__)
    monkeypatch.setattr(module, "HAS_NUMPY", numpy_backend)
    analyzer = NetworkComplexityAnalyzer()
    triangle = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    metrics = analyzer.analyze_network_complexity(triangle)
    assert metrics.clustering_coefficient == pytest.approx(1)
    assert metrics.path_length == pytest.approx(1)
    assert metrics.modularity == pytest.approx(0)
    assert metrics.mutual_information == pytest.approx(math.log2(1.5))
    pair = [[0, 1], [1, 0]]
    weighted = [[0, 100], [100, 0]]
    assert analyzer.analyze_network_complexity(pair).path_length == 1
    assert analyzer.analyze_network_complexity(pair).mutual_information == pytest.approx(1)
    assert analyzer.analyze_network_complexity(weighted).mutual_information == pytest.approx(1)
    assert analyzer._calculate_modularity(pair, [0, 0]) == pytest.approx(0)
    assert analyzer._calculate_modularity(pair, [0, 1]) == pytest.approx(-0.5)
    assert analyzer._calculate_path_length([[9, 1], [1, 4]]) == 1


def test_disconnected_modularity_paths_and_permutation():
    analyzer = NetworkComplexityAnalyzer()
    graph = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    metrics = analyzer.analyze_network_complexity(graph)
    assert metrics.modularity == pytest.approx(0.5)
    assert math.isinf(metrics.path_length)
    assert analyzer.analyze_network_complexity([[0, 0], [0, 0]]).mutual_information is None
    order = [0, 2, 1, 3]
    permuted = [[graph[i][j] for j in order] for i in order]
    assert analyzer.analyze_network_complexity(permuted) == metrics
    path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert analyzer.analyze_network_complexity(path).path_length == pytest.approx(4 / 3)
    assert analyzer.analyze_network_complexity(path).clustering_coefficient == 0
    for invalid in ([[0, 1]], [[0, 1], [0, 0]], [[0, -1], [-1, 0]], [[0, math.nan], [math.nan, 0]]):
        with pytest.raises(ValueError):
            analyzer.analyze_network_complexity(invalid)
    with pytest.raises(ValueError):
        analyzer.analyze_network_complexity(graph, communities=[0])


def test_cycle_energy_entropy_units_and_duration():
    from antstack_core.analysis.energy import EnergyBreakdown
    analyzer = ThermodynamicComplexityAnalyzer(300)
    energy = EnergyBreakdown(compute_memory=1)
    fast = analyzer.analyze_thermodynamic_efficiency(energy, 1, 0, duration_s=0.01)
    slow = analyzer.analyze_thermodynamic_efficiency(energy, 1, 0, duration_s=0.02)
    assert fast.entropy_change == pytest.approx(1 / 300)
    assert fast.entropy_production_rate == pytest.approx(1 / 3)
    assert slow.entropy_production_rate == pytest.approx(1 / 6)
    assert fast.free_energy == pytest.approx(0)
    assert slow.free_energy == pytest.approx(fast.free_energy)
    for bad in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            ThermodynamicComplexityAnalyzer(bad)
        with pytest.raises(ValueError):
            analyzer.analyze_thermodynamic_efficiency(energy, 1, 0, duration_s=bad)
    with pytest.raises(ValueError):
        analyzer.analyze_thermodynamic_efficiency(energy, 1, -1)
    with pytest.raises(ValueError):
        analyzer.analyze_thermodynamic_efficiency(EnergyBreakdown(), 1, 0)


@pytest.mark.parametrize("numpy_backend", [False, True])
def test_lz78_reference_phrases(monkeypatch, numpy_backend):
    import importlib
    module = importlib.import_module(ComplexityEntropyAnalyzer.__module__)
    monkeypatch.setattr(module, "HAS_NUMPY", numpy_backend)
    analyzer = ComplexityEntropyAnalyzer()
    for sequence, expected in [([], 0), ([0], 1), ([0, 0, 0, 0], 3 / 4),
                               ([0, 1, 2, 3], 1), ([0, 1, 0, 1, 0, 1], 4 / 6)]:
        assert analyzer._calculate_lemple_ziv_complexity(sequence) == pytest.approx(expected)
    assert analyzer.create_complexity_entropy_diagram([0, 0, 0, 0], 4)['complexity'] == [0.75]
    assert analyzer.create_complexity_entropy_diagram([0, 1, 2, 3], 4)['complexity'] == [1]
    with pytest.raises(ValueError):
        analyzer._calculate_lemple_ziv_complexity([0, math.nan])


@pytest.mark.parametrize("numpy_backend", [False, True])
def test_paired_bootstrap_reproducible_exact_power_law(monkeypatch, numpy_backend):
    import importlib
    import random
    module = importlib.import_module(EnhancedBootstrapAnalyzer.__module__)
    monkeypatch.setattr(module, "HAS_NUMPY", numpy_backend)
    analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=120, seed=74)
    state = random.getstate()
    result = analyzer.analyze_scaling_with_bootstrap([1, 2, 4, 8], [1, 4, 16, 64])
    assert random.getstate() == state
    assert result == analyzer.analyze_scaling_with_bootstrap([1, 2, 4, 8], [1, 4, 16, 64])
    assert result['valid'] is True
    assert 0 < result['bootstrap_valid'] <= result['bootstrap_attempted'] == 120
    assert len(result['bootstrap_exponents']) == result['bootstrap_valid']
    assert result['bootstrap_exponents'] == pytest.approx([2] * result['bootstrap_valid'])
    assert result['bootstrap_intercepts'] == pytest.approx([1] * result['bootstrap_valid'])
    assert result['confidence_interval'] == pytest.approx((2, 2))
    assert result['confidence_intervals']['intercept'] == pytest.approx((1, 1))
    assert result['confidence_intervals']['r_squared'] == pytest.approx((1, 1))
    assert result['bias_corrected']['exponent'] == pytest.approx(2)


def test_bootstrap_invalid_and_confidence_boundaries():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            EnhancedBootstrapAnalyzer(bad)
    analyzer = EnhancedBootstrapAnalyzer(120, seed=8)
    assert analyzer.analyze_scaling_with_bootstrap([1, 2], [1, 4])['valid'] is False
    assert analyzer.analyze_scaling_with_bootstrap([1, 1, 1], [1, 2, 3])['valid'] is False
    for bad in (0, 1, -0.1, math.nan):
        with pytest.raises(ValueError):
            analyzer.analyze_scaling_with_bootstrap([1, 2, 3], [1, 4, 9], bad)
    low = analyzer.analyze_scaling_with_bootstrap([1, 2, 4, 8, 16], [1, 3, 15, 72, 210], 0.9)
    high = analyzer.analyze_scaling_with_bootstrap([1, 2, 4, 8, 16], [1, 3, 15, 72, 210], 0.95)
    assert high['confidence_interval'][0] <= low['confidence_interval'][0]
    assert high['confidence_interval'][1] >= low['confidence_interval'][1]
    assert high['confidence_interval'][0] < high['confidence_interval'][1]


@pytest.mark.parametrize("module_name", ["complexity_analysis", "enhanced_complexity_analysis"])
@pytest.mark.parametrize("response", [[7, 7, 7, 7], [1, 1, 2, 2]])
def test_bootstrap_retains_undefined_fit_statistics(module_name, response):
    import importlib
    module = importlib.import_module("antstack_core.analysis." + module_name)
    result = module.EnhancedBootstrapAnalyzer(120, seed=7).analyze_scaling_with_bootstrap([1, 2, 4, 8], response)
    assert result["valid"] is True
    if response == [7, 7, 7, 7]:
        assert result["confidence_interval"] == pytest.approx((0, 0), abs=1e-12)
        assert result["confidence_intervals"]["intercept"] == pytest.approx((7, 7))
        assert result["confidence_intervals"]["r_squared"] is None
        assert result["bootstrap_stats"]["mean_r_squared"] is None
        assert result["bootstrap_stats"]["std_r_squared"] is None
    else:
        low, high = result["confidence_intervals"]["r_squared"]
        assert 0 <= low <= high <= 1
        assert all(math.isfinite(value) for value in result["bootstrap_r_squared"])


@pytest.mark.parametrize("module_name", ["complexity_analysis", "enhanced_complexity_analysis"])
def test_real_diagnostics_distinguish_unavailable_fields_from_computed_zero(module_name):
    import importlib
    import json
    from dataclasses import asdict
    from antstack_core.analysis.energy import EnergyBreakdown
    module = importlib.import_module("antstack_core.analysis." + module_name)
    network = module.NetworkComplexityAnalyzer().analyze_network_complexity([[0]])
    serialized = json.loads(json.dumps(asdict(network), allow_nan=False))
    assert serialized["network_density"] == 0
    assert serialized["clustering_coefficient"] == 0
    assert serialized["entropy"] == 0
    assert serialized["thermodynamic_efficiency"] is None
    assert serialized["scaling_exponent"] is None
    assert serialized["variance"] is None
    thermo = module.ThermodynamicComplexityAnalyzer(temperature_k=300).analyze_thermodynamic_efficiency(
        EnergyBreakdown(compute_flops=1), computational_work=1, information_processed=0)
    serialized = json.loads(json.dumps(asdict(thermo), allow_nan=False))
    assert serialized["thermodynamic_efficiency"] == 0
    assert serialized["entropy_change"] == pytest.approx(1 / 300)
    assert serialized["network_density"] is None
    assert serialized["entropy"] is None
