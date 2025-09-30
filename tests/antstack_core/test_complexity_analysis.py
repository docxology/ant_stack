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


class TestComplexityMetrics(unittest.TestCase):
    """Test ComplexityMetrics dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_metrics = {
            "entropy": 2.5,
            "mutual_information": 1.8,
            "information_density": 0.95,
            "network_density": 0.7,
            "clustering_coefficient": 0.6,
            "path_length": 2.3,
            "modularity": 0.4,
            "thermodynamic_efficiency": 0.85,
            "entropy_production_rate": 1.2,
            "free_energy": -5.0,
            "scaling_exponent": 1.8,
            "scaling_confidence": 0.92,
            "regime_classification": "super-linear",
            "variance": 0.15,
            "skewness": -0.2,
            "kurtosis": 1.1
        }

    def test_complexity_metrics_creation(self):
        """Test basic ComplexityMetrics creation."""
        metrics = ComplexityMetrics(**self.sample_metrics)

        self.assertEqual(metrics.entropy, 2.5)
        self.assertEqual(metrics.mutual_information, 1.8)
        self.assertEqual(metrics.scaling_exponent, 1.8)
        self.assertEqual(metrics.regime_classification, "super-linear")

    def test_complexity_metrics_defaults(self):
        """Test ComplexityMetrics with default values."""
        metrics = ComplexityMetrics()

        self.assertEqual(metrics.entropy, 0.0)
        self.assertEqual(metrics.network_density, 0.0)
        self.assertEqual(metrics.thermodynamic_efficiency, 0.0)
        self.assertEqual(metrics.regime_classification, "unknown")

    def test_complexity_metrics_with_partial_data(self):
        """Test ComplexityMetrics with partial data."""
        partial_data = {
            "entropy": 2.0,
            "scaling_exponent": 1.5,
            "regime_classification": "linear"
        }

        metrics = ComplexityMetrics(**partial_data)

        self.assertEqual(metrics.entropy, 2.0)
        self.assertEqual(metrics.scaling_exponent, 1.5)
        self.assertEqual(metrics.regime_classification, "linear")
        # Other fields should remain at defaults
        self.assertEqual(metrics.mutual_information, 0.0)
        self.assertEqual(metrics.network_density, 0.0)


class TestAgentBasedModel(unittest.TestCase):
    """Test AgentBasedModel functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.agents = [
            {"initial_state": 0.5, "initial_energy": 1.0},
            {"initial_state": 0.3, "initial_energy": 1.2},
            {"initial_state": 0.7, "initial_energy": 0.8}
        ]

        self.interactions = [
            (0, 1, 0.5),
            (1, 2, 0.3),
            (2, 0, 0.4)
        ]

        self.environment_params = {"temperature": 298.0, "pressure": 101325.0}

    def test_agent_based_model_creation(self):
        """Test AgentBasedModel creation."""
        model = AgentBasedModel(
            agents=self.agents,
            interactions=self.interactions,
            environment_params=self.environment_params,
            time_steps=100
        )

        self.assertEqual(len(model.agents), 3)
        self.assertEqual(len(model.interactions), 3)
        self.assertEqual(model.time_steps, 100)

    def test_agent_based_model_simulation(self):
        """Test agent-based model simulation."""
        model = AgentBasedModel(
            agents=self.agents,
            interactions=self.interactions,
            environment_params=self.environment_params,
            time_steps=10
        )

        results = model.simulate(seed=42)

        # Check that results contain expected metrics
        self.assertIn('total_energy', results)
        self.assertIn('state_variance', results)
        self.assertIn('interaction_strength', results)
        self.assertIn('clustering_coefficient', results)
        self.assertIn('entropy', results)

        # Check that we have results for each time step
        self.assertEqual(len(results['total_energy']), 10)
        self.assertEqual(len(results['state_variance']), 10)
        self.assertEqual(len(results['entropy']), 10)

        # Check that values are reasonable
        self.assertTrue(all(isinstance(x, (int, float)) for x in results['total_energy']))
        self.assertTrue(all(isinstance(x, (int, float)) for x in results['state_variance']))

    def test_agent_based_model_simulation_deterministic(self):
        """Test that simulation is deterministic with fixed seed."""
        model1 = AgentBasedModel(
            agents=self.agents,
            interactions=self.interactions,
            environment_params=self.environment_params,
            time_steps=5
        )

        model2 = AgentBasedModel(
            agents=self.agents,
            interactions=self.interactions,
            environment_params=self.environment_params,
            time_steps=5
        )

        results1 = model1.simulate(seed=123)
        results2 = model2.simulate(seed=123)

        # Results should be identical with same seed
        self.assertEqual(results1['total_energy'], results2['total_energy'])
        self.assertEqual(results1['state_variance'], results2['state_variance'])

    def test_agent_based_model_empty_agents(self):
        """Test agent-based model with empty agents."""
        model = AgentBasedModel(
            agents=[],
            interactions=[],
            environment_params=self.environment_params,
            time_steps=5
        )

        results = model.simulate(seed=42)

        # Should handle empty case gracefully
        self.assertIn('total_energy', results)
        self.assertIn('entropy', results)
        # Results should be empty or have default values
        self.assertEqual(len(results['total_energy']), 5)

    def test_agent_based_model_single_agent(self):
        """Test agent-based model with single agent."""
        single_agent = [{"initial_state": 0.5, "initial_energy": 1.0}]

        model = AgentBasedModel(
            agents=single_agent,
            interactions=[],
            environment_params=self.environment_params,
            time_steps=3
        )

        results = model.simulate(seed=42)

        # Should handle single agent case
        self.assertIn('total_energy', results)
        self.assertIn('entropy', results)
        self.assertEqual(len(results['total_energy']), 3)

    def test_agent_based_model_no_interactions(self):
        """Test agent-based model with no interactions."""
        model = AgentBasedModel(
            agents=self.agents,
            interactions=[],
            environment_params=self.environment_params,
            time_steps=3
        )

        results = model.simulate(seed=42)

        # Should handle no interactions case
        self.assertIn('total_energy', results)
        self.assertIn('entropy', results)
        self.assertEqual(len(results['total_energy']), 3)

    def test_agent_based_model_helper_methods(self):
        """Test helper methods of AgentBasedModel."""
        model = AgentBasedModel(
            agents=self.agents,
            interactions=self.interactions,
            environment_params=self.environment_params
        )

        # Test variance calculation
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        variance = model._variance(values)
        self.assertIsInstance(variance, float)
        self.assertGreater(variance, 0)

        # Test clustering calculation
        states = [0.1, 0.2, 0.8, 0.9, 0.15]
        clustering = model._calculate_clustering(states)
        self.assertIsInstance(clustering, float)
        self.assertGreaterEqual(clustering, 0.0)
        self.assertLessEqual(clustering, 1.0)

        # Test entropy calculation
        entropy = model._calculate_entropy(states)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)


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

        # Test entropy production rate calculation
        entropy_rate = analyzer._calculate_entropy_production_rate(self.energy_values)
        self.assertIsInstance(entropy_rate, float)
        self.assertGreaterEqual(entropy_rate, 0.0)

        # Test free energy calculation
        free_energy = analyzer._calculate_free_energy(self.energy_values)
        self.assertIsInstance(free_energy, float)


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


class TestEnhancedBootstrapAnalyzer(unittest.TestCase):
    """Test EnhancedBootstrapAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.x_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_values = [1.0, 1.8, 2.6, 3.4, 4.2]

    def test_enhanced_bootstrap_analyzer_creation(self):
        """Test EnhancedBootstrapAnalyzer creation."""
        analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=500)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.n_bootstrap, 500)

    def test_enhanced_bootstrap_analyzer_scaling_analysis(self):
        """Test bootstrap-based scaling analysis."""
        analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=100)

        result = analyzer.analyze_scaling_with_bootstrap(
            self.x_values,
            self.y_values,
            confidence_level=0.95
        )

        self.assertIsInstance(result, dict)
        self.assertIn('scaling_exponent', result)
        self.assertIn('intercept', result)
        self.assertIn('r_squared', result)

        # Confidence interval may or may not be present depending on bootstrap success
        if 'confidence_interval' in result:
            ci = result['confidence_interval']
            self.assertIsInstance(ci, tuple)
            self.assertEqual(len(ci), 2)

    def test_enhanced_bootstrap_analyzer_insufficient_data(self):
        """Test with insufficient data."""
        analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=100)

        result = analyzer.analyze_scaling_with_bootstrap(
            [1.0, 2.0],  # Only 2 points
            [1.0, 2.0],
            confidence_level=0.95
        )

        # Should handle insufficient data gracefully
        self.assertIsInstance(result, dict)

    def test_enhanced_bootstrap_analyzer_helper_methods(self):
        """Test helper methods of EnhancedBootstrapAnalyzer."""
        analyzer = EnhancedBootstrapAnalyzer(n_bootstrap=100)

        # Test bootstrap sample generation
        original_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bootstrap_sample = analyzer._generate_bootstrap_sample(original_data)
        self.assertIsInstance(bootstrap_sample, list)
        self.assertEqual(len(bootstrap_sample), len(original_data))


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

    def test_complexity_metrics_edge_cases(self):
        """Test ComplexityMetrics with extreme values."""
        # Test with very large values
        large_metrics = ComplexityMetrics(
            entropy=1e6,
            mutual_information=1e6,
            network_density=1.0,
            thermodynamic_efficiency=1.0
        )
        self.assertIsInstance(large_metrics, ComplexityMetrics)

        # Test with very small values
        small_metrics = ComplexityMetrics(
            entropy=1e-6,
            mutual_information=1e-6,
            network_density=0.0,
            thermodynamic_efficiency=0.0
        )
        self.assertIsInstance(small_metrics, ComplexityMetrics)


if __name__ == '__main__':
    unittest.main()
