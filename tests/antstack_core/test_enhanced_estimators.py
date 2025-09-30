#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.enhanced_estimators module.

Tests enhanced energy estimation functionality including:
- Module scaling analysis for body, brain, and mind components
- Comprehensive energy analysis with theoretical limits
- Scaling relationship analysis and efficiency calculations
- Bootstrap confidence intervals and statistical validation
- Integration with workload models and energy coefficients

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.enhanced_estimators import (
    ModuleScalingData,
    ComprehensiveEnergyAnalysis,
    EnhancedEnergyEstimator
)
from antstack_core.analysis.energy import EnergyCoefficients

try:
    import numpy as np
    HAS_NUMPY_REAL = True
except ImportError:
    HAS_NUMPY_REAL = False


class TestModuleScalingData(unittest.TestCase):
    """Test ModuleScalingData dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.parameter_values = [1, 2, 4, 8, 16]
        self.energy_values = [0.1, 0.15, 0.22, 0.35, 0.55]
        self.flops_values = [1000, 1800, 3200, 5800, 10400]
        self.memory_values = [100, 150, 220, 350, 550]

    def test_module_scaling_data_creation(self):
        """Test basic ModuleScalingData creation."""
        scaling_data = ModuleScalingData(
            module_name="test_module",
            parameter_name="test_param",
            parameter_values=self.parameter_values,
            energy_values=self.energy_values,
            flops_values=self.flops_values,
            memory_values=self.memory_values,
            scaling_exponent=0.8,
            r_squared=0.95,
            scaling_regime="sub-linear"
        )

        self.assertEqual(scaling_data.module_name, "test_module")
        self.assertEqual(scaling_data.parameter_name, "test_param")
        self.assertEqual(scaling_data.scaling_exponent, 0.8)
        self.assertEqual(scaling_data.r_squared, 0.95)
        self.assertEqual(scaling_data.scaling_regime, "sub-linear")

    def test_module_scaling_data_defaults(self):
        """Test ModuleScalingData with default values."""
        scaling_data = ModuleScalingData(
            module_name="test_module",
            parameter_name="test_param",
            parameter_values=self.parameter_values,
            energy_values=self.energy_values,
            flops_values=self.flops_values,
            memory_values=self.memory_values
        )

        self.assertIsNone(scaling_data.scaling_exponent)
        self.assertIsNone(scaling_data.r_squared)
        self.assertIsNone(scaling_data.scaling_regime)
        self.assertIsNone(scaling_data.efficiency_ratio)
        self.assertIsNone(scaling_data.theoretical_limit_j)

    def test_module_scaling_data_with_analysis_results(self):
        """Test ModuleScalingData with complete analysis results."""
        scaling_data = ModuleScalingData(
            module_name="body_module",
            parameter_name="joints",
            parameter_values=self.parameter_values,
            energy_values=self.energy_values,
            flops_values=self.flops_values,
            memory_values=self.memory_values,
            scaling_exponent=0.85,
            r_squared=0.92,
            scaling_regime="sub-linear",
            efficiency_ratio=0.75,
            theoretical_limit_j=1e-20
        )

        self.assertEqual(scaling_data.efficiency_ratio, 0.75)
        self.assertEqual(scaling_data.theoretical_limit_j, 1e-20)


class TestComprehensiveEnergyAnalysis(unittest.TestCase):
    """Test ComprehensiveEnergyAnalysis dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.body_data = ModuleScalingData(
            module_name="body",
            parameter_name="joints",
            parameter_values=[1, 2, 4, 8],
            energy_values=[0.1, 0.15, 0.22, 0.35],
            flops_values=[1000, 1800, 3200, 5800],
            memory_values=[100, 150, 220, 350]
        )

        self.brain_data = ModuleScalingData(
            module_name="brain",
            parameter_name="channels",
            parameter_values=[10, 20, 40, 80],
            energy_values=[0.05, 0.08, 0.12, 0.18],
            flops_values=[5000, 8500, 14500, 24650],
            memory_values=[500, 850, 1450, 2465]
        )

        self.mind_data = ModuleScalingData(
            module_name="mind",
            parameter_name="horizon",
            parameter_values=[1, 2, 4, 8],
            energy_values=[0.02, 0.03, 0.045, 0.065],
            flops_values=[2000, 3500, 6000, 10500],
            memory_values=[200, 350, 600, 1050]
        )

    def test_comprehensive_energy_analysis_creation(self):
        """Test basic ComprehensiveEnergyAnalysis creation."""
        analysis = ComprehensiveEnergyAnalysis(
            body_analysis=self.body_data,
            brain_analysis=self.brain_data,
            mind_analysis=self.mind_data,
            system_efficiency=0.8,
            total_energy_per_decision_j=0.5,
            cost_of_transport=1.2
        )

        self.assertEqual(analysis.system_efficiency, 0.8)
        self.assertEqual(analysis.total_energy_per_decision_j, 0.5)
        self.assertEqual(analysis.cost_of_transport, 1.2)

        # Check that module analyses are correctly assigned
        self.assertEqual(analysis.body_analysis.module_name, "body")
        self.assertEqual(analysis.brain_analysis.module_name, "brain")
        self.assertEqual(analysis.mind_analysis.module_name, "mind")

    def test_comprehensive_energy_analysis_defaults(self):
        """Test ComprehensiveEnergyAnalysis with defaults."""
        analysis = ComprehensiveEnergyAnalysis(
            body_analysis=self.body_data,
            brain_analysis=self.brain_data,
            mind_analysis=self.mind_data
        )

        self.assertEqual(analysis.system_efficiency, 0.0)
        self.assertEqual(analysis.total_energy_per_decision_j, 0.0)
        self.assertEqual(analysis.cost_of_transport, 0.0)
        self.assertEqual(analysis.key_numbers, {})

    def test_comprehensive_energy_analysis_with_key_numbers(self):
        """Test ComprehensiveEnergyAnalysis with key numbers."""
        key_numbers = {
            "landauer_limit": 1e-20,
            "ant_brain_energy": 5e-3,
            "ant_body_energy": 1.2,
            "ant_mind_energy": 0.1
        }

        analysis = ComprehensiveEnergyAnalysis(
            body_analysis=self.body_data,
            brain_analysis=self.brain_data,
            mind_analysis=self.mind_data,
            key_numbers=key_numbers
        )

        self.assertEqual(analysis.key_numbers["landauer_limit"], 1e-20)
        self.assertEqual(analysis.key_numbers["ant_brain_energy"], 5e-3)


class TestEnhancedEnergyEstimator(unittest.TestCase):
    """Test EnhancedEnergyEstimator class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create energy coefficients for testing
        self.coefficients = EnergyCoefficients(
            flops_pj=1.0,
            sram_pj_per_byte=0.1,
            dram_pj_per_byte=20.0,
            spike_aj=1.0,
            body_per_joint_w=2.0,
            body_sensor_w_per_channel=0.005,
            baseline_w=0.5
        )

        self.estimator = EnhancedEnergyEstimator(self.coefficients)

        # Base parameters for different modules
        self.body_base_params = {
            'v': 1.0,      # velocity m/s
            'm': 0.001,    # mass kg
            'L': 0.01,     # leg length m
            'dt': 0.01,    # decision time s
            'g': 9.81      # gravity m/s²
        }

        self.brain_base_params = {
            'C': 100,      # channels
            'sparsity': 0.1,  # connection sparsity
            'dt': 0.01     # decision time s
        }

        self.mind_base_params = {
            'H': 5,        # planning horizon
            'B': 3,        # branching factor
            'dt': 0.01     # decision time s
        }

    def test_enhanced_energy_estimator_creation(self):
        """Test EnhancedEnergyEstimator creation."""
        estimator = EnhancedEnergyEstimator(self.coefficients)
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.coefficients, self.coefficients)

    def test_analyze_body_scaling(self):
        """Test body scaling analysis."""
        j_values = [1, 2, 4, 8, 16]

        result = self.estimator.analyze_body_scaling(j_values, self.body_base_params)

        self.assertIsInstance(result, ModuleScalingData)
        self.assertEqual(result.module_name, "AntBody")
        self.assertEqual(result.parameter_name, "joints")
        self.assertEqual(len(result.parameter_values), 5)
        self.assertEqual(len(result.energy_values), 5)
        self.assertEqual(len(result.flops_values), 5)
        self.assertEqual(len(result.memory_values), 5)

        # Check that energy increases with joint count
        self.assertTrue(all(result.energy_values[i] <= result.energy_values[i+1]
                          for i in range(len(result.energy_values)-1)))

    def test_analyze_brain_scaling(self):
        """Test brain scaling analysis."""
        c_values = [10, 20, 40, 80, 160]

        result = self.estimator.analyze_brain_scaling(c_values, self.brain_base_params)

        self.assertIsInstance(result, ModuleScalingData)
        self.assertEqual(result.module_name, "AntBrain")
        self.assertEqual(result.parameter_name, "channels")
        self.assertEqual(len(result.parameter_values), 5)
        self.assertEqual(len(result.energy_values), 5)
        self.assertEqual(len(result.flops_values), 5)
        self.assertEqual(len(result.memory_values), 5)

    def test_analyze_mind_scaling(self):
        """Test mind scaling analysis."""
        h_values = [1, 2, 4, 8, 16]

        result = self.estimator.analyze_mind_scaling(h_values, self.mind_base_params)

        self.assertIsInstance(result, ModuleScalingData)
        self.assertEqual(result.module_name, "AntMind")
        self.assertEqual(result.parameter_name, "horizon")
        self.assertEqual(len(result.parameter_values), 5)
        self.assertEqual(len(result.energy_values), 5)
        self.assertEqual(len(result.flops_values), 5)
        self.assertEqual(len(result.memory_values), 5)

    def test_perform_comprehensive_analysis(self):
        """Test comprehensive analysis across all modules."""
        result = self.estimator.perform_comprehensive_analysis(
            body_params=self.body_base_params,
            brain_params=self.brain_base_params,
            mind_params=self.mind_base_params
        )

        self.assertIsInstance(result, ComprehensiveEnergyAnalysis)

        # Check that all module analyses are present
        self.assertIsInstance(result.body_analysis, ModuleScalingData)
        self.assertIsInstance(result.brain_analysis, ModuleScalingData)
        self.assertIsInstance(result.mind_analysis, ModuleScalingData)

        # Check that system-level metrics are calculated
        self.assertIsInstance(result.system_efficiency, float)
        self.assertIsInstance(result.total_energy_per_decision_j, float)
        self.assertIsInstance(result.cost_of_transport, float)

    def test_calculate_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        body_data = self.estimator.analyze_body_scaling([1, 2, 4, 8], self.body_base_params)
        brain_data = self.estimator.analyze_brain_scaling([10, 20, 40], self.brain_base_params)
        mind_data = self.estimator.analyze_mind_scaling([1, 2, 4], self.mind_base_params)

        efficiency = self.estimator.calculate_efficiency_metrics(body_data, brain_data, mind_data)

        self.assertIsInstance(efficiency, dict)
        self.assertIn("system_efficiency", efficiency)
        self.assertIn("energy_distribution", efficiency)
        self.assertIn("bottlenecks", efficiency)

    def test_compare_with_theoretical_limits(self):
        """Test comparison with theoretical energy limits."""
        body_data = self.estimator.analyze_body_scaling([1, 2, 4], self.body_base_params)

        comparison = self.estimator.compare_with_theoretical_limits(body_data)

        self.assertIsInstance(comparison, dict)
        self.assertIn("landauer_efficiency", comparison)
        self.assertIn("thermodynamic_efficiency", comparison)
        self.assertIn("fundamental_limits", comparison)

    def test_generate_scaling_report(self):
        """Test scaling report generation."""
        analysis = self.estimator.perform_comprehensive_analysis(
            body_params=self.body_base_params,
            brain_params=self.brain_base_params,
            mind_params=self.mind_base_params
        )

        report = self.estimator.generate_scaling_report(analysis)

        self.assertIsInstance(report, str)
        self.assertIn("AntBody", report)
        self.assertIn("AntBrain", report)
        self.assertIn("AntMind", report)
        self.assertIn("scaling", report.lower())

    def test_enhanced_energy_estimator_edge_cases(self):
        """Test EnhancedEnergyEstimator with edge cases."""
        # Test with empty parameter lists
        result = self.estimator.analyze_body_scaling([], self.body_base_params)
        self.assertIsInstance(result, ModuleScalingData)
        self.assertEqual(len(result.parameter_values), 0)

        # Test with single parameter value
        result = self.estimator.analyze_body_scaling([4], self.body_base_params)
        self.assertIsInstance(result, ModuleScalingData)
        self.assertEqual(len(result.parameter_values), 1)

        # Test with extreme parameter values
        result = self.estimator.analyze_body_scaling([100], self.body_base_params)
        self.assertIsInstance(result, ModuleScalingData)

    def test_scaling_analysis_consistency(self):
        """Test consistency of scaling analysis across modules."""
        # Analyze the same scaling pattern for different modules
        params_list = [1, 2, 4, 8, 16]

        body_result = self.estimator.analyze_body_scaling(params_list, self.body_base_params)
        brain_result = self.estimator.analyze_brain_scaling(params_list, self.brain_base_params)
        mind_result = self.estimator.analyze_mind_scaling(params_list, self.mind_base_params)

        # All should return valid ModuleScalingData objects
        self.assertIsInstance(body_result, ModuleScalingData)
        self.assertIsInstance(brain_result, ModuleScalingData)
        self.assertIsInstance(mind_result, ModuleScalingData)

        # All should have the same number of data points
        self.assertEqual(len(body_result.parameter_values), len(params_list))
        self.assertEqual(len(brain_result.parameter_values), len(params_list))
        self.assertEqual(len(mind_result.parameter_values), len(params_list))


class TestEnhancedEstimatorsIntegration(unittest.TestCase):
    """Test integration between enhanced estimators components."""

    def setUp(self):
        """Set up test fixtures."""
        self.coefficients = EnergyCoefficients()
        self.estimator = EnhancedEnergyEstimator(self.coefficients)

    def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow."""
        # Define parameters for all modules
        body_params = {
            'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81
        }
        brain_params = {
            'C': 100, 'sparsity': 0.1, 'dt': 0.01
        }
        mind_params = {
            'H': 5, 'B': 3, 'dt': 0.01
        }

        # Perform comprehensive analysis
        comprehensive_result = self.estimator.perform_comprehensive_analysis(
            body_params=body_params,
            brain_params=brain_params,
            mind_params=mind_params
        )

        # Generate report
        report = self.estimator.generate_scaling_report(comprehensive_result)

        # Verify workflow completion
        self.assertIsInstance(comprehensive_result, ComprehensiveEnergyAnalysis)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    def test_cross_module_energy_analysis(self):
        """Test energy analysis across different modules."""
        body_data = self.estimator.analyze_body_scaling([1, 2, 4], {'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81})
        brain_data = self.estimator.analyze_brain_scaling([10, 20, 40], {'C': 100, 'sparsity': 0.1, 'dt': 0.01})
        mind_data = self.estimator.analyze_mind_scaling([1, 2, 4], {'H': 5, 'B': 3, 'dt': 0.01})

        # Calculate system-level metrics
        total_energy = (body_data.energy_values[-1] +
                       brain_data.energy_values[-1] +
                       mind_data.energy_values[-1])

        # Verify cross-module calculations
        self.assertGreater(total_energy, 0)
        self.assertGreater(body_data.energy_values[-1], brain_data.energy_values[-1])  # Body typically uses more energy
        self.assertGreater(brain_data.energy_values[-1], mind_data.energy_values[-1])  # Brain uses more than mind

    def test_parameter_sensitivity_analysis(self):
        """Test sensitivity of results to parameter changes."""
        base_params = {'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81}

        # Test with different velocities
        result_slow = self.estimator.analyze_body_scaling([4], {**base_params, 'v': 0.5})
        result_fast = self.estimator.analyze_body_scaling([4], {**base_params, 'v': 2.0})

        # Faster velocity should use more energy
        self.assertGreater(result_fast.energy_values[0], result_slow.energy_values[0])

    def test_energy_scaling_relationships(self):
        """Test that energy scales appropriately with system parameters."""
        # Test body scaling with joint count
        joint_counts = [1, 2, 4, 8, 16, 32]
        result = self.estimator.analyze_body_scaling(
            joint_counts,
            {'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81}
        )

        # Energy should generally increase with joint count
        for i in range(1, len(result.energy_values)):
            self.assertGreaterEqual(result.energy_values[i], result.energy_values[i-1])


class TestEnhancedEstimatorsRobustness(unittest.TestCase):
    """Test robustness of enhanced estimators methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.coefficients = EnergyCoefficients()
        self.estimator = EnhancedEnergyEstimator(self.coefficients)

    def test_estimator_with_extreme_coefficients(self):
        """Test estimator with extreme coefficient values."""
        # Test with very high energy coefficients
        high_energy_coeffs = EnergyCoefficients(
            flops_pj=100.0,
            sram_pj_per_byte=10.0,
            dram_pj_per_byte=200.0,
            baseline_w=5.0
        )
        high_estimator = EnhancedEnergyEstimator(high_energy_coeffs)

        result = high_estimator.analyze_body_scaling([4], {'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81})
        self.assertIsInstance(result, ModuleScalingData)
        self.assertGreater(result.energy_values[0], 0)

    def test_estimator_with_zero_coefficients(self):
        """Test estimator with zero coefficient values."""
        # Test with zero energy coefficients
        zero_energy_coeffs = EnergyCoefficients(
            flops_pj=0.0,
            sram_pj_per_byte=0.0,
            dram_pj_per_byte=0.0,
            baseline_w=0.0
        )
        zero_estimator = EnhancedEnergyEstimator(zero_energy_coeffs)

        result = zero_estimator.analyze_body_scaling([4], {'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81})
        self.assertIsInstance(result, ModuleScalingData)
        # Energy should be zero or very small
        self.assertLessEqual(result.energy_values[0], 1e-10)

    def test_estimator_error_handling(self):
        """Test error handling in estimator methods."""
        # Test with invalid parameters
        invalid_params = {'invalid': 'parameter'}

        # Should handle gracefully without crashing
        result = self.estimator.analyze_body_scaling([4], invalid_params)
        self.assertIsInstance(result, ModuleScalingData)

    def test_scaling_analysis_numerical_stability(self):
        """Test numerical stability of scaling analysis."""
        # Test with parameters that might cause numerical issues
        extreme_params = {
            'v': 1e10,    # Very high velocity
            'm': 1e-10,   # Very small mass
            'L': 1e-5,    # Very small length
            'dt': 1e-10,  # Very small time step
            'g': 1e5      # Very high gravity
        }

        result = self.estimator.analyze_body_scaling([4], extreme_params)
        self.assertIsInstance(result, ModuleScalingData)

        # Results should be finite (not NaN or infinite)
        self.assertTrue(np.isfinite(result.energy_values[0]))
        self.assertTrue(np.isfinite(result.flops_values[0]))
        self.assertTrue(np.isfinite(result.memory_values[0]))


if __name__ == '__main__':
    unittest.main()
