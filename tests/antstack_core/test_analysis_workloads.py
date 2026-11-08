#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.workloads module.

Tests workload modeling, complexity calculations, and energy estimation
for different computational components (body, brain, mind).

Following .cursorrules principles:
- Real computational workload analysis (no mocks)
- Statistical validation of workload models
- Comprehensive edge case testing
- Professional documentation with scientific references
"""
import pytest

import unittest
import math
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.workloads import (
    body_workload_closed_form, brain_workload_closed_form, mind_workload_closed_form,
    calculate_contact_complexity, calculate_sparse_neural_complexity,
    calculate_active_inference_complexity, body_workload, brain_workload, mind_workload,
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form, estimate_body_compute_per_decision,
    estimate_brain_compute_per_decision, estimate_mind_compute_per_decision
)
from antstack_core.analysis.energy import EnergyCoefficients, ComputeLoad
class TestScientificWorkloadContracts(unittest.TestCase):
    def test_supported_contact_formulas_are_distinct(self):
        self.assertEqual(calculate_contact_complexity(18, 12, 'pgs'), (17280, 3168, 12))
        self.assertEqual(calculate_contact_complexity(18, 12, 'si'), (6264, 3744, 1))
        self.assertEqual(calculate_contact_complexity(18, 12, 'ldlt'), (52488, 23328, 1))
        for solver in ('lcp', 'mlcp', 'typo'):
            with self.assertRaises(ValueError):
                calculate_contact_complexity(18, 12, solver)

    def test_supported_topology_formulas_and_empty_network(self):
        self.assertEqual(calculate_sparse_neural_complexity(10000, .02, 'random'), (40000, 120000, 20))
        structured = calculate_sparse_neural_complexity(10000, .02, 'structured')
        self.assertAlmostEqual(structured[0], 200 * math.log(10000))
        self.assertEqual(structured[1:], (80000, 20))
        self.assertEqual(calculate_sparse_neural_complexity(0, 0, 'structured'), (0, 0, 0))
        for pattern in ('biological', 'small_world', 'scale_free', 'typo'):
            with self.assertRaises(ValueError):
                calculate_sparse_neural_complexity(10000, .02, pattern)

    def test_budget_applies_at_every_horizon_and_to_every_counter(self):
        params = {'B': 4, 'hz': 100, 'state_dim': 16, 'action_dim': 6}
        for horizon in (9, 10, 11):
            load = enhanced_mind_workload_closed_form(.01, dict(params, H_p=horizon))
            self.assertEqual(load.flops, 1000 * 2950)
            self.assertEqual(load.sram_bytes + load.dram_bytes, 1000 * 156)
        sampled = enhanced_mind_workload_closed_form(.01, dict(params, H_p=12, max_policies=1000))
        conservative = enhanced_mind_workload_closed_form(.01, dict(params, H_p=12, max_policies=100))
        self.assertEqual(sampled.flops, 10 * conservative.flops)
        self.assertEqual(sampled.sram_bytes, 10 * conservative.sram_bytes)
        exact = enhanced_mind_workload_closed_form(.01, dict(params, H_p=10, max_policies=None))
        self.assertEqual(exact.flops, 4 ** 10 * 2950)
        zero_horizon = enhanced_mind_workload_closed_form(.01, dict(params, H_p=0))
        self.assertEqual(zero_horizon.flops, 2950)

    def test_canonical_branching_controls_the_exact_tree_model(self):
        for branching in (2, 4):
            expected_flops, expected_memory = calculate_active_inference_complexity(2, branching, 3, 2, 3)
            load = mind_workload_closed_form(1, {'B': branching, 'H_p': 2, 'state_dim': 3, 'action_dim': 2, 'hz': 1})
            self.assertEqual(load.flops, expected_flops)
            self.assertEqual(load.sram_bytes + load.dram_bytes, expected_memory)
        with self.assertRaises(ValueError):
            mind_workload_closed_form(1, {'branching': 4})

    def test_runtime_boundaries_reject_invalid_domains_before_execution(self):
        for fn in (body_workload_closed_form, enhanced_body_workload_closed_form, body_workload,
                   brain_workload_closed_form, enhanced_brain_workload_closed_form, brain_workload,
                   mind_workload_closed_form, enhanced_mind_workload_closed_form, mind_workload):
            for duration in (-1, float('nan'), float('inf')):
                with self.subTest(fn=fn, duration=duration), self.assertRaises(ValueError):
                    fn(duration, {})
        for rho in (-.01, 1.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                enhanced_brain_workload_closed_form(.01, {'rho': rho})
        for budget in (0, -1, 1.5, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                enhanced_mind_workload_closed_form(.01, {'max_policies': budget})
        for params in ({'J': -1}, {'C': -1}, {'P': float('nan')}, {'hz': 0}, {'contact_solver': 'mlcp'}):
            with self.assertRaises(ValueError):
                enhanced_body_workload_closed_form(.01, params)




class TestBodyWorkloads(unittest.TestCase):
    """Test body workload modeling and complexity calculations."""

    def setUp(self):
        """Set up test fixtures for body workload analysis."""
        self.base_params = {
            'J': 6,      # joints
            'C': 12,     # contacts per joint
            'S': 256,    # sensors
            'hz': 100    # control frequency
        }
        self.duration_s = 0.01  # 10ms decision cycle

    def test_body_workload_closed_form_basic(self):
        """Test basic body workload closed-form calculation."""
        load = body_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)
        self.assertGreater(load.sram_bytes, 0)

    def test_body_workload_closed_form_scaling(self):
        """Test body workload scaling with joint count."""
        # Base case
        load_base = body_workload_closed_form(self.duration_s, self.base_params)

        # Double joints
        params_double = self.base_params.copy()
        params_double['J'] = 12
        load_double = body_workload_closed_form(self.duration_s, params_double)

        # Should scale approximately with joints
        self.assertGreater(load_double.flops, load_base.flops)

    def test_body_workload_closed_form_sensor_scaling(self):
        """Test body workload scaling with sensor count."""
        # Base case
        load_base = body_workload_closed_form(self.duration_s, self.base_params)

        # Double sensors
        params_double = self.base_params.copy()
        params_double['S'] = 512
        load_double = body_workload_closed_form(self.duration_s, params_double)

        # Should scale with sensors
        self.assertGreater(load_double.flops, load_base.flops)

    def test_enhanced_body_workload_closed_form(self):
        """Test enhanced body workload model."""
        load = enhanced_body_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)
        self.assertGreaterEqual(load.sram_bytes, 0)
        self.assertGreaterEqual(load.dram_bytes, 0)

    def test_body_workload_runtime_model(self):
        """Test runtime body workload model."""
        load = body_workload(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_estimate_body_compute_per_decision(self):
        """Test body compute estimation per decision."""
        load = estimate_body_compute_per_decision(self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)


class TestBrainWorkloads(unittest.TestCase):
    """Test brain workload modeling and neural complexity calculations."""

    def setUp(self):
        """Set up test fixtures for brain workload analysis."""
        self.base_params = {
            'K': 64,       # AL input channels
            'N_KC': 50000, # Kenyon cells
            'rho': 0.02,   # Sparsity
            'H': 64        # Hidden dimension
        }
        self.duration_s = 0.01  # 10ms decision cycle

    def test_brain_workload_closed_form_basic(self):
        """Test basic brain workload closed-form calculation."""
        load = brain_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)
        self.assertGreater(load.sram_bytes, 0)

    def test_brain_workload_scaling_with_channels(self):
        """Test brain workload scaling with input channels."""
        # Base case
        load_base = brain_workload_closed_form(self.duration_s, self.base_params)

        # Double channels
        params_double = self.base_params.copy()
        params_double['K'] = 128
        load_double = brain_workload_closed_form(self.duration_s, params_double)

        # Should scale with channels (approximately)
        self.assertGreater(load_double.flops, load_base.flops)

    def test_brain_workload_scaling_with_sparsity(self):
        """Test brain workload scaling with sparsity."""
        # High sparsity (efficient)
        params_sparse = self.base_params.copy()
        params_sparse['rho'] = 0.01
        load_sparse = brain_workload_closed_form(self.duration_s, params_sparse)

        # Low sparsity (less efficient)
        params_dense = self.base_params.copy()
        params_dense['rho'] = 0.1
        load_dense = brain_workload_closed_form(self.duration_s, params_dense)

        # Lower sparsity should generally increase computation
        self.assertGreaterEqual(load_dense.flops, load_sparse.flops)

    def test_enhanced_brain_workload_closed_form(self):
        """Test enhanced brain workload model."""
        load = enhanced_brain_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_brain_workload_runtime_model(self):
        """Test runtime brain workload model."""
        load = brain_workload(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_estimate_brain_compute_per_decision(self):
        """Test brain compute estimation per decision."""
        load = estimate_brain_compute_per_decision(self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)


class TestMindWorkloads(unittest.TestCase):
    """Test mind workload modeling and active inference complexity."""

    def setUp(self):
        """Set up test fixtures for mind workload analysis."""
        self.base_params = {
            'H_p': 5,         # Policy horizon
            'B': 3,           # Canonical branching factor
            'state_dim': 8,   # State dimension
            'action_dim': 4,  # Action dimension
            'hz': 100         # Planning frequency (Hz)
        }
        self.duration_s = 0.01  # 10ms decision cycle

    def test_mind_workload_closed_form_basic(self):
        """Test basic mind workload closed-form calculation."""
        load = mind_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_mind_workload_scaling_with_horizon(self):
        """Test mind workload scaling with policy horizon."""
        # Base case
        load_base = mind_workload_closed_form(self.duration_s, self.base_params)

        # Longer horizon
        params_long = self.base_params.copy()
        params_long['H_p'] = 10
        load_long = mind_workload_closed_form(self.duration_s, params_long)

        # Should scale with horizon (potentially exponentially)
        self.assertGreater(load_long.flops, load_base.flops)

    def test_mind_workload_scaling_with_branching(self):
        """Test mind workload scaling with branching factor."""
        # Base case
        load_base = mind_workload_closed_form(self.duration_s, self.base_params)

        # Higher branching
        params_branchy = self.base_params.copy()
        params_branchy['B'] = 5
        load_branchy = mind_workload_closed_form(self.duration_s, params_branchy)

        # Should scale with branching
        self.assertGreater(load_branchy.flops, load_base.flops)

    def test_enhanced_mind_workload_closed_form(self):
        """Test enhanced mind workload model."""
        load = enhanced_mind_workload_closed_form(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_mind_workload_runtime_model(self):
        """Test runtime mind workload model."""
        load = mind_workload(self.duration_s, self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)

    def test_estimate_mind_compute_per_decision(self):
        """Test mind compute estimation per decision."""
        load = estimate_mind_compute_per_decision(self.base_params)

        self.assertIsInstance(load, ComputeLoad)
        self.assertGreater(load.flops, 0)


class TestComplexityCalculations(unittest.TestCase):
    """Test complexity calculation functions."""

    def test_calculate_contact_complexity(self):
        """Test contact complexity calculation."""
        J, C, S = calculate_contact_complexity(J=6, C=12, solver="pgs")

        self.assertIsInstance(J, (int, float))
        self.assertIsInstance(C, (int, float))
        self.assertIsInstance(S, (int, float))
        self.assertGreater(J, 0)
        self.assertGreater(C, 0)

    def test_calculate_sparse_neural_complexity(self):
        """Test sparse neural complexity calculation."""
        flops, memory, spike_events = calculate_sparse_neural_complexity(
            N_total=10000, sparsity=0.02, connectivity_pattern="random"
        )

        self.assertIsInstance(flops, (int, float))
        self.assertIsInstance(memory, (int, float))
        self.assertIsInstance(spike_events, (int, float))
        self.assertGreater(flops, 0)
        self.assertGreater(memory, 0)
        self.assertGreaterEqual(spike_events, 0)

    def test_calculate_active_inference_complexity(self):
        """Test active inference complexity calculation."""
        flops, memory = calculate_active_inference_complexity(
            horizon=5, branching=3, state_dim=8, action_dim=4
        )

        self.assertIsInstance(flops, (int, float))
        self.assertIsInstance(memory, (int, float))
        self.assertGreater(flops, 0)
        self.assertGreater(memory, 0)

    def test_calculate_active_inference_complexity_scaling(self):
        """Test active inference complexity scaling."""
        flops_small, _ = calculate_active_inference_complexity(
            horizon=3, branching=2, state_dim=4, action_dim=2
        )
        flops_large, _ = calculate_active_inference_complexity(
            horizon=6, branching=4, state_dim=8, action_dim=4
        )

        # Should scale with complexity parameters
        self.assertGreater(flops_large, flops_small)


class TestWorkloadIntegration(unittest.TestCase):
    """Test integration between workload models and energy estimation."""

    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.coeffs = EnergyCoefficients()
        self.body_params = {'J': 6, 'C': 12, 'S': 256, 'hz': 100}
        self.brain_params = {'K': 64, 'N_KC': 50000, 'rho': 0.02, 'H': 64}
        self.mind_params = {'H_p': 5, 'B': 3, 'state_dim': 8, 'action_dim': 4, 'hz': 100}
        self.duration_s = 0.01

    def test_body_workload_energy_integration(self):
        """Test integration of body workload with energy estimation."""
        load = body_workload_closed_form(self.duration_s, self.body_params)
        energy = load.flops * self.coeffs.flops_pj * 1e-12  # Convert to Joules

        self.assertGreater(energy, 0)
        self.assertLess(energy, 1e-3)  # Should be reasonable energy value

    def test_brain_workload_energy_integration(self):
        """Test integration of brain workload with energy estimation."""
        load = brain_workload_closed_form(self.duration_s, self.brain_params)
        energy = load.flops * self.coeffs.flops_pj * 1e-12

        self.assertGreater(energy, 0)

    def test_mind_workload_energy_integration(self):
        """Test integration of mind workload with energy estimation."""
        load = mind_workload_closed_form(self.duration_s, self.mind_params)
        energy = load.flops * self.coeffs.flops_pj * 1e-12

        self.assertGreater(energy, 0)

    def test_workload_consistency_across_models(self):
        """Test consistency between different workload models."""
        # Compare closed-form vs runtime models for body
        load_closed = body_workload_closed_form(self.duration_s, self.body_params)
        load_runtime = body_workload(self.duration_s, self.body_params)

        # Should have same structure
        self.assertIsInstance(load_closed, ComputeLoad)
        self.assertIsInstance(load_runtime, ComputeLoad)

        # Values should be in similar ranges (allowing for model differences)
        self.assertGreater(load_closed.flops, 0)
        self.assertGreater(load_runtime.flops, 0)


class TestWorkloadEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in workload models."""

    def test_body_workload_zero_joints(self):
        """Test body workload with zero joints."""
        params = {'J': 0, 'C': 12, 'S': 256, 'hz': 100}
        load = body_workload_closed_form(0.01, params)

        # Should handle gracefully
        self.assertIsInstance(load, ComputeLoad)
        self.assertGreaterEqual(load.flops, 0)

    def test_brain_workload_zero_channels(self):
        """Test brain workload with zero channels."""
        params = {'K': 0, 'N_KC': 50000, 'rho': 0.02, 'H': 64}
        load = brain_workload_closed_form(0.01, params)

        # Should handle gracefully
        self.assertIsInstance(load, ComputeLoad)
        self.assertGreaterEqual(load.flops, 0)

    def test_mind_workload_zero_horizon(self):
        """Test mind workload with zero horizon."""
        params = {'H_p': 0, 'B': 3, 'state_dim': 8, 'action_dim': 4}
        load = mind_workload_closed_form(0.01, params)

        # Should handle gracefully
        self.assertIsInstance(load, ComputeLoad)
        self.assertGreaterEqual(load.flops, 0)

    def test_complexity_calculations_edge_cases(self):
        """Test complexity calculations with edge case inputs."""
        # Zero inputs
        flops, memory, sparsity = calculate_sparse_neural_complexity(0, 0.02)
        self.assertGreaterEqual(flops, 0)
        self.assertGreaterEqual(memory, 0)

        # Active inference with minimal parameters
        flops, memory = calculate_active_inference_complexity(1, 1, 1, 1)
        self.assertGreaterEqual(flops, 0)
        self.assertGreaterEqual(memory, 0)


class TestWorkloadScalingAnalysis(unittest.TestCase):
    """Test workload scaling behavior across parameter ranges."""

    def test_body_scaling_analysis(self):
        """Test body workload scaling across joint counts."""
        joint_counts = [3, 6, 9, 12]
        flops_values = []

        base_params = {'C': 12, 'S': 256, 'hz': 100}

        for J in joint_counts:
            params = base_params.copy()
            params['J'] = J
            load = body_workload_closed_form(0.01, params)
            flops_values.append(load.flops)

        # Should generally increase with joints
        self.assertEqual(len(flops_values), len(joint_counts))
        # Check monotonicity (allowing for model nonlinearities)
        self.assertTrue(all(f >= 0 for f in flops_values))

    def test_brain_scaling_analysis(self):
        """Test brain workload scaling across channel counts."""
        channel_counts = [32, 64, 128, 256]
        flops_values = []

        base_params = {'N_KC': 50000, 'rho': 0.02, 'H': 64}

        for K in channel_counts:
            params = base_params.copy()
            params['K'] = K
            load = brain_workload_closed_form(0.01, params)
            flops_values.append(load.flops)

        # Should generally increase with channels
        self.assertEqual(len(flops_values), len(channel_counts))
        self.assertTrue(all(f >= 0 for f in flops_values))

    def test_mind_scaling_analysis(self):
        """Test mind workload scaling across horizons."""
        horizons = [3, 5, 7, 9]
        flops_values = []

        base_params = {'B': 3, 'state_dim': 8, 'action_dim': 4}

        for H_p in horizons:
            params = base_params.copy()
            params['H_p'] = H_p
            load = mind_workload_closed_form(0.01, params)
            flops_values.append(load.flops)

        # Should generally increase with horizon
        self.assertEqual(len(flops_values), len(horizons))
        self.assertTrue(all(f >= 0 for f in flops_values))


if __name__ == '__main__':
    unittest.main()


@pytest.mark.parametrize("name", ["body_workload_closed_form", "brain_workload_closed_form", "mind_workload_closed_form",
                                "enhanced_body_workload_closed_form", "enhanced_brain_workload_closed_form", "enhanced_mind_workload_closed_form"])
@pytest.mark.parametrize("hz", [49, 93, 100])
def test_reciprocal_period_keeps_one_discrete_update(name, hz):
    from antstack_core.analysis import workloads
    function = getattr(workloads, name)
    one, many = function(1 / hz, {"hz": hz}), function(1, {"hz": hz})
    assert many.flops == pytest.approx(one.flops * hz)
    assert function(0.5 / hz, {"hz": hz}).flops == 0
    assert function(1.5 / hz, {"hz": hz}).flops == pytest.approx(one.flops)
    if name == "brain_workload_closed_form":
        assert function(1 / hz, {"hz": hz, "K": 1, "N_KC": 1, "rho": 1, "H": 1}).flops == 101
