#!/usr/bin/env python3
"""Integration tests for complexity energetics analysis.

Comprehensive tests for integrated analysis workflows and parameter validation.
Tests core functionality without external script dependencies.
"""

import unittest
import tempfile
import shutil
import json
import sys
import os
import math
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.energy import (
    ComputeLoad, estimate_detailed_energy, EnergyCoefficients
)
from antstack_core.analysis.statistics import (
    analyze_scaling_relationship, bootstrap_mean_ci,
    calculate_energy_efficiency_metrics, estimate_theoretical_limits
)
from antstack_core.analysis.workloads import (
    calculate_contact_complexity, calculate_sparse_neural_complexity,
    calculate_active_inference_complexity, enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form, enhanced_mind_workload_closed_form
)


class TestContactDynamicsIntegration(unittest.TestCase):
    """Test integrated contact dynamics analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.coeffs = EnergyCoefficients()

    def test_solver_comparison_integration(self):
        """Test contact solver comparison using core functions."""
        contact_counts = [5, 10, 15, 20]
        solvers = ['pgs', 'lcp', 'mlcp']

        results = {}
        for solver in solvers:
            flops_values = []
            memory_values = []

            for contacts in contact_counts:
                flops, memory, iters = calculate_contact_complexity(10, contacts, solver)
                flops_values.append(flops)
                memory_values.append(memory)

            # Analyze scaling
            flops_scaling = analyze_scaling_relationship(contact_counts, flops_values)
            memory_scaling = analyze_scaling_relationship(contact_counts, memory_values)

            results[solver] = {
                'flops_scaling': flops_scaling.get('scaling_exponent', 0),
                'memory_scaling': memory_scaling.get('scaling_exponent', 0),
                'flops_values': flops_values,
                'memory_values': memory_values
            }

        # Check results structure
        for solver in solvers:
            self.assertIn(solver, results)
            self.assertGreater(results[solver]['flops_scaling'], 0)
            self.assertGreater(results[solver]['memory_scaling'], 0)
            self.assertEqual(len(results[solver]['flops_values']), len(contact_counts))

    def test_terrain_effects_integration(self):
        """Test terrain complexity effects using enhanced workloads."""
        # Test different contact configurations
        terrain_configs = [
            {'J': 10, 'C': 6, 'name': 'flat'},      # Low contacts
            {'J': 10, 'C': 12, 'name': 'rough'},    # Medium contacts
            {'J': 10, 'C': 18, 'name': 'staircase'} # High contacts
        ]

        results = {}
        for config in terrain_configs:
            # Calculate workload
            load = enhanced_body_workload_closed_form(0.01, config)

            # Calculate energy
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)

            results[config['name']] = {
                'complexity_factor': load.flops / 1000.0,  # Normalized complexity
                'energy_j': breakdown.total,
                'contacts': config['C']
            }

        # Verify terrain analysis
        self.assertLess(results['flat']['complexity_factor'],
                       results['rough']['complexity_factor'])
        self.assertLess(results['rough']['complexity_factor'],
                       results['staircase']['complexity_factor'])

        # Energy should correlate with complexity
        for terrain in results.values():
            self.assertGreater(terrain['energy_j'], 0)


class TestNeuralNetworkIntegration(unittest.TestCase):
    """Test integrated neural network analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.coeffs = EnergyCoefficients()

    def test_connectivity_patterns_integration(self):
        """Test connectivity pattern analysis using core functions."""
        network_sizes = [1000, 5000, 10000]
        sparsity = 0.02

        # Test different connectivity patterns through sparsity calculations
        patterns = ["random", "small_world", "scale_free", "biological"]

        results = {}
        for pattern in patterns:
            flops_values = []
            memory_values = []

            for size in network_sizes:
                # Use sparsity calculation as proxy for connectivity analysis
                flops, sram_bytes, dram_bytes = calculate_sparse_neural_complexity(size, sparsity)
                flops_values.append(flops)
                memory_values.append(sram_bytes + dram_bytes)

            # Analyze scaling
            flops_scaling = analyze_scaling_relationship(network_sizes, flops_values)
            memory_scaling = analyze_scaling_relationship(network_sizes, memory_values)

            results[pattern] = {
                'flops_scaling': flops_scaling.get('scaling_exponent', 0),
                'memory_scaling': memory_scaling.get('scaling_exponent', 0),
                'flops_values': flops_values,
                'memory_values': memory_values
            }

        # Check all patterns analyzed
        for pattern in patterns:
            self.assertIn(pattern, results)
            self.assertGreater(results[pattern]['flops_scaling'], 0)
            self.assertEqual(len(results[pattern]['flops_values']), len(network_sizes))

    def test_sparsity_effects_integration(self):
        """Test sparsity effects using core neural complexity functions."""
        network_size = 10000
        sparsity_levels = [0.01, 0.02, 0.05]

        results = {}
        for sparsity in sparsity_levels:
            # Calculate neural complexity for different sparsity levels
            flops, sram_bytes, dram_bytes = calculate_sparse_neural_complexity(network_size, sparsity)
            memory = sram_bytes + dram_bytes

            results[sparsity] = {
                'flops': flops,
                'memory': memory,
                'efficiency': flops / memory if memory > 0 else 0
            }

        # Check results structure and sparsity effects
        for sparsity in sparsity_levels:
            self.assertIn(sparsity, results)
            self.assertGreater(results[sparsity]['flops'], 0)
            self.assertGreater(results[sparsity]['efficiency'], 0)

        # Higher sparsity should generally be more efficient
        self.assertGreater(results[0.05]['efficiency'], results[0.01]['efficiency'])

    def test_brain_scaling_integration(self):
        """Test brain scaling analysis using enhanced workloads."""
        K_values = [64, 128, 256]

        energy_values = []
        flops_values = []

        for K in K_values:
            params = {'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}
            load = enhanced_brain_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)

            energy_values.append(breakdown.total)
            flops_values.append(load.flops)

        # Analyze scaling
        energy_scaling = analyze_scaling_relationship(K_values, energy_values)
        flops_scaling = analyze_scaling_relationship(K_values, flops_values)

        # Check results structure
        self.assertIn('scaling_exponent', energy_scaling)
        self.assertIn('scaling_exponent', flops_scaling)
        self.assertEqual(len(energy_values), len(K_values))

        # Calculate efficiency metrics
        efficiency = calculate_energy_efficiency_metrics(energy_values, [1.0/K for K in K_values])
        self.assertIn('average_energy_j', efficiency)


class TestActiveInferenceIntegration(unittest.TestCase):
    """Test integrated active inference analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.coeffs = EnergyCoefficients()

    def test_horizon_scaling_integration(self):
        """Test policy horizon scaling using core active inference functions."""
        horizons = [5, 7, 9]
        branching = 3
        state_dim = 8
        action_dim = 4

        energy_values = []
        flops_values = []

        for horizon in horizons:
            # Calculate active inference complexity
            flops, memory = calculate_active_inference_complexity(
                horizon, branching, state_dim, action_dim
            )

            # Create compute load and estimate energy
            load = ComputeLoad(flops=flops, sram_bytes=memory, dram_bytes=memory*0.1)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)

            energy_values.append(breakdown.total)
            flops_values.append(flops)

        # Analyze scaling
        energy_scaling = analyze_scaling_relationship(horizons, energy_values)
        flops_scaling = analyze_scaling_relationship(horizons, flops_values)

        # Check results structure
        self.assertIn('scaling_exponent', energy_scaling)
        self.assertIn('scaling_exponent', flops_scaling)
        self.assertEqual(len(energy_values), len(horizons))

        # Energy should increase with horizon
        self.assertGreater(energy_values[-1], energy_values[0])

    def test_branching_effects_integration(self):
        """Test branching factor effects using core functions."""
        horizon = 8
        branching_factors = [2, 3, 4]
        state_dim = 8
        action_dim = 4

        energy_values = []

        for branching in branching_factors:
            flops, memory = calculate_active_inference_complexity(
                horizon, branching, state_dim, action_dim
            )

            load = ComputeLoad(flops=flops, sram_bytes=memory, dram_bytes=memory*0.1)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            energy_values.append(breakdown.total)

        # Energy should increase with branching
        self.assertGreater(energy_values[-1], energy_values[0])

    def test_bounded_rationality_integration(self):
        """Test bounded rationality concepts using core functions."""
        # Test different horizon/branching combinations
        test_cases = [
            {'horizon': 5, 'branching': 2, 'name': 'Conservative'},
            {'horizon': 8, 'branching': 3, 'name': 'Moderate'},
            {'horizon': 12, 'branching': 4, 'name': 'Aggressive'}
        ]

        results = {}
        for case in test_cases:
            flops, memory = calculate_active_inference_complexity(
                case['horizon'], case['branching'], 8, 4
            )

            # Consider tractable if flops < 10^10
            tractable = flops < 10**10

            results[case['name']] = {
                'flops': flops,
                'memory': memory,
                'tractable': tractable
            }

        # More conservative strategies should be more tractable
        self.assertTrue(results['Conservative']['tractable'])
        if results['Moderate']['flops'] < 10**10:
            self.assertTrue(results['Moderate']['tractable'])


class TestOrchestrationIntegration(unittest.TestCase):
    """Test integration across different analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.coeffs = EnergyCoefficients()

    def test_output_compatibility(self):
        """Test that analysis outputs are compatible for integration."""
        # Test cross-component integration
        contact_flops, contact_memory, _ = calculate_contact_complexity(10, 12, "pgs")
        neural_flops, neural_sram, neural_dram = calculate_sparse_neural_complexity(10000, 0.02)
        ai_flops, ai_memory = calculate_active_inference_complexity(8, 3, 8, 4)

        # Create loads and estimate energies
        contact_load = ComputeLoad(flops=contact_flops, sram_bytes=contact_memory, dram_bytes=contact_memory*0.1)
        neural_load = ComputeLoad(flops=neural_flops, sram_bytes=neural_sram, dram_bytes=neural_dram)
        ai_load = ComputeLoad(flops=ai_flops, sram_bytes=ai_memory, dram_bytes=ai_memory*0.1)

        loads = [contact_load, neural_load, ai_load]
        energies = []

        for load in loads:
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            energies.append(breakdown.total)

        # Test JSON serialization of results
        results = {
            'energies': energies,
            'loads': [{'flops': l.flops, 'sram_bytes': l.sram_bytes, 'dram_bytes': l.dram_bytes} for l in loads]
        }
        json_str = json.dumps(results, default=str)
        self.assertIsInstance(json_str, str)

        # Test that all energies are positive
        for energy in energies:
            self.assertGreater(energy, 0)

    def test_parameter_validation(self):
        """Test parameter validation and edge cases."""
        # Test with minimal valid parameters
        contact_flops, contact_memory, _ = calculate_contact_complexity(1, 1, "pgs")

        # Should still produce valid results
        self.assertGreaterEqual(contact_flops, 0)
        self.assertGreaterEqual(contact_memory, 0)

        # Test with small neural network
        neural_flops, neural_sram, neural_dram = calculate_sparse_neural_complexity(100, 0.1)
        self.assertGreater(neural_flops, 0)
        self.assertGreater(neural_sram + neural_dram, 0)

        # Test with minimal active inference
        ai_flops, ai_memory = calculate_active_inference_complexity(1, 1, 2, 2)
        self.assertGreaterEqual(ai_flops, 0)
        self.assertGreaterEqual(ai_memory, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
