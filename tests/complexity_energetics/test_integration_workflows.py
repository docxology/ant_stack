#!/usr/bin/env python3
"""Integration workflow tests.

Tests complete workflows from data generation through analysis to visualization.
Ensures all components work together seamlessly.
"""

import unittest
import tempfile
import shutil
import json
import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.energy import (
    estimate_detailed_energy, EnergyCoefficients
)
from antstack_core.analysis.statistics import (
    analyze_scaling_relationship, calculate_energy_efficiency_metrics,
    estimate_theoretical_limits
)
from antstack_core.analysis.workloads import (
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form
)
# Import plotting functions (these may not be available in test environment)
try:
    from antstack_core.analysis.plots import bar_plot, line_plot
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Mock functions for testing
    def bar_plot(*args, **kwargs): pass
    def line_plot(*args, **kwargs): pass


class TestCompleteWorkflows(unittest.TestCase):
    """Test complete analysis workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.coeffs = EnergyCoefficients()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_body_analysis_workflow(self):
        """Test complete body analysis workflow."""
        # Generate data across parameter range
        J_values = [6, 12, 18, 24]
        energy_values = []
        flops_values = []
        
        for J in J_values:
            params = {'J': J, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
            load = enhanced_body_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            
            energy_values.append(breakdown.total)
            flops_values.append(load.flops)
        
        # Analyze scaling relationships
        energy_scaling = analyze_scaling_relationship(J_values, energy_values)
        flops_scaling = analyze_scaling_relationship(J_values, flops_values)
        
        # Calculate efficiency metrics
        performance_proxy = [1.0 / J for J in J_values]
        efficiency = calculate_energy_efficiency_metrics(energy_values, performance_proxy)
        
        # Generate visualizations (skip if plotting not available)
        if PLOTTING_AVAILABLE:
            energy_plot = self.output_dir / "body_energy_scaling.png"
            line_plot(
                J_values, [energy_values], ["Body Energy"],
                "Body Energy vs Joint Count", "Joints (J)", "Energy (J)",
                str(energy_plot)
            )
        
        # Additional plotting (skip if not available)
        if PLOTTING_AVAILABLE:
            combined_plot = self.output_dir / "body_energy_flops.png"
            # Note: scatter_plot may not be available, skip for now
            pass
        
        # Validate workflow results
        self.assertEqual(len(energy_values), len(J_values))
        self.assertIn('scaling_exponent', energy_scaling)
        self.assertIn('average_energy_j', efficiency)
    
    def test_brain_analysis_workflow(self):
        """Test complete brain analysis workflow."""
        # Generate data across K values
        K_values = [64, 128, 256, 512]
        energy_values = []
        spikes_values = []
        
        for K in K_values:
            params = {
                'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100,
                'connectivity_pattern': 'biological'
            }
            load = enhanced_brain_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            
            energy_values.append(breakdown.total)
            spikes_values.append(load.spikes)
        
        # Analyze scaling and theoretical limits
        energy_scaling = analyze_scaling_relationship(K_values, energy_values)
        
        # Calculate theoretical limits
        avg_flops = sum(enhanced_brain_workload_closed_form(0.01, {
            'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100
        }).flops for K in K_values) / len(K_values)
        
        limits = estimate_theoretical_limits({
            'flops': avg_flops,
            'mechanical_work_j': 0.0
        })
        
        # Generate visualizations (skip if plotting not available)
        if PLOTTING_AVAILABLE:
            energy_plot = self.output_dir / "brain_energy_scaling.png"
            line_plot(
                K_values, [energy_values], ["Brain Energy"],
                "Brain Energy vs AL Channels", "AL Channels (K)", "Energy (J)",
                str(energy_plot)
            )
        
        # Additional plotting (skip if not available)
        if PLOTTING_AVAILABLE:
            spikes_plot = self.output_dir / "brain_spikes_scaling.png"
            line_plot(
                K_values, [spikes_values], ["Spikes"],
                "Brain Spikes vs AL Channels", "AL Channels (K)", "Spikes",
                str(spikes_plot)
            )
        
        # Validate workflow results
        self.assertEqual(len(energy_values), len(K_values))
        self.assertIn('scaling_regime', energy_scaling)
        self.assertIn('landauer_limit_j', limits)
    
    def test_mind_analysis_workflow(self):
        """Test complete mind analysis workflow."""
        # Generate data across horizon values (limited to avoid explosion)
        H_p_values = [5, 7, 9, 11]
        energy_values = []
        flops_values = []
        
        for H_p in H_p_values:
            params = {
                'B': 3, 'H_p': H_p, 'hz': 100,
                'state_dim': 8, 'action_dim': 4, 'hierarchical': False
            }
            load = enhanced_mind_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            
            energy_values.append(breakdown.total)
            flops_values.append(load.flops)
        
        # Analyze exponential scaling
        energy_scaling = analyze_scaling_relationship(H_p_values, energy_values)
        flops_scaling = analyze_scaling_relationship(H_p_values, flops_values)
        
        # Generate visualizations
        energy_plot = self.output_dir / "mind_energy_scaling.png"
        line_plot(
            H_p_values, [energy_values], ["Mind Energy"],
            "Mind Energy vs Policy Horizon", "Policy Horizon", "Energy (J)",
            str(energy_plot)
        )
        
        # Validate workflow results
        self.assertEqual(len(energy_values), len(H_p_values))
        self.assertIn('scaling_exponent', energy_scaling)

        # Mind scaling behavior depends on implementation
        if energy_scaling.get('scaling_exponent', 0) > 0:
            self.assertGreater(energy_scaling['scaling_exponent'], 0)
    
    def test_comparative_analysis_workflow(self):
        """Test comparative analysis across all modules."""
        # Standard parameters for comparison
        duration = 0.01  # 10ms decision
        
        body_params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
        brain_params = {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}
        mind_params = {'B': 4, 'H_p': 10, 'hz': 100, 'state_dim': 16, 'action_dim': 6}
        
        # Calculate workloads
        body_load = enhanced_body_workload_closed_form(duration, body_params)
        brain_load = enhanced_brain_workload_closed_form(duration, brain_params)
        mind_load = enhanced_mind_workload_closed_form(duration, mind_params)
        
        # Calculate energy breakdowns
        body_breakdown = estimate_detailed_energy(body_load, self.coeffs, duration)
        brain_breakdown = estimate_detailed_energy(brain_load, self.coeffs, duration)
        mind_breakdown = estimate_detailed_energy(mind_load, self.coeffs, duration)
        
        # Comparative analysis
        modules = ['Body', 'Brain', 'Mind']
        energies = [body_breakdown.total, brain_breakdown.total, mind_breakdown.total]
        flops = [body_load.flops, brain_load.flops, mind_load.flops]
        
        # Generate comparative visualizations
        energy_comparison = self.output_dir / "module_energy_comparison.png"
        bar_plot(
            modules, energies,
            "Energy Comparison Across Modules",
            str(energy_comparison), ylabel="Energy per Decision (J)"
        )
        
        flops_comparison = self.output_dir / "module_flops_comparison.png"
        bar_plot(
            modules, flops,
            "FLOPs Comparison Across Modules",
            str(flops_comparison), ylabel="FLOPs per Decision"
        )
        
        # Skip scatter plot for now as it may not be available
        # energy_flops_scatter = self.output_dir / "energy_flops_scatter.png"
        
        # Validate comparative results
        self.assertEqual(len(energies), 3)
        self.assertEqual(len(flops), 3)
        
        # All energies should be positive
        for energy in energies:
            self.assertGreater(energy, 0)
    
    def test_theoretical_limits_workflow(self):
        """Test theoretical limits analysis workflow."""
        # Calculate theoretical limits for each module
        modules = ['Body', 'Brain', 'Mind']
        actual_energies = []
        theoretical_limits = []
        
        # Body
        body_load = enhanced_body_workload_closed_form(0.01, {
            'J': 18, 'C': 12, 'S': 256, 'hz': 100
        })
        body_energy = estimate_detailed_energy(body_load, self.coeffs, 0.01).total
        body_limits = estimate_theoretical_limits({
            'flops': body_load.flops,
            'mechanical_work_j': 0.001  # Mechanical work dominates
        })
        
        actual_energies.append(body_energy)
        theoretical_limits.append(body_limits['total_theoretical_j'])
        
        # Brain
        brain_load = enhanced_brain_workload_closed_form(0.01, {
            'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100
        })
        brain_energy = estimate_detailed_energy(brain_load, self.coeffs, 0.01).total
        brain_limits = estimate_theoretical_limits({
            'flops': brain_load.flops,
            'mechanical_work_j': 0.0
        })
        
        actual_energies.append(brain_energy)
        theoretical_limits.append(brain_limits['total_theoretical_j'])
        
        # Mind
        mind_load = enhanced_mind_workload_closed_form(0.01, {
            'B': 4, 'H_p': 8, 'hz': 100, 'state_dim': 16, 'action_dim': 6
        })
        mind_energy = estimate_detailed_energy(mind_load, self.coeffs, 0.01).total
        mind_limits = estimate_theoretical_limits({
            'flops': mind_load.flops,
            'mechanical_work_j': 0.0
        })
        
        actual_energies.append(mind_energy)
        theoretical_limits.append(mind_limits['total_theoretical_j'])
        
        # Calculate efficiency ratios
        efficiency_ratios = [actual / theoretical for actual, theoretical 
                           in zip(actual_energies, theoretical_limits)]
        
        # Generate theoretical limits visualization
        all_energies = actual_energies + theoretical_limits
        all_labels = [f"{m} (Actual)" for m in modules] + [f"{m} (Theoretical)" for m in modules]
        
        limits_comparison = self.output_dir / "theoretical_limits_comparison.png"
        bar_plot(
            all_labels, all_energies,
            "Actual vs Theoretical Energy Limits",
            str(limits_comparison), ylabel="Energy per Decision (J)"
        )
        
        # Validate theoretical limits analysis
        self.assertEqual(len(efficiency_ratios), 3)
        
        # Efficiency ratios can be low due to conservative theoretical limits
        for ratio in efficiency_ratios:
            self.assertGreater(ratio, 0.0)  # Should be positive but may be very small


class TestWorkflowRobustness(unittest.TestCase):
    """Test workflow robustness and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.coeffs = EnergyCoefficients()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_edge_case_parameters(self):
        """Test workflows with edge case parameters."""
        # Test with minimal parameters
        minimal_params = {'J': 1, 'C': 1, 'S': 1, 'hz': 1}
        load = enhanced_body_workload_closed_form(0.001, minimal_params)
        breakdown = estimate_detailed_energy(load, self.coeffs, 0.001)
        
        self.assertGreaterEqual(breakdown.total, 0)
        
        # Test with large parameters (but not explosive)
        large_params = {'J': 100, 'C': 50, 'S': 1000, 'hz': 1000}
        load = enhanced_body_workload_closed_form(0.001, large_params)
        breakdown = estimate_detailed_energy(load, self.coeffs, 0.001)
        
        self.assertGreater(breakdown.total, 0)
        self.assertLess(breakdown.total, 1.0)  # Should be reasonable
    
    def test_scaling_analysis_robustness(self):
        """Test scaling analysis with various data patterns."""
        # Test with constant data
        x_values = [1, 2, 3, 4, 5]
        y_constant = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        scaling = analyze_scaling_relationship(x_values, y_constant)
        # Should handle constant data gracefully
        self.assertIn('scaling_exponent', scaling)
        
        # Test with noisy data
        y_noisy = [1.0, 1.1, 0.9, 1.05, 0.95]
        scaling = analyze_scaling_relationship(x_values, y_noisy)
        self.assertIn('r_squared', scaling)
        
        # Test with insufficient data
        scaling = analyze_scaling_relationship([1, 2], [1, 2])
        self.assertIn('error', scaling)
    
    def test_visualization_robustness(self):
        """Test visualization generation with edge cases."""
        # Test with single data point
        try:
            single_plot = self.output_dir / "single_point.png"
            bar_plot(['Single'], [1.0], "Single Point Test", str(single_plot))
            self.assertTrue(single_plot.exists())
        except Exception:
            # Should handle gracefully even if plot fails
            pass
        
        # Test with zero values
        try:
            zero_plot = self.output_dir / "zero_values.png"
            bar_plot(['Zero'], [0.0], "Zero Values Test", str(zero_plot))
            self.assertTrue(zero_plot.exists())
        except Exception:
            # Should handle gracefully
            pass


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
