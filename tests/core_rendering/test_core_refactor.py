#!/usr/bin/env python3
"""Comprehensive tests for refactored antstack_core system.

Tests the new modular architecture including:
- Core scientific analysis functionality
- Figure generation and management
- Energy estimation and scaling analysis
- Paper configuration and validation
- Integration with existing workflows

Following .cursorrules principles:
- Test-driven development with real methods (no mocks)
- Comprehensive validation and error checking
- Professional, well-documented test cases
- Statistical validation of scientific methods

References:
- Testing scientific code: https://doi.org/10.1371/journal.pcbi.1004668
- Software testing best practices: https://docs.python.org/3/library/unittest.html
"""

import unittest
import tempfile
import shutil
import json
import sys
import os
from pathlib import Path
import yaml

# Add the new core package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from antstack_core.analysis import (
        EnergyCoefficients, ComputeLoad, EnergyBreakdown,
        estimate_detailed_energy, pj_to_j, calculate_energy_efficiency_metrics
    )
    from antstack_core.figures import bar_plot, line_plot, scatter_plot
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core modules not available: {e}")
    CORE_AVAILABLE = False


class TestCoreAnalysis(unittest.TestCase):
    """Test core analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CORE_AVAILABLE:
            self.skipTest("Core modules not available")
            
        self.coeffs = EnergyCoefficients()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def test_energy_coefficients(self):
        """Test energy coefficient dataclass and methods."""
        # Test default values (should match paper config)
        self.assertEqual(self.coeffs.flops_pj, 1.0)
        self.assertEqual(self.coeffs.sram_pj_per_byte, 0.1)
        self.assertEqual(self.coeffs.dram_pj_per_byte, 20.0)
        self.assertEqual(self.coeffs.baseline_w, 0.50)  # Matches paper config
        
        # Test dictionary conversion
        coeffs_dict = self.coeffs.to_dict()
        self.assertIsInstance(coeffs_dict, dict)
        self.assertIn('flops_pj', coeffs_dict)
        
        # Test technology scaling
        scaled = self.coeffs.scale_by_technology(14.0)  # 14nm node
        self.assertNotEqual(scaled.flops_pj, self.coeffs.flops_pj)
        self.assertGreater(scaled.flops_pj, self.coeffs.flops_pj)
    
    def test_compute_load(self):
        """Test compute load specification and operations."""
        load = ComputeLoad(flops=1000, sram_bytes=500, dram_bytes=100)
        
        # Test basic properties
        self.assertEqual(load.flops, 1000)
        self.assertEqual(load.sram_bytes, 500)
        
        # Test scaling
        scaled = load.scale(2.0)
        self.assertEqual(scaled.flops, 2000)
        self.assertEqual(scaled.sram_bytes, 1000)
        
        # Test dictionary conversion
        load_dict = load.to_dict()
        self.assertIsInstance(load_dict, dict)
        self.assertEqual(load_dict['flops'], 1000)
    
    def test_energy_breakdown(self):
        """Test energy breakdown analysis and metrics."""
        breakdown = EnergyBreakdown(
            actuation=0.01,
            sensing=0.005,
            compute_flops=0.002,
            compute_memory=0.008,
            baseline=0.001
        )
        
        # Test computed properties
        self.assertEqual(breakdown.total_compute, 0.01)  # 0.002 + 0.008
        self.assertAlmostEqual(breakdown.total, 0.026, places=6)  # Sum of all components
        
        # Test energy fraction calculation
        compute_fraction = breakdown.compute_fraction
        self.assertGreater(compute_fraction, 0)
        self.assertLessEqual(compute_fraction, 1.0)
        
        # Test dominant component identification
        dominant = breakdown.dominant_component()
        self.assertEqual(dominant, 'Actuation')  # Highest value
        
    def test_detailed_energy_estimation(self):
        """Test comprehensive energy estimation pipeline."""
        load = ComputeLoad(
            flops=10000,
            sram_bytes=1000,
            dram_bytes=500,
            spikes=100
        )
        
        breakdown = estimate_detailed_energy(
            load, self.coeffs, duration_s=0.01,
            actuation_energy=0.01, sensing_energy=0.005
        )
        
        # Validate breakdown components
        self.assertGreater(breakdown.compute_flops, 0)
        self.assertGreater(breakdown.compute_memory, 0)  
        self.assertGreater(breakdown.compute_spikes, 0)
        self.assertEqual(breakdown.actuation, 0.01)
        self.assertEqual(breakdown.sensing, 0.005)
        
        # Test total energy consistency
        manual_total = (breakdown.actuation + breakdown.sensing + 
                       breakdown.total_compute + breakdown.baseline)
        self.assertAlmostEqual(breakdown.total, manual_total, places=10)
        
        # Test uncertainty quantification
        self.assertGreater(breakdown.total_uncertainty, 0)
        self.assertLess(breakdown.total_uncertainty, breakdown.total)
    
    def test_energy_efficiency_metrics(self):
        """Test energy efficiency calculation and validation."""
        energy_values = [0.01, 0.015, 0.012, 0.018]
        performance_values = [100, 90, 95, 85]
        
        metrics = calculate_energy_efficiency_metrics(energy_values, performance_values)
        
        # Validate metric structure
        self.assertIn('average_energy_j', metrics)
        self.assertIn('performance_per_joule', metrics)
        self.assertIn('efficiency_score', metrics)
        
        # Test error handling
        empty_metrics = calculate_energy_efficiency_metrics([], [])
        self.assertIn('error', empty_metrics)
        
        mismatched_metrics = calculate_energy_efficiency_metrics([0.01], [100, 90])
        self.assertIn('error', mismatched_metrics)
    
    def test_unit_conversions(self):
        """Test energy unit conversion utilities."""
        # Test picojoule to joule conversion
        pj_value = 1000.0
        j_value = pj_to_j(pj_value)
        expected = 1000.0 * 1e-12
        self.assertEqual(j_value, expected)
        
        # Test zero and negative values
        self.assertEqual(pj_to_j(0), 0)
        self.assertEqual(pj_to_j(-100), -100 * 1e-12)


class TestCoreFigures(unittest.TestCase):
    """Test core figure generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CORE_AVAILABLE:
            self.skipTest("Core modules not available")
            
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_bar_plot_generation(self):
        """Test publication-quality bar plot generation."""
        labels = ['AntBody', 'AntBrain', 'AntMind']
        values = [0.015, 0.008, 0.003]
        yerr = [0.002, 0.001, 0.0005]
        
        output_path = Path(self.temp_dir) / "test_bar.png"
        
        # Generate plot (should not raise exceptions)
        bar_plot(
            labels, values, 
            "Energy Breakdown by Module",
            str(output_path),
            ylabel="Energy (J)",
            yerr=yerr
        )
        
        # Verify file creation (if matplotlib available)
        try:
            import matplotlib.pyplot as plt
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
        except ImportError:
            pass  # Skip file check if matplotlib not available
    
    def test_line_plot_generation(self):
        """Test line plot with scaling analysis."""
        x = [64, 128, 256, 512, 1024]
        y1 = [0.001, 0.004, 0.016, 0.064, 0.256]  # Quadratic scaling
        y2 = [0.002, 0.004, 0.008, 0.016, 0.032]  # Linear scaling
        
        output_path = Path(self.temp_dir) / "test_line.png"
        
        # Generate line plot
        line_plot(
            x, [y1, y2], ['Quadratic', 'Linear'],
            "Scaling Analysis", "Parameter K", "Energy (J)",
            str(output_path)
        )
        
        # Verify file creation
        try:
            import matplotlib.pyplot as plt
            self.assertTrue(output_path.exists())
        except ImportError:
            pass
    
    def test_scatter_plot_generation(self):
        """Test scatter plot with statistical analysis."""
        import random
        random.seed(42)  # Reproducible test data
        
        x = [random.uniform(100, 1000) for _ in range(20)]
        y = [2.5 * xi + random.uniform(-50, 50) for xi in x]  # Linear with noise
        
        output_path = Path(self.temp_dir) / "test_scatter.png"
        
        # Generate scatter plot
        scatter_plot(
            x, y,
            "Energy vs. Complexity",
            "FLOPs", "Energy (J)",
            str(output_path)
        )
        
        # Verify file creation
        try:
            import matplotlib.pyplot as plt
            self.assertTrue(output_path.exists())
        except ImportError:
            pass


class TestPaperConfiguration(unittest.TestCase):
    """Test paper configuration and validation system."""
    
    def test_paper_config_loading(self):
        """Test paper configuration file loading and validation."""
        # Test ant_stack configuration
        ant_stack_config = Path("papers/ant_stack/paper_config.yaml")
        if ant_stack_config.exists():
            with open(ant_stack_config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            self.assertIn('paper', config)
            self.assertIn('content', config)
            self.assertIn('build', config)
            
            # Validate paper metadata
            paper_info = config['paper']
            self.assertEqual(paper_info['name'], 'ant_stack')
            self.assertIn('title', paper_info)
            self.assertIn('author', paper_info)
            
            # Validate content structure
            content = config['content']
            self.assertIn('files', content)
            self.assertIsInstance(content['files'], list)
        
        # Test complexity_energetics configuration  
        ce_config = Path("papers/complexity_energetics/paper_config.yaml")
        if ce_config.exists():
            with open(ce_config, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate computational analysis features
            self.assertIn('analysis', config)
            analysis = config['analysis']
            self.assertIn('generated_figures', analysis)
            
            # Validate energy coefficients
            self.assertIn('energy_coefficients', config)
            coeffs = config['energy_coefficients']
            self.assertIn('e_flop_pj', coeffs)
            self.assertIn('baseline_w', coeffs)
            self.assertEqual(coeffs['baseline_w'], 0.50)  # Should match paper config


class TestIntegrationWorkflow(unittest.TestCase):
    """Test integration between refactored components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        if not CORE_AVAILABLE:
            self.skipTest("Core modules not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.coeffs = EnergyCoefficients()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_analysis_to_visualization_pipeline(self):
        """Test complete pipeline from analysis to visualization."""
        # Generate test data (simulating different workload parameters)
        K_values = [64, 128, 256, 512]
        energy_values = []
        flops_values = []
        
        for K in K_values:
            # Simulate brain workload scaling
            flops = K * 1000 + K**1.5 * 100  # Sub-quadratic scaling
            load = ComputeLoad(flops=flops, sram_bytes=K*10, dram_bytes=K*2)
            
            breakdown = estimate_detailed_energy(load, self.coeffs, 0.01)
            energy_values.append(breakdown.total)
            flops_values.append(flops)
        
        # Generate efficiency metrics
        performance_proxy = [1.0/K for K in K_values]  # Inverse relationship
        efficiency = calculate_energy_efficiency_metrics(energy_values, performance_proxy)
        
        # Validate pipeline results
        self.assertEqual(len(energy_values), len(K_values))
        self.assertIn('efficiency_score', efficiency)
        self.assertGreater(efficiency['efficiency_score'], 0)
        
        # Generate visualization
        output_path = Path(self.temp_dir) / "pipeline_test.png"
        line_plot(
            K_values, [energy_values], ['Energy'],
            "Brain Module Energy Scaling", "Channels (K)", "Energy (J)",
            str(output_path)
        )
        
        # Validate visualization output
        try:
            import matplotlib.pyplot as plt
            self.assertTrue(output_path.exists())
        except ImportError:
            pass


if __name__ == '__main__':
    unittest.main()
