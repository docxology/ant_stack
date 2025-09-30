#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.theoretical_limits module.

Tests theoretical limits analysis functionality including:
- Landauer's principle calculations
- Thermodynamic efficiency bounds
- Information-theoretic limits
- Efficiency gap analysis
- Optimization recommendations
- Cross-validation with empirical data

Following .cursorrules principles:
- Rigorous theoretical calculations
- Comprehensive validation against known limits
- Physical accuracy verification
- Scientific reproducibility
"""

import unittest
import math
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.theoretical_limits import (
    TheoreticalLimit,
    EfficiencyAnalysis,
    ModuleTheoreticalAnalysis,
    TheoreticalLimitsAnalyzer
)


class TestTheoreticalLimit(unittest.TestCase):
    """Test TheoreticalLimit dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.limit = TheoreticalLimit(
            limit_type="landauer",
            value_j=1.4e-21,
            description="Minimum energy for bit erasure at room temperature",
            assumptions=["Room temperature (298K)", "Ideal thermodynamic efficiency"],
            uncertainty_factor=1.0,
            confidence_level=1.0
        )

    def test_theoretical_limit_creation(self):
        """Test TheoreticalLimit creation."""
        self.assertEqual(self.limit.limit_type, "landauer")
        self.assertAlmostEqual(self.limit.value_j, 1.4e-21, places=23)
        self.assertEqual(self.limit.description, "Minimum energy for bit erasure at room temperature")
        self.assertEqual(len(self.limit.assumptions), 2)

    def test_theoretical_limit_defaults(self):
        """Test TheoreticalLimit with defaults."""
        minimal_limit = TheoreticalLimit(
            limit_type="test",
            value_j=1.0,
            description="Test limit"
        )

        self.assertIsNone(minimal_limit.uncertainty_factor)
        self.assertIsNone(minimal_limit.confidence_level)

    def test_theoretical_limit_with_uncertainty(self):
        """Test TheoreticalLimit with uncertainty."""
        uncertain_limit = TheoreticalLimit(
            limit_type="approximate",
            value_j=2.0,
            description="Approximate limit",
            assumptions=["Approximation used"],
            uncertainty_factor=0.1,  # 10% uncertainty
            confidence_level=0.95
        )

        self.assertEqual(uncertain_limit.uncertainty_factor, 0.1)
        self.assertEqual(uncertain_limit.confidence_level, 0.95)


class TestEfficiencyAnalysis(unittest.TestCase):
    """Test EfficiencyAnalysis dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analysis = EfficiencyAnalysis(
            actual_energy_j=1e-12,
            theoretical_limit_j=1.4e-21,
            efficiency_ratio=7.14e8,  # Very low efficiency
            optimization_potential=0.999999999,  # Almost 100% optimization potential
            limit_type="landauer",
            bottleneck_identified="Memory access patterns"
        )

    def test_efficiency_analysis_creation(self):
        """Test EfficiencyAnalysis creation."""
        self.assertAlmostEqual(self.analysis.actual_energy_j, 1e-12)
        self.assertAlmostEqual(self.analysis.theoretical_limit_j, 1.4e-21)
        self.assertAlmostEqual(self.analysis.efficiency_ratio, 7.14e8, places=2)
        self.assertAlmostEqual(self.analysis.optimization_potential, 0.999999999, places=8)
        self.assertEqual(self.analysis.limit_type, "landauer")
        self.assertEqual(self.analysis.bottleneck_identified, "Memory access patterns")

    def test_efficiency_analysis_defaults(self):
        """Test EfficiencyAnalysis with minimal data."""
        minimal_analysis = EfficiencyAnalysis(
            actual_energy_j=1e-10,
            theoretical_limit_j=1e-20,
            efficiency_ratio=1000.0,
            optimization_potential=0.999,
            limit_type="thermodynamic"
        )

        self.assertIsNone(minimal_analysis.bottleneck_identified)


class TestModuleTheoreticalAnalysis(unittest.TestCase):
    """Test ModuleTheoreticalAnalysis dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.landauer_limit = TheoreticalLimit(
            limit_type="landauer",
            value_j=1.4e-21,
            description="Landauer limit",
            assumptions=["Room temperature"]
        )

        self.efficiency = EfficiencyAnalysis(
            actual_energy_j=1e-10,
            theoretical_limit_j=1.4e-21,
            efficiency_ratio=7.14e7,
            optimization_potential=0.999999986,
            limit_type="landauer"
        )

        self.module_analysis = ModuleTheoreticalAnalysis(
            module_name="AntBrain",
            limits=[self.landauer_limit],
            efficiency_analysis=self.efficiency,
            dominant_limit="landauer",
            optimization_recommendations=[
                "Optimize memory access patterns",
                "Implement neuromorphic computing",
                "Reduce computational redundancy"
            ]
        )

    def test_module_theoretical_analysis_creation(self):
        """Test ModuleTheoreticalAnalysis creation."""
        self.assertEqual(self.module_analysis.module_name, "AntBrain")
        self.assertEqual(len(self.module_analysis.limits), 1)
        self.assertIsNotNone(self.module_analysis.efficiency_analysis)
        self.assertEqual(self.module_analysis.dominant_limit, "landauer")
        self.assertEqual(len(self.module_analysis.optimization_recommendations), 3)

    def test_module_theoretical_analysis_minimal(self):
        """Test ModuleTheoreticalAnalysis with minimal data."""
        minimal_analysis = ModuleTheoreticalAnalysis(
            module_name="TestModule",
            limits=[]
        )

        self.assertEqual(minimal_analysis.module_name, "TestModule")
        self.assertEqual(len(minimal_analysis.limits), 0)
        self.assertIsNone(minimal_analysis.efficiency_analysis)
        self.assertIsNone(minimal_analysis.dominant_limit)
        self.assertIsNone(minimal_analysis.optimization_recommendations)


class TestTheoreticalLimitsAnalyzer(unittest.TestCase):
    """Test TheoreticalLimitsAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TheoreticalLimitsAnalyzer()

        # Test parameters
        self.bits_processed = 1e6  # 1 million bits
        self.mechanical_work = 1e-6  # 1 microjoule
        self.entropy_change = 100.0  # 100 J/K
        self.information_processed = 1e9  # 1 billion bits/second

    def test_theoretical_limits_analyzer_creation(self):
        """Test TheoreticalLimitsAnalyzer creation."""
        self.assertIsInstance(self.analyzer, TheoreticalLimitsAnalyzer)

        # Check fundamental constants
        self.assertAlmostEqual(self.analyzer.k_B, 1.380649e-23, places=25)
        self.assertAlmostEqual(self.analyzer.T_room, 298.15, places=2)
        self.assertAlmostEqual(self.analyzer.landauer_limit, 1.4e-21, places=22)

    def test_calculate_landauer_limits(self):
        """Test Landauer's principle limit calculation."""
        limit = self.analyzer.calculate_landauer_limits(self.bits_processed)

        self.assertIsInstance(limit, TheoreticalLimit)
        self.assertEqual(limit.limit_type, "landauer")
        self.assertAlmostEqual(limit.value_j, self.bits_processed * self.analyzer.landauer_limit, places=15)

        # Check assumptions
        self.assertIn("Irreversible computation", limit.assumptions)
        self.assertIn("Room temperature", limit.assumptions)

    def test_calculate_landauer_limits_edge_cases(self):
        """Test Landauer's limit calculation with edge cases."""
        # Zero bits
        zero_limit = self.analyzer.calculate_landauer_limits(0)
        self.assertEqual(zero_limit.value_j, 0)

        # Very large number of bits
        large_bits = 1e15
        large_limit = self.analyzer.calculate_landauer_limits(large_bits)
        self.assertAlmostEqual(large_limit.value_j, large_bits * self.analyzer.landauer_limit, places=10)

    def test_calculate_thermodynamic_limits(self):
        """Test thermodynamic limit calculation."""
        limit = self.analyzer.calculate_thermodynamic_limits(
            self.mechanical_work, self.entropy_change
        )

        self.assertIsInstance(limit, TheoreticalLimit)
        self.assertEqual(limit.limit_type, "thermodynamic")

        # Energy should be positive
        self.assertGreater(limit.value_j, 0)

        # Check thermodynamic assumptions
        self.assertIn("Second law of thermodynamics", limit.assumptions)

    def test_calculate_thermodynamic_limits_zero_entropy(self):
        """Test thermodynamic limits with zero entropy change."""
        limit = self.analyzer.calculate_thermodynamic_limits(self.mechanical_work, 0)

        self.assertIsInstance(limit, TheoreticalLimit)
        # With zero entropy change, energy should equal mechanical work
        self.assertAlmostEqual(limit.value_j, self.mechanical_work, places=10)

    def test_calculate_information_theoretic_limits(self):
        """Test information-theoretic limit calculation."""
        limit = self.analyzer.calculate_information_theoretic_limits(self.information_processed)

        self.assertIsInstance(limit, TheoreticalLimit)
        self.assertEqual(limit.limit_type, "information_theoretic")

        # Energy should be positive
        self.assertGreater(limit.value_j, 0)

        # Check information theory assumptions
        self.assertIn("Shannon entropy", limit.assumptions)

    def test_calculate_comprehensive_limits(self):
        """Test comprehensive limit calculation for a module."""
        comprehensive_limits = self.analyzer.calculate_comprehensive_limits(
            bits_processed=self.bits_processed,
            mechanical_work=self.mechanical_work,
            entropy_change=self.entropy_change,
            information_processed=self.information_processed
        )

        self.assertIsInstance(comprehensive_limits, dict)
        self.assertIn("landauer", comprehensive_limits)
        self.assertIn("thermodynamic", comprehensive_limits)
        self.assertIn("information_theoretic", comprehensive_limits)

        # All limits should be TheoreticalLimit objects
        for limit in comprehensive_limits.values():
            self.assertIsInstance(limit, TheoreticalLimit)

    def test_analyze_module_efficiency(self):
        """Test module efficiency analysis."""
        actual_energy = 1e-9  # 1 nanojoule
        theoretical_energy = 1.4e-21  # Landauer limit

        efficiency = self.analyzer.analyze_module_efficiency(
            actual_energy, theoretical_energy, "landauer"
        )

        self.assertIsInstance(efficiency, EfficiencyAnalysis)
        self.assertEqual(efficiency.actual_energy_j, actual_energy)
        self.assertEqual(efficiency.theoretical_limit_j, theoretical_energy)
        self.assertEqual(efficiency.limit_type, "landauer")

        # Calculate expected efficiency ratio
        expected_ratio = actual_energy / theoretical_energy
        self.assertAlmostEqual(efficiency.efficiency_ratio, expected_ratio, places=10)

    def test_analyze_module_efficiency_perfect_efficiency(self):
        """Test efficiency analysis with perfect efficiency."""
        actual_energy = 1.4e-21  # Exactly at Landauer limit
        theoretical_energy = 1.4e-21

        efficiency = self.analyzer.analyze_module_efficiency(
            actual_energy, theoretical_energy, "landauer"
        )

        self.assertAlmostEqual(efficiency.efficiency_ratio, 1.0, places=10)
        self.assertAlmostEqual(efficiency.optimization_potential, 0.0, places=10)

    def test_perform_module_analysis(self):
        """Test complete module analysis."""
        module_analysis = self.analyzer.perform_module_analysis(
            module_name="TestModule",
            actual_energy_j=1e-9,
            bits_processed=self.bits_processed,
            mechanical_work=self.mechanical_work,
            entropy_change=self.entropy_change,
            information_processed=self.information_processed
        )

        self.assertIsInstance(module_analysis, ModuleTheoreticalAnalysis)
        self.assertEqual(module_analysis.module_name, "TestModule")

        # Should have multiple limits
        self.assertGreater(len(module_analysis.limits), 1)

        # Should have efficiency analysis
        self.assertIsNotNone(module_analysis.efficiency_analysis)

        # Should have dominant limit identified
        self.assertIsNotNone(module_analysis.dominant_limit)

    def test_identify_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        current_energy = 1e-9
        theoretical_limit = 1.4e-21
        limit_type = "landauer"

        opportunities = self.analyzer.identify_optimization_opportunities(
            current_energy, theoretical_limit, limit_type
        )

        self.assertIsInstance(opportunities, list)
        self.assertGreater(len(opportunities), 0)

        # Should contain actionable recommendations
        for opportunity in opportunities:
            self.assertIsInstance(opportunity, str)
            self.assertGreater(len(opportunity), 10)  # Reasonable length

    def test_compare_with_empirical_data(self):
        """Test comparison with empirical data."""
        empirical_energy = 1e-9
        empirical_data = {
            "efficiency": 0.7,
            "bottleneck": "memory_access",
            "optimization_potential": 0.85
        }

        comparison = self.analyzer.compare_with_empirical_data(
            empirical_energy, empirical_data
        )

        self.assertIsInstance(comparison, dict)
        self.assertIn("empirical_vs_theoretical", comparison)
        self.assertIn("validation_score", comparison)

    def test_generate_efficiency_report(self):
        """Test efficiency report generation."""
        module_analysis = self.analyzer.perform_module_analysis(
            module_name="TestModule",
            actual_energy_j=1e-9,
            bits_processed=1e6
        )

        report = self.analyzer.generate_efficiency_report(module_analysis)

        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)  # Should be substantial

        # Should contain key information
        self.assertIn("TestModule", report)
        self.assertIn("efficiency", report.lower())
        self.assertIn("optimization", report.lower())


class TestTheoreticalLimitsIntegration(unittest.TestCase):
    """Test integration between theoretical limits components."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TheoreticalLimitsAnalyzer()

        # Realistic parameters for a neuromorphic system
        self.test_params = {
            "module_name": "NeuromorphicProcessor",
            "actual_energy": 1e-9,  # 1 nanojoule per operation
            "bits_processed": 1e8,  # 100 million bits
            "mechanical_work": 1e-12,  # 1 picojoule
            "entropy_change": 1.0,  # 1 J/K
            "information_processed": 1e10  # 10 billion bits/second
        }

    def test_end_to_end_theoretical_analysis(self):
        """Test complete theoretical analysis workflow."""
        # Step 1: Calculate individual limits
        landauer = self.analyzer.calculate_landauer_limits(self.test_params["bits_processed"])
        thermodynamic = self.analyzer.calculate_thermodynamic_limits(
            self.test_params["mechanical_work"], self.test_params["entropy_change"]
        )
        info_theoretic = self.analyzer.calculate_information_theoretic_limits(
            self.test_params["information_processed"]
        )

        # All should be valid TheoreticalLimit objects
        self.assertIsInstance(landauer, TheoreticalLimit)
        self.assertIsInstance(thermodynamic, TheoreticalLimit)
        self.assertIsInstance(info_theoretic, TheoreticalLimit)

        # Step 2: Analyze efficiency
        efficiency = self.analyzer.analyze_module_efficiency(
            self.test_params["actual_energy"], landauer.value_j, "landauer"
        )
        self.assertIsInstance(efficiency, EfficiencyAnalysis)

        # Step 3: Perform comprehensive module analysis
        module_analysis = self.analyzer.perform_module_analysis(
            module_name=self.test_params["module_name"],
            actual_energy_j=self.test_params["actual_energy"],
            bits_processed=self.test_params["bits_processed"],
            mechanical_work=self.test_params["mechanical_work"],
            entropy_change=self.test_params["entropy_change"],
            information_processed=self.test_params["information_processed"]
        )

        self.assertIsInstance(module_analysis, ModuleTheoreticalAnalysis)

        # Step 4: Generate report
        report = self.analyzer.generate_efficiency_report(module_analysis)
        self.assertIsInstance(report, str)

    def test_cross_validation_different_limits(self):
        """Test cross-validation between different theoretical limits."""
        bits = 1e6

        # Calculate same system using different theoretical frameworks
        landauer_limit = self.analyzer.calculate_landauer_limits(bits)
        thermodynamic_limit = self.analyzer.calculate_thermodynamic_limits(1e-12, 1.0)

        # Both should provide positive energy bounds
        self.assertGreater(landauer_limit.value_j, 0)
        self.assertGreater(thermodynamic_limit.value_j, 0)

        # Landauer's limit should be much smaller for information processing
        self.assertLess(landauer_limit.value_j, thermodynamic_limit.value_j)

    def test_scaling_analysis_with_limits(self):
        """Test how theoretical limits scale with system parameters."""
        bit_counts = [1e3, 1e4, 1e5, 1e6, 1e7]

        limits = []
        for bits in bit_counts:
            limit = self.analyzer.calculate_landauer_limits(bits)
            limits.append(limit.value_j)

        # Limits should scale linearly with bit count
        for i in range(1, len(limits)):
            ratio = limits[i] / limits[i-1]
            expected_ratio = bit_counts[i] / bit_counts[i-1]
            self.assertAlmostEqual(ratio, expected_ratio, places=10)

    def test_efficiency_gap_analysis(self):
        """Test analysis of efficiency gaps."""
        # Simulate different system efficiencies
        efficiencies = [0.01, 0.1, 0.5, 0.9, 0.99]  # 1% to 99%

        landauer_energy = 1.4e-21  # Fixed theoretical limit

        gap_analyses = []
        for eff in efficiencies:
            actual_energy = landauer_energy / eff
            analysis = self.analyzer.analyze_module_efficiency(
                actual_energy, landauer_energy, "landauer"
            )
            gap_analyses.append(analysis)

        # Optimization potential should increase as efficiency decreases
        for i in range(1, len(gap_analyses)):
            self.assertGreater(
                gap_analyses[i].optimization_potential,
                gap_analyses[i-1].optimization_potential
            )

    def test_theoretical_limits_temperature_dependence(self):
        """Test temperature dependence of theoretical limits."""
        # Landauer's limit depends on temperature
        # kT * ln(2) where T is temperature

        # Test at different temperatures
        temperatures = [77, 298, 373]  # Liquid N2, Room temp, Boiling water

        limits_by_temp = {}
        for temp in temperatures:
            # Create analyzer with different temperature
            analyzer_temp = TheoreticalLimitsAnalyzer()
            analyzer_temp.T_room = temp
            analyzer_temp.kT = analyzer_temp.k_B * temp
            analyzer_temp.landauer_limit = analyzer_temp.kT * math.log(2)

            limit = analyzer_temp.calculate_landauer_limits(1e6)
            limits_by_temp[temp] = limit.value_j

        # Limits should increase with temperature
        self.assertLess(limits_by_temp[77], limits_by_temp[298])
        self.assertLess(limits_by_temp[298], limits_by_temp[373])


class TestTheoreticalLimitsRobustness(unittest.TestCase):
    """Test robustness of theoretical limits functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TheoreticalLimitsAnalyzer()

    def test_extreme_parameter_values(self):
        """Test theoretical limits with extreme parameter values."""
        # Test with very small values
        tiny_bits = 1e-6
        tiny_limit = self.analyzer.calculate_landauer_limits(tiny_bits)
        self.assertGreater(tiny_limit.value_j, 0)

        # Test with very large values
        huge_bits = 1e20
        huge_limit = self.analyzer.calculate_landauer_limits(huge_bits)
        self.assertGreater(huge_limit.value_j, 0)

        # Test with zero values (edge case)
        zero_work = self.analyzer.calculate_thermodynamic_limits(0, 1.0)
        self.assertEqual(zero_work.value_j, 0)

    def test_numerical_stability(self):
        """Test numerical stability of calculations."""
        # Test with values that might cause floating point issues
        problematic_values = [
            1e-15,  # Very small
            1e15,   # Very large
            1.0,    # Normal
            0.0,    # Zero
            float('inf'),  # Infinity
            float('nan')   # NaN
        ]

        for value in problematic_values:
            try:
                if value in [float('inf'), float('nan')] or value <= 0:
                    continue  # Skip invalid inputs

                limit = self.analyzer.calculate_landauer_limits(value)
                self.assertIsInstance(limit, TheoreticalLimit)

                if math.isfinite(value) and value > 0:
                    self.assertGreater(limit.value_j, 0)

            except (ValueError, OverflowError):
                # Expected for some edge cases
                pass

    def test_physical_consistency_checks(self):
        """Test physical consistency of theoretical calculations."""
        # Test that calculated limits make physical sense

        # 1. Landauer's limit should always be positive
        landauer = self.analyzer.calculate_landauer_limits(1000)
        self.assertGreater(landauer.value_j, 0)

        # 2. Thermodynamic limit should be >= mechanical work
        mech_work = 1e-6
        thermo = self.analyzer.calculate_thermodynamic_limits(mech_work, 10.0)
        self.assertGreaterEqual(thermo.value_j, mech_work)

        # 3. Information theoretic limit should scale with information rate
        info_rate1 = 1e6
        info_rate2 = 2e6

        limit1 = self.analyzer.calculate_information_theoretic_limits(info_rate1)
        limit2 = self.analyzer.calculate_information_theoretic_limits(info_rate2)

        # Should scale with information rate (approximately)
        ratio = limit2.value_j / limit1.value_j
        expected_ratio = info_rate2 / info_rate1
        self.assertAlmostEqual(ratio, expected_ratio, places=5)

    def test_assumption_validation(self):
        """Test validation of theoretical assumptions."""
        # Test that assumptions are properly documented
        landauer = self.analyzer.calculate_landauer_limits(1000)

        # Should have relevant assumptions
        assumptions_str = ' '.join(landauer.assumptions).lower()
        self.assertIn('temperature', assumptions_str)
        self.assertIn('irreversible', assumptions_str)

        thermodynamic = self.analyzer.calculate_thermodynamic_limits(1e-6, 1.0)
        thermo_assumptions = ' '.join(thermodynamic.assumptions).lower()
        self.assertIn('thermodynamics', thermo_assumptions)

    def test_limit_comparison_framework(self):
        """Test framework for comparing different theoretical limits."""
        # Calculate limits for the same system using different approaches
        bits = 1e8

        limits = self.analyzer.calculate_comprehensive_limits(
            bits_processed=bits,
            mechanical_work=1e-9,
            entropy_change=0.1,
            information_processed=1e9
        )

        # All limits should be reasonable and positive
        for limit_name, limit in limits.items():
            self.assertIsInstance(limit, TheoreticalLimit)
            self.assertGreater(limit.value_j, 0)
            self.assertIsInstance(limit.assumptions, list)
            self.assertGreater(len(limit.assumptions), 0)

        # Landauer's limit should generally be the most restrictive
        landauer_energy = limits['landauer'].value_j
        thermo_energy = limits['thermodynamic'].value_j

        # Thermodynamic limit should be >= Landauer's limit for computation
        self.assertGreaterEqual(thermo_energy, landauer_energy)


if __name__ == '__main__':
    unittest.main()
