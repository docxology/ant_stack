#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.analysis.veridical_reporting module.

Tests veridical reporting functionality with comprehensive data-driven analysis,
empirical evidence generation, case study creation, and report generation.

Following .cursorrules principles:
- Real data analysis (no mocks)
- Statistical validation of scientific methods
- Comprehensive edge case testing
- Professional documentation with references
"""

import unittest
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from unittest.mock import patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis.veridical_reporting import (
    EmpiricalEvidence,
    CaseStudy,
    VeridicalReporter
)
from antstack_core.analysis.statistical_analysis import HAS_NUMPY

try:
    import numpy as np
    HAS_NUMPY_REAL = True
except ImportError:
    HAS_NUMPY_REAL = False


class TestEmpiricalEvidence(unittest.TestCase):
    """Test EmpiricalEvidence dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "experiment_id": "test_exp_001",
            "timestamp": 1640995200.0,  # 2022-01-01 00:00:00
            "data_source": "test_data",
            "raw_measurements": {"energy": [1.0, 1.1, 0.9], "latency": [10.0, 11.0, 9.0]},
            "measurement_uncertainty": {"energy": 0.05, "latency": 0.5},
            "statistical_metrics": {"energy_mean": 1.0, "latency_mean": 10.0},
            "confidence_intervals": {"energy_95ci": (0.9, 1.1), "latency_95ci": (9.0, 11.0)},
            "validation_checks": {"sufficient_data_points": True, "statistical_power": True},
            "quality_metrics": {"data_completeness": 1.0, "measurement_consistency": 0.95},
            "analysis_parameters": {"method": "bootstrap", "samples": 1000},
            "software_versions": {"python": "3.8+", "numpy": "1.20+"},
            "hardware_specifications": {"platform": "test", "precision": "double"}
        }

    def test_empirical_evidence_creation(self):
        """Test basic EmpiricalEvidence creation."""
        evidence = EmpiricalEvidence(**self.sample_data)

        self.assertEqual(evidence.experiment_id, "test_exp_001")
        self.assertEqual(evidence.timestamp, 1640995200.0)
        self.assertEqual(evidence.data_source, "test_data")
        self.assertEqual(len(evidence.raw_measurements), 2)
        self.assertEqual(len(evidence.measurement_uncertainty), 2)

    def test_empirical_evidence_asdict(self):
        """Test conversion to dictionary."""
        evidence = EmpiricalEvidence(**self.sample_data)
        data_dict = asdict(evidence)

        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict["experiment_id"], "test_exp_001")
        self.assertEqual(data_dict["timestamp"], 1640995200.0)

    def test_empirical_evidence_with_empty_data(self):
        """Test EmpiricalEvidence with empty measurements."""
        empty_data = {
            "experiment_id": "empty_test",
            "timestamp": time.time(),
            "data_source": "test",
            "raw_measurements": {},
            "measurement_uncertainty": {},
            "statistical_metrics": {},
            "confidence_intervals": {},
            "validation_checks": {},
            "quality_metrics": {},
            "analysis_parameters": {},
            "software_versions": {},
            "hardware_specifications": {}
        }

        evidence = EmpiricalEvidence(**empty_data)
        self.assertEqual(len(evidence.raw_measurements), 0)
        self.assertEqual(len(evidence.measurement_uncertainty), 0)


class TestCaseStudy(unittest.TestCase):
    """Test CaseStudy dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.case_data = {
            "case_id": "case_001",
            "title": "Test Case Study",
            "description": "A comprehensive test case study",
            "experimental_conditions": {"temperature": 25.0, "humidity": 60.0},
            "parameter_ranges": {"energy": (0.8, 1.2), "latency": (8.0, 12.0)},
            "control_conditions": {"baseline": True},
            "primary_results": {"efficiency": 0.95, "accuracy": 0.98},
            "secondary_results": {"uncertainty": 0.02, "robustness": 0.9},
            "statistical_analysis": {"p_value": 0.001, "effect_size": 0.8},
            "reproducibility_metrics": {"data_availability": 1.0, "methodology_clarity": 0.9},
            "validation_results": {"internal_consistency": True, "external_validation": True},
            "cross_validation": {"bootstrap_score": 0.95, "cross_fold_score": 0.92},
            "methodology": "Comprehensive testing methodology",
            "limitations": ["Limited sample size", "Controlled environment only"],
            "implications": ["Improves efficiency", "Enables scaling"]
        }

    def test_case_study_creation(self):
        """Test basic CaseStudy creation."""
        case = CaseStudy(**self.case_data)

        self.assertEqual(case.case_id, "case_001")
        self.assertEqual(case.title, "Test Case Study")
        self.assertEqual(len(case.experimental_conditions), 2)
        self.assertEqual(len(case.limitations), 2)
        self.assertEqual(len(case.implications), 2)

    def test_case_study_with_minimal_data(self):
        """Test CaseStudy with minimal required data."""
        minimal_data = {
            "case_id": "minimal",
            "title": "Minimal Case",
            "description": "Minimal test case",
            "experimental_conditions": {},
            "parameter_ranges": {},
            "control_conditions": {},
            "primary_results": {},
            "secondary_results": {},
            "statistical_analysis": {},
            "reproducibility_metrics": {},
            "validation_results": {},
            "cross_validation": {},
            "methodology": "Basic methodology",
            "limitations": [],
            "implications": []
        }

        case = CaseStudy(**minimal_data)
        self.assertEqual(case.case_id, "minimal")
        self.assertEqual(case.title, "Minimal Case")


class TestVeridicalReporter(unittest.TestCase):
    """Test VeridicalReporter class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reporter = VeridicalReporter(self.temp_dir)

        # Sample experimental data
        self.experimental_data = {
            "energy_consumption": [1.0, 1.1, 0.9, 1.05, 0.95],
            "latency": [10.0, 11.0, 9.0, 10.5, 9.5],
            "throughput": [100.0, 105.0, 95.0, 102.0, 98.0]
        }

        self.analysis_results = {
            "parameters": {"method": "bootstrap", "samples": 1000},
            "scaling_analysis": {"power_law": True, "exponent": 1.2},
            "efficiency_analysis": {"efficiency": 0.85, "bottleneck": "memory"},
            "statistical_tests": {"p_value": 0.001, "significant": True},
            "uncertainty_quantification": {"confidence": 0.95},
            "sensitivity_analysis": {"robustness": 0.9},
            "cross_validation": {"score": 0.92},
            "confidence_intervals": {"energy_95ci": (0.9, 1.1)},
            "hypothesis_tests": {"null_rejected": True},
            "effect_sizes": {"cohen_d": 0.8},
            "bootstrap_results": {"mean": 1.0, "std": 0.05},
            "cross_fold_results": {"accuracy": 0.95},
            "independent_validation": {"reproduced": True}
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reporter_initialization(self):
        """Test VeridicalReporter initialization."""
        self.assertTrue(self.temp_dir.exists())
        self.assertTrue((self.temp_dir / "data").exists())
        self.assertTrue((self.temp_dir / "analysis").exists())
        self.assertTrue((self.temp_dir / "figures").exists())
        self.assertTrue((self.temp_dir / "reports").exists())

    def test_generate_empirical_evidence(self):
        """Test empirical evidence generation."""
        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data,
            self.analysis_results
        )

        # Check basic properties
        self.assertIsInstance(evidence, EmpiricalEvidence)
        self.assertTrue(evidence.experiment_id.startswith("exp_"))
        self.assertEqual(evidence.data_source, "computational_analysis")

        # Check measurements
        self.assertIn("energy_consumption", evidence.raw_measurements)
        self.assertIn("latency", evidence.raw_measurements)
        self.assertIn("throughput", evidence.raw_measurements)

        # Check uncertainty calculations
        self.assertIn("energy_consumption", evidence.measurement_uncertainty)
        self.assertIn("latency", evidence.measurement_uncertainty)
        self.assertIn("throughput", evidence.measurement_uncertainty)

        # Check statistical metrics
        self.assertIn("energy_consumption_mean", evidence.statistical_metrics)
        self.assertIn("latency_mean", evidence.statistical_metrics)

        # Check validation checks
        self.assertIn("sufficient_data_points", evidence.validation_checks)
        self.assertIn("statistical_power", evidence.validation_checks)
        self.assertIn("measurement_precision", evidence.validation_checks)

        # Check quality metrics
        self.assertIn("data_completeness", evidence.quality_metrics)
        self.assertIn("measurement_consistency", evidence.quality_metrics)
        self.assertIn("statistical_robustness", evidence.quality_metrics)

    def test_generate_empirical_evidence_empty_data(self):
        """Test empirical evidence generation with empty data."""
        empty_data = {}
        evidence = self.reporter.generate_empirical_evidence(empty_data, {})

        self.assertIsInstance(evidence, EmpiricalEvidence)
        self.assertEqual(len(evidence.raw_measurements), 0)
        self.assertEqual(len(evidence.measurement_uncertainty), 0)

    def test_create_case_study(self):
        """Test case study creation."""
        case = self.reporter.create_case_study(
            case_id="test_case_001",
            title="Test Case Study",
            experimental_data=self.experimental_data,
            analysis_results=self.analysis_results,
            methodology="Comprehensive test methodology"
        )

        self.assertIsInstance(case, CaseStudy)
        self.assertEqual(case.case_id, "test_case_001")
        self.assertEqual(case.title, "Test Case Study")
        self.assertEqual(case.methodology, "Comprehensive test methodology")

        # Check experimental conditions
        self.assertIn("data_points", case.experimental_conditions)
        self.assertIn("parameter_ranges", case.experimental_conditions)

        # Check parameter ranges
        self.assertIn("energy_consumption", case.parameter_ranges)
        self.assertIn("latency", case.parameter_ranges)
        self.assertIn("throughput", case.parameter_ranges)

        # Check results
        self.assertIn("scaling_relationships", case.primary_results)
        self.assertIn("efficiency_metrics", case.primary_results)
        self.assertIn("statistical_significance", case.primary_results)

        # Check reproducibility metrics
        self.assertIn("data_availability", case.reproducibility_metrics)
        self.assertIn("methodology_clarity", case.reproducibility_metrics)

        # Check limitations and implications
        self.assertIsInstance(case.limitations, list)
        self.assertIsInstance(case.implications, list)
        self.assertGreater(len(case.limitations), 0)
        self.assertGreater(len(case.implications), 0)

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data, self.analysis_results
        )

        case_studies = [
            self.reporter.create_case_study(
                "case_001", "Primary Analysis",
                self.experimental_data, self.analysis_results,
                "Primary analysis methodology"
            )
        ]

        report_path = self.reporter.generate_comprehensive_report(
            evidence, case_studies, self.analysis_results
        )

        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.is_file())

        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()

        self.assertIn("# Comprehensive Complexity Energetics Analysis Report", content)
        self.assertIn(evidence.experiment_id, content)
        self.assertIn("Executive Summary", content)
        self.assertIn("Key Empirical Findings", content)
        self.assertIn("Data Quality Assessment", content)
        self.assertIn("Detailed Analysis Results", content)
        self.assertIn("Case Study 1: Primary Analysis", content)
        self.assertIn("Methodology", content)
        self.assertIn("References", content)

    def test_comprehensive_workflow(self):
        """Test the complete workflow from data to report generation."""
        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data, self.analysis_results
        )

        case_studies = [
            self.reporter.create_case_study(
                "case_001", "Test Case",
                self.experimental_data, self.analysis_results,
                "Test methodology"
            )
        ]

        # Generate comprehensive report (this should create all supporting files)
        report_path = self.reporter.generate_comprehensive_report(
            evidence, case_studies, self.analysis_results
        )

        # Check that the main report was created
        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.is_file())

        # Check that supporting data files were created
        data_path = self.temp_dir / "data" / "raw_measurements.json"
        stats_path = self.temp_dir / "analysis" / "statistical_analysis.json"
        cases_path = self.temp_dir / "analysis" / "case_studies.json"
        results_path = self.temp_dir / "analysis" / "complete_analysis.json"

        self.assertTrue(data_path.exists())
        self.assertTrue(stats_path.exists())
        self.assertTrue(cases_path.exists())
        self.assertTrue(results_path.exists())

        # Check JSON content
        with open(data_path, 'r') as f:
            data_content = json.load(f)
        self.assertIn("experiment_id", data_content)
        self.assertIn("raw_measurements", data_content)

        with open(stats_path, 'r') as f:
            stats_content = json.load(f)
        self.assertIn("statistical_metrics", stats_content)
        self.assertIn("validation_checks", stats_content)

    def test_generate_data_driven_summary(self):
        """Test data-driven summary generation."""
        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data, self.analysis_results
        )

        case_studies = [
            self.reporter.create_case_study(
                "case_001", "Summary Test",
                self.experimental_data, self.analysis_results,
                "Summary test methodology"
            )
        ]

        summary = self.reporter.generate_data_driven_summary(evidence, case_studies)

        self.assertIsInstance(summary, dict)

        # Check required sections
        self.assertIn("experiment_metadata", summary)
        self.assertIn("statistical_summary", summary)
        self.assertIn("quality_assessment", summary)
        self.assertIn("key_findings", summary)
        self.assertIn("reproducibility", summary)

        # Check experiment metadata
        exp_meta = summary["experiment_metadata"]
        self.assertEqual(exp_meta["experiment_id"], evidence.experiment_id)
        self.assertEqual(exp_meta["number_of_case_studies"], 1)
        self.assertEqual(exp_meta["number_of_parameters"], 3)

        # Check statistical summary
        stat_summary = summary["statistical_summary"]
        self.assertEqual(stat_summary["confidence_level"], "95%")
        self.assertEqual(stat_summary["bootstrap_samples"], 1000)

        # Check quality assessment
        quality = summary["quality_assessment"]
        self.assertIsInstance(quality["overall_quality_score"], float)
        self.assertGreaterEqual(quality["overall_quality_score"], 0.0)
        self.assertLessEqual(quality["overall_quality_score"], 1.0)

        # Check key findings
        key_findings = summary["key_findings"]
        self.assertIn("energy_consumption", key_findings)
        self.assertIn("latency", key_findings)
        self.assertIn("throughput", key_findings)

        for param, metrics in key_findings.items():
            self.assertIn("mean", metrics)
            self.assertIn("uncertainty", metrics)
            self.assertIn("confidence_interval", metrics)

    def test_generate_data_driven_summary_empty(self):
        """Test data-driven summary with empty data."""
        empty_experimental_data = {}
        evidence = self.reporter.generate_empirical_evidence(empty_experimental_data, {})

        summary = self.reporter.generate_data_driven_summary(evidence, [])

        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["experiment_metadata"]["number_of_parameters"], 0)
        self.assertEqual(summary["experiment_metadata"]["number_of_case_studies"], 0)
        self.assertEqual(len(summary["key_findings"]), 0)

    @patch('antstack_core.analysis.statistical_analysis.bootstrap_mean_ci')
    def test_generate_empirical_evidence_bootstrap_failure(self, mock_bootstrap):
        """Test empirical evidence generation when bootstrap fails."""
        mock_bootstrap.side_effect = Exception("Bootstrap failed")

        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data, self.analysis_results
        )

        # Should still generate evidence but with fallback values
        self.assertIsInstance(evidence, EmpiricalEvidence)
        self.assertIn("energy_consumption_mean", evidence.statistical_metrics)
        # The statistical analyzer now handles bootstrap failures gracefully
        self.assertIn("energy_consumption_95ci", evidence.confidence_intervals)

    def test_generate_comprehensive_report_with_multiple_cases(self):
        """Test comprehensive report with multiple case studies."""
        evidence = self.reporter.generate_empirical_evidence(
            self.experimental_data, self.analysis_results
        )

        case_studies = [
            self.reporter.create_case_study(
                f"case_{i+1:03d}", f"Case Study {i+1}",
                self.experimental_data, self.analysis_results,
                f"Methodology for case {i+1}"
            ) for i in range(3)
        ]

        report_path = self.reporter.generate_comprehensive_report(
            evidence, case_studies, self.analysis_results
        )

        with open(report_path, 'r') as f:
            content = f.read()

        self.assertIn("Case Study 1: Case Study 1", content)
        self.assertIn("Case Study 2: Case Study 2", content)
        self.assertIn("Case Study 3: Case Study 3", content)

    def test_measurement_uncertainty_calculation(self):
        """Test measurement uncertainty calculation."""
        # Test with numpy available
        if HAS_NUMPY:
            single_value_data = {"single_param": [5.0]}
            evidence = self.reporter.generate_empirical_evidence(single_value_data, {})

            # For single values, should use fallback calculation
            self.assertIn("single_param", evidence.measurement_uncertainty)

        # Test with multiple values
        multi_value_data = {"multi_param": [1.0, 2.0, 3.0, 4.0, 5.0]}
        evidence = self.reporter.generate_empirical_evidence(multi_value_data, {})

        self.assertIn("multi_param", evidence.measurement_uncertainty)
        uncertainty = evidence.measurement_uncertainty["multi_param"]

        # Uncertainty should be positive
        self.assertGreater(uncertainty, 0.0)

        # Check that statistical metrics include mean and std
        self.assertIn("multi_param_mean", evidence.statistical_metrics)
        self.assertIn("multi_param_std", evidence.statistical_metrics)

    def test_validation_checks_comprehensive(self):
        """Test comprehensive validation checks."""
        # Test with sufficient data
        sufficient_data = {
            "param1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10 points
            "param2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # 11 points
        }
        evidence = self.reporter.generate_empirical_evidence(sufficient_data, {})

        self.assertTrue(evidence.validation_checks["sufficient_data_points"])
        self.assertTrue(evidence.validation_checks["statistical_power"])

        # Test with insufficient data
        insufficient_data = {"param1": [1.0, 2.0]}  # Only 2 points
        evidence = self.reporter.generate_empirical_evidence(insufficient_data, {})

        self.assertFalse(evidence.validation_checks["sufficient_data_points"])
        self.assertFalse(evidence.validation_checks["statistical_power"])

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        test_data = {
            "param1": [1.0, 1.01, 0.99, 1.02, 0.98],  # Low uncertainty
            "param2": [10.0, 10.1, 9.9, 10.2, 9.8]     # Low uncertainty
        }
        evidence = self.reporter.generate_empirical_evidence(test_data, {})

        # Check that quality metrics are calculated
        self.assertIn("data_completeness", evidence.quality_metrics)
        self.assertIn("measurement_consistency", evidence.quality_metrics)
        self.assertIn("statistical_robustness", evidence.quality_metrics)

        # Data completeness should be 1.0 for complete data
        self.assertEqual(evidence.quality_metrics["data_completeness"], 1.0)

        # Quality metrics should be between 0 and 1
        for metric_name, value in evidence.quality_metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == '__main__':
    unittest.main()
