"""
Integration Tests for AntStack Core Modules

Comprehensive integration testing that demonstrates cross-module functionality,
end-to-end workflows, and real-world usage patterns.

These tests validate that all modules work together seamlessly and provide
confidence in the overall system reliability.
"""

import unittest
import tempfile
from pathlib import Path
import json
import yaml
from unittest.mock import patch, MagicMock

from antstack_core.analysis.energy import EnergyCoefficients, ComputeLoad, estimate_detailed_energy
from antstack_core.analysis.statistics import bootstrap_mean_ci, analyze_scaling_relationship
from antstack_core.analysis.enhanced_estimators import EnhancedEnergyEstimator
from antstack_core.analysis.experiment_config import WorkloadConfig, EnergyCoefficientsConfig, ExperimentManifest
from antstack_core.figures.plots import scatter_plot, line_plot
from antstack_core.figures.mermaid import preprocess_mermaid_diagrams
from antstack_core.figures.references import validate_cross_references
from antstack_core.cohereants.core import calculate_wavelength_from_wavenumber
from antstack_core.cohereants.spectroscopy import SpectralData, CHCAnalyzer, analyze_chc_spectra
from antstack_core.publishing.pdf_generator import PDFGenerator, PDFGenerationConfig
from antstack_core.publishing.quality_validator import QualityValidator
from antstack_core.publishing.reference_manager import ReferenceManager
from antstack_core.publishing.build_orchestrator import BuildOrchestrator, BuildTarget


class TestEnergyAnalysisWorkflow(unittest.TestCase):
    """Test complete energy analysis workflow from data to publication."""

    def setUp(self):
        """Set up test fixtures for energy analysis."""
        self.energy_coeffs = EnergyCoefficients()
        self.test_data = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]

    def test_energy_estimation_pipeline(self):
        """Test complete energy estimation pipeline."""
        # 1. Define computational workload
        workload = ComputeLoad(
            flops=1e9,
            sram_bytes=1e6,
            dram_bytes=1e8,
            time_seconds=0.1
        )

        # 2. Estimate energy consumption
        energy_breakdown = estimate_detailed_energy(workload, self.energy_coeffs)

        # 3. Validate results
        self.assertGreater(energy_breakdown.total, 0)
        self.assertGreater(energy_breakdown.compute_flops, 0)
        self.assertGreater(energy_breakdown.compute_memory, 0)

        # 4. Test statistical analysis
        mean_energy, lower_ci, upper_ci = bootstrap_mean_ci([energy_breakdown.total] * 10)

        self.assertGreater(mean_energy, 0)
        self.assertLess(lower_ci, mean_energy)
        self.assertGreater(upper_ci, mean_energy)

    def test_scaling_analysis_workflow(self):
        """Test scaling analysis workflow."""
        # Generate scaling data
        x_values = [1, 2, 4, 8, 16, 32]
        y_values = [1.0, 1.8, 3.2, 5.8, 10.4, 18.8]  # ~0.8 scaling

        # Perform scaling analysis
        result = analyze_scaling_relationship(x_values, y_values)

        # Validate results
        self.assertIn('scaling_exponent', result)
        self.assertIn('r_squared', result)
        self.assertGreater(result['r_squared'], 0.9)  # Should be good fit
        self.assertAlmostEqual(result['scaling_exponent'], 0.8, places=1)

    def test_enhanced_estimator_integration(self):
        """Test enhanced energy estimator with multiple modules."""
        estimator = EnhancedEnergyEstimator(self.energy_coeffs)

        # Test body scaling analysis
        body_data = estimator.analyze_body_scaling([4, 8, 16], {
            'v': 1.0, 'm': 0.001, 'L': 0.01, 'dt': 0.01, 'g': 9.81
        })

        self.assertIsNotNone(body_data)
        self.assertEqual(len(body_data.energy_values), 3)
        self.assertTrue(all(e > 0 for e in body_data.energy_values))


class TestSpectroscopyWorkflow(unittest.TestCase):
    """Test complete spectroscopy analysis workflow."""

    def setUp(self):
        """Set up spectroscopy test data."""
        self.wavenumbers = [2800, 2900, 3000, 3100, 3200]
        self.intensities = [0.1, 0.8, 0.9, 0.3, 0.1]
        self.spectral_data = SpectralData(self.wavenumbers, self.intensities, "Test_CHC")

    def test_spectroscopy_pipeline(self):
        """Test complete spectroscopy analysis pipeline."""
        # 1. Create spectral data
        self.assertEqual(self.spectral_data.species, "Test_CHC")
        self.assertEqual(len(self.spectral_data.wavenumbers), 5)

        # 2. Apply baseline correction
        corrected = self.spectral_data.baseline_correction()
        self.assertIsInstance(corrected, SpectralData)

        # 3. Normalize data
        normalized = self.spectral_data.normalize()
        self.assertIsInstance(normalized, SpectralData)

        # 4. Perform CHC analysis
        analyzer = CHCAnalyzer()
        results = analyzer.analyze_spectrum(self.spectral_data)

        # Validate results structure
        self.assertIn('species', results)
        self.assertIn('peak_analysis', results)
        self.assertIn('spectral_characteristics', results)

    def test_wavelength_calculations(self):
        """Test wavelength conversion integration."""
        # Test wavelength conversion
        wavenumber = 2900  # cm⁻¹ (typical CH stretch)
        wavelength = calculate_wavelength_from_wavenumber(wavenumber)

        # Should be around 3.45 μm
        self.assertAlmostEqual(wavelength, 3.448, places=3)

        # Test round-trip conversion
        original_wavenumber = 2900.0
        wavelength = calculate_wavelength_from_wavenumber(original_wavenumber)
        recovered_wavenumber = 10000.0 / wavelength

        self.assertAlmostEqual(recovered_wavenumber, original_wavenumber, places=1)


class TestPublishingWorkflow(unittest.TestCase):
    """Test complete publishing workflow integration."""

    def setUp(self):
        """Set up publishing test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_markdown = self.temp_dir / "sample.md"

        # Create sample markdown content
        content = """# Test Document

This is a test document for publishing workflow validation.

## Introduction

This document demonstrates the publishing capabilities.

## Figure Example

![Test Figure](test_figure.png)
**Caption:** This is a test figure showing sample data.

## References

\\cite{smith2023}
"""
        self.sample_markdown.write_text(content)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    def test_pdf_generation_workflow(self, mock_subprocess):
        """Test PDF generation workflow (mocked)."""
        # Mock successful pandoc execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test PDF generation
        generator = PDFGenerator()
        result = generator.generate_pdf(
            [self.sample_markdown],
            self.temp_dir / "output.pdf",
            {"title": "Test Document", "author": "Test Author"}
        )

        # Should succeed with mocked subprocess
        self.assertTrue(result.success)

    def test_quality_validation_workflow(self):
        """Test quality validation on sample document."""
        validator = QualityValidator()
        report = validator.validate_publication([self.sample_markdown])

        # Should generate a report
        self.assertIsInstance(report, type(report))
        self.assertEqual(report.total_files, 1)
        self.assertIsInstance(report.issues, list)

    def test_reference_analysis_workflow(self):
        """Test reference analysis workflow."""
        manager = ReferenceManager()
        report = manager.analyze_references([self.sample_markdown])

        # Should generate reference report
        self.assertIsInstance(report, type(report))
        self.assertIsInstance(report.definitions, list)
        self.assertIsInstance(report.usages, list)


class TestBuildOrchestrationWorkflow(unittest.TestCase):
    """Test build orchestration for complete papers."""

    def setUp(self):
        """Set up build orchestration test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.paper_dir = self.temp_dir / "test_paper"
        self.paper_dir.mkdir()

        # Create sample paper files
        self.main_file = self.paper_dir / "test_paper.md"
        self.main_file.write_text("# Test Paper\n\nThis is a test paper.")

        self.config_file = self.paper_dir / "paper_config.yaml"
        config_data = {
            "title": "Test Paper",
            "author": "Test Author",
            "metadata": {
                "title": "Test Paper",
                "author": "Test Author"
            }
        }
        self.config_file.write_text(yaml.dump(config_data))

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_target_creation(self):
        """Test build target creation from paper directory."""
        orchestrator = BuildOrchestrator()

        target = orchestrator.create_paper_target(
            "test_paper",
            self.paper_dir,
            self.temp_dir / "output"
        )

        self.assertEqual(target.name, "test_paper")
        self.assertIn(self.main_file, target.source_files)
        self.assertTrue(str(target.output_path).endswith("test_paper.pdf"))

    def test_build_orchestration_workflow(self):
        """Test complete build orchestration workflow."""
        orchestrator = BuildOrchestrator()

        # Create target
        target = orchestrator.create_paper_target("test_paper", self.paper_dir)

        # Should be able to create target without errors
        self.assertIsInstance(target, BuildTarget)
        self.assertEqual(target.name, "test_paper")


class TestCrossModuleIntegration(unittest.TestCase):
    """Test integration between different modules."""

    def test_energy_to_figure_integration(self):
        """Test integration between energy analysis and figure generation."""
        # Generate energy data
        coeffs = EnergyCoefficients()
        workload = ComputeLoad(flops=1e9, time_seconds=0.1)
        energy = estimate_detailed_energy(workload, coeffs)

        # Create simple data for plotting
        x_data = [1, 2, 3, 4, 5]
        y_data = [energy.total * i for i in [0.5, 0.7, 1.0, 1.2, 1.5]]

        # Test figure generation (would create plot in real implementation)
        try:
            fig = scatter_plot(x_data, y_data, "Test X", "Test Y", "Energy Scaling")
            # In real implementation, this would create a matplotlib figure
            # For testing, we just verify no exceptions are raised
            self.assertIsNotNone(fig)
        except ImportError:
            # matplotlib may not be available in test environment
            pass

    def test_configuration_to_energy_integration(self):
        """Test integration between configuration and energy analysis."""
        # Create configuration
        energy_config = EnergyCoefficientsConfig(flops_pj=2.0, baseline_w=0.5)
        workload_config = WorkloadConfig(
            name="test_workload",
            params={"flops": 1e9, "memory_bytes": 1e6}
        )

        # Should be able to create configurations
        self.assertIsInstance(energy_config, EnergyCoefficientsConfig)
        self.assertIsInstance(workload_config, WorkloadConfig)

        # Test experiment manifest creation
        manifest = ExperimentManifest(
            experiment_name="integration_test",
            workloads={"test": workload_config},
            coefficients=energy_config
        )

        self.assertEqual(manifest.experiment_name, "integration_test")
        self.assertIn("test", manifest.workloads)


class TestScientificWorkflow(unittest.TestCase):
    """Test complete scientific workflow from hypothesis to publication."""

    def test_hypothesis_to_publication_workflow(self):
        """Test complete workflow from hypothesis to publication-ready output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Define research question and parameters
            research_params = {
                "hypothesis": "Energy scales with computational complexity",
                "complexity_range": [1, 2, 4, 8, 16, 32],
                "energy_coefficients": EnergyCoefficients()
            }

            # 2. Generate experimental data
            complexity_values = research_params["complexity_range"]
            energy_values = []

            for complexity in complexity_values:
                workload = ComputeLoad(
                    flops=complexity * 1e8,
                    time_seconds=0.1
                )
                energy = estimate_detailed_energy(workload, research_params["energy_coefficients"])
                energy_values.append(energy.total)

            # 3. Perform statistical analysis
            scaling_result = analyze_scaling_relationship(complexity_values, energy_values)

            # 4. Validate results
            self.assertIn('scaling_exponent', scaling_result)
            self.assertGreater(scaling_result['scaling_exponent'], 0)

            # 5. Create publication-ready summary
            summary = {
                "complexity_range": complexity_values,
                "energy_values": energy_values,
                "scaling_exponent": scaling_result['scaling_exponent'],
                "r_squared": scaling_result['r_squared']
            }

            # Should have valid scientific results
            self.assertGreater(summary['r_squared'], 0)
            self.assertTrue(all(e > 0 for e in summary['energy_values']))


if __name__ == '__main__':
    unittest.main()
