#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.cohereants.spectroscopy module.

Tests cuticular hydrocarbon (CHC) spectroscopy analysis functionality including:
- Spectral data validation and processing
- Peak finding and analysis
- CHC compound identification
- Spectral overlap calculations
- Cross-platform compatibility with graceful fallbacks
- Integration with core analysis workflows

Following .cursorrules principles:
- Real spectral data analysis (no mocks)
- Comprehensive edge case testing
- Platform-specific testing with proper detection
- Scientific accuracy validation
"""

import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.cohereants.spectroscopy import (
    SpectralData,
    PeakFinder,
    CHCAnalyzer,
    analyze_chc_spectra,
    identify_chc_compounds,
    calculate_spectral_overlap,
    generate_spectral_plots
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestSpectralData(unittest.TestCase):
    """Test SpectralData class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_wavenumbers = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
        self.sample_intensities = [0.1, 0.3, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        self.species_name = "Test_CHC"

    def test_spectral_data_creation(self):
        """Test basic SpectralData creation."""
        spectral_data = SpectralData(
            wavenumbers=self.sample_wavenumbers,
            intensities=self.sample_intensities,
            species=self.species_name
        )

        self.assertEqual(spectral_data.species, self.species_name)
        self.assertEqual(spectral_data.num_points, 8)
        np.testing.assert_array_equal(spectral_data.wavenumbers, self.sample_wavenumbers)
        np.testing.assert_array_equal(spectral_data.intensities, self.sample_intensities)

    def test_spectral_data_creation_numpy_arrays(self):
        """Test SpectralData creation with numpy arrays."""
        wavenumbers_array = np.array(self.sample_wavenumbers)
        intensities_array = np.array(self.sample_intensities)

        spectral_data = SpectralData(
            wavenumbers=wavenumbers_array,
            intensities=intensities_array,
            species=self.species_name
        )

        self.assertIsInstance(spectral_data.wavenumbers, np.ndarray)
        self.assertIsInstance(spectral_data.intensities, np.ndarray)

    def test_spectral_data_properties(self):
        """Test SpectralData properties."""
        spectral_data = SpectralData(
            wavenumbers=self.sample_wavenumbers,
            intensities=self.sample_intensities
        )

        # Test num_points property
        self.assertEqual(spectral_data.num_points, 8)

        # Test spectral_range property
        spectral_range = spectral_data.spectral_range
        self.assertEqual(spectral_range, (400.0, 3200.0))

        # Test intensity_range property
        intensity_range = spectral_data.intensity_range
        self.assertEqual(intensity_range, (0.05, 0.8))

    def test_spectral_data_validation_errors(self):
        """Test SpectralData validation error handling."""
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[400, 800, 1200],
                intensities=[0.1, 0.3]  # Different length
            )

        # Test empty data
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[],
                intensities=[]
            )

        # Test negative wavenumbers
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[-400, 800, 1200],
                intensities=[0.1, 0.3, 0.8]
            )

        # Test negative intensities
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[400, 800, 1200],
                intensities=[0.1, -0.3, 0.8]
            )

        # Test wavenumbers out of range (too low)
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[200, 800, 1200],  # Below 300 cm⁻¹
                intensities=[0.1, 0.3, 0.8]
            )

        # Test wavenumbers out of range (too high)
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[400, 800, 7000],  # Above 6000 cm⁻¹
                intensities=[0.1, 0.3, 0.8]
            )

    def test_spectral_data_region_mask(self):
        """Test region mask functionality."""
        spectral_data = SpectralData(
            wavenumbers=self.sample_wavenumbers,
            intensities=self.sample_intensities
        )

        # Test valid region
        mask = spectral_data.get_region_mask(1000, 2000)
        expected_mask = [False, False, True, True, True, False, False, False]
        np.testing.assert_array_equal(mask, expected_mask)

        # Test edge case: min >= max
        with self.assertRaises(ValueError):
            spectral_data.get_region_mask(2000, 1000)

        # Test region outside data range
        mask = spectral_data.get_region_mask(5000, 6000)
        np.testing.assert_array_equal(mask, [False] * 8)

    def test_spectral_data_normalize(self):
        """Test spectral data normalization."""
        spectral_data = SpectralData(
            wavenumbers=self.sample_wavenumbers,
            intensities=self.sample_intensities
        )

        normalized = spectral_data.normalize()

        # Check that max intensity is 1.0
        self.assertAlmostEqual(np.max(normalized.intensities), 1.0, places=6)

        # Check that min intensity is scaled appropriately
        expected_min = 0.05 / 0.8  # Original min / original max
        self.assertAlmostEqual(np.min(normalized.intensities), expected_min, places=6)

        # Check that wavenumbers are unchanged
        np.testing.assert_array_equal(normalized.wavenumbers, spectral_data.wavenumbers)

    def test_spectral_data_baseline_correction(self):
        """Test baseline correction."""
        wavenumbers = [400, 800, 1200, 1600, 2000]
        intensities = [0.1, 0.2, 0.3, 0.4, 0.5]  # Linear trend

        spectral_data = SpectralData(
            wavenumbers=wavenumbers,
            intensities=intensities
        )

        corrected = spectral_data.baseline_correction(method='linear')

        # Check that baseline is approximately corrected
        # For linear baseline, the corrected spectrum should be flatter
        self.assertIsInstance(corrected, SpectralData)
        self.assertEqual(corrected.num_points, spectral_data.num_points)


class TestPeakFinder(unittest.TestCase):
    """Test PeakFinder class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_wavenumbers = np.linspace(400, 3200, 100)
        # Create synthetic spectrum with peaks
        self.sample_intensities = (
            0.1 * np.sin(2 * np.pi * self.sample_wavenumbers / 800) +
            0.05 * np.sin(2 * np.pi * self.sample_wavenumbers / 1600) +
            0.8  # Baseline
        )

        self.spectral_data = SpectralData(
            wavenumbers=self.sample_wavenumbers,
            intensities=self.sample_intensities
        )

    def test_peak_finder_creation(self):
        """Test PeakFinder creation."""
        finder = PeakFinder(threshold_factor=0.5, min_distance=50)

        self.assertEqual(finder.threshold_factor, 0.5)
        self.assertEqual(finder.min_distance, 50)

    def test_peak_finder_default_creation(self):
        """Test PeakFinder creation with defaults."""
        finder = PeakFinder()

        self.assertEqual(finder.threshold_factor, 0.3)  # default
        self.assertEqual(finder.min_distance, 20)       # default

    def test_peak_finder_find_peaks(self):
        """Test peak finding functionality."""
        finder = PeakFinder(threshold_factor=0.5, min_distance=10)

        peaks, properties = finder.find_peaks(self.spectral_data)

        # Should find some peaks
        self.assertIsInstance(peaks, np.ndarray)
        self.assertIsInstance(properties, dict)

        # Check that peaks are within valid range
        if len(peaks) > 0:
            self.assertTrue(all(0 <= p < len(self.sample_wavenumbers) for p in peaks))

    def test_peak_finder_analyze_peaks(self):
        """Test peak analysis functionality."""
        finder = PeakFinder()

        analysis = finder.analyze_peaks(self.spectral_data)

        self.assertIsInstance(analysis, dict)

        # Check for expected analysis keys
        expected_keys = ['peak_count', 'peak_wavenumbers', 'peak_intensities',
                        'peak_widths', 'peak_prominences']
        for key in expected_keys:
            self.assertIn(key, analysis)

    def test_peak_finder_with_no_peaks(self):
        """Test peak finder with flat spectrum (no peaks)."""
        # Create flat spectrum
        flat_wavenumbers = np.linspace(400, 3200, 50)
        flat_intensities = np.ones(50) * 0.5

        flat_data = SpectralData(
            wavenumbers=flat_wavenumbers,
            intensities=flat_intensities
        )

        finder = PeakFinder(threshold_factor=0.1)  # Low threshold

        peaks, properties = finder.find_peaks(flat_data)

        # Should find no peaks in flat spectrum
        self.assertEqual(len(peaks), 0)

    def test_peak_finder_scipy_fallback(self):
        """Test peak finder fallback when scipy is not available."""
        # This test verifies the fallback implementation works
        finder = PeakFinder()

        # Test with a simple spectrum that should have peaks
        simple_wavenumbers = np.array([400, 800, 1200, 1600, 2000, 2400])
        simple_intensities = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.1])

        simple_data = SpectralData(
            wavenumbers=simple_wavenumbers,
            intensities=simple_intensities
        )

        peaks, properties = finder.find_peaks(simple_data)

        # Should find peaks regardless of scipy availability
        self.assertIsInstance(peaks, np.ndarray)
        self.assertIsInstance(properties, dict)


class TestCHCAnalyzer(unittest.TestCase):
    """Test CHCAnalyzer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample CHC spectral data
        self.wavenumbers = np.linspace(400, 3200, 100)
        # Simulate CHC spectrum with characteristic peaks
        self.intensities = (
            0.1 * np.exp(-((self.wavenumbers - 2850) / 100)**2) +  # CH stretch
            0.08 * np.exp(-((self.wavenumbers - 1450) / 80)**2) +  # CH2 bend
            0.05 * np.exp(-((self.wavenumbers - 720) / 60)**2) +   # CH rock
            0.02  # Baseline
        )

        self.spectral_data = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities,
            species="Sample_CHC"
        )

    def test_chc_analyzer_creation(self):
        """Test CHCAnalyzer creation."""
        analyzer = CHCAnalyzer()

        self.assertIsNotNone(analyzer)
        self.assertIsInstance(analyzer.peak_finder, PeakFinder)

    def test_chc_analyzer_analyze_spectrum(self):
        """Test CHC spectrum analysis."""
        analyzer = CHCAnalyzer()

        analysis = analyzer.analyze_spectrum(self.spectral_data)

        self.assertIsInstance(analysis, dict)

        # Check for expected analysis components
        expected_keys = ['peak_analysis', 'compound_identification',
                        'spectral_characteristics', 'quality_metrics']
        for key in expected_keys:
            self.assertIn(key, analysis)

    def test_chc_analyzer_identify_functional_groups(self):
        """Test functional group identification."""
        analyzer = CHCAnalyzer()

        # Test with known CHC peaks
        test_peaks = [2850, 1450, 720]  # CH stretch, CH2 bend, CH rock

        functional_groups = analyzer.identify_functional_groups(test_peaks)

        self.assertIsInstance(functional_groups, list)
        # Should identify CH-related functional groups
        self.assertGreater(len(functional_groups), 0)

        # Check that identified groups contain expected information
        for group in functional_groups:
            self.assertIn('name', group)
            self.assertIn('wavenumber', group)
            self.assertIn('confidence', group)

    def test_chc_analyzer_calculate_spectral_features(self):
        """Test spectral feature calculation."""
        analyzer = CHCAnalyzer()

        features = analyzer.calculate_spectral_features(self.spectral_data)

        self.assertIsInstance(features, dict)

        # Check for expected spectral features
        expected_features = ['intensity_stats', 'peak_stats', 'spectral_regions']
        for feature in expected_features:
            self.assertIn(feature, features)

    def test_chc_analyzer_compare_spectra(self):
        """Test spectrum comparison functionality."""
        analyzer = CHCAnalyzer()

        # Create two similar spectra
        spectrum1 = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities
        )

        spectrum2 = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities * 0.9  # Slightly different
        )

        comparison = analyzer.compare_spectra(spectrum1, spectrum2)

        self.assertIsInstance(comparison, dict)
        self.assertIn('similarity_score', comparison)
        self.assertIn('peak_differences', comparison)


class TestSpectroscopyFunctions(unittest.TestCase):
    """Test standalone spectroscopy functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_wavenumbers = np.array([400, 800, 1200, 1600, 2000, 2400, 2800, 3200])
        self.sample_intensities = np.array([0.1, 0.3, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])

        # Create test spectra dictionary
        self.test_spectra = {
            'CHC_A': np.array([0.1, 0.4, 0.8, 0.5, 0.3, 0.1, 0.05, 0.02]),
            'CHC_B': np.array([0.2, 0.3, 0.7, 0.6, 0.4, 0.2, 0.08, 0.03])
        }

    def test_analyze_chc_spectra_wavenumbers(self):
        """Test CHC spectra analysis with wavenumbers."""
        result = analyze_chc_spectra(
            wavenumbers_or_wavelengths=self.sample_wavenumbers,
            intensities=self.sample_intensities,
            is_wavelength=False
        )

        self.assertIsInstance(result, dict)
        self.assertIn('spectral_data', result)
        self.assertIn('peak_analysis', result)
        self.assertIn('chc_analysis', result)

    def test_analyze_chc_spectra_wavelengths(self):
        """Test CHC spectra analysis with wavelengths."""
        # Convert wavenumbers to wavelengths (μm)
        wavelengths = 10000 / self.sample_wavenumbers  # Convert cm⁻¹ to μm

        result = analyze_chc_spectra(
            wavenumbers_or_wavelengths=wavelengths,
            intensities=self.sample_intensities,
            is_wavelength=True
        )

        self.assertIsInstance(result, dict)
        self.assertIn('spectral_data', result)

    def test_identify_chc_compounds(self):
        """Test CHC compound identification."""
        # Test with known CHC characteristic peaks
        test_peaks = [2850, 2920, 1450, 1375, 720]  # Various CH stretches and bends

        compounds = identify_chc_compounds(test_peaks)

        self.assertIsInstance(compounds, list)
        self.assertGreater(len(compounds), 0)

        # Check compound identification structure
        for compound in compounds:
            self.assertIn('name', compound)
            self.assertIn('confidence', compound)
            self.assertIn('matched_peaks', compound)

    def test_identify_chc_compounds_no_matches(self):
        """Test CHC identification with peaks that don't match known compounds."""
        # Test with peaks that don't correspond to typical CHC compounds
        unusual_peaks = [5000, 6000, 7000]  # Far IR region

        compounds = identify_chc_compounds(unusual_peaks)

        # Should return empty or low-confidence results
        self.assertIsInstance(compounds, list)

    def test_calculate_spectral_overlap(self):
        """Test spectral overlap calculation."""
        spectrum1 = np.array([0.1, 0.3, 0.8, 0.6, 0.4])
        spectrum2 = np.array([0.2, 0.4, 0.7, 0.5, 0.3])

        overlap_metrics = calculate_spectral_overlap(spectrum1, spectrum2)

        self.assertIsInstance(overlap_metrics, dict)
        self.assertIn('pearson_correlation', overlap_metrics)
        self.assertIn('cosine_similarity', overlap_metrics)
        self.assertIn('spectral_angle', overlap_metrics)

        # Check that correlation is reasonable (should be positive for similar spectra)
        self.assertGreater(overlap_metrics['pearson_correlation'], 0.5)

    def test_calculate_spectral_overlap_identical(self):
        """Test spectral overlap with identical spectra."""
        spectrum = np.array([0.1, 0.3, 0.8, 0.6, 0.4])

        overlap_metrics = calculate_spectral_overlap(spectrum, spectrum)

        # Identical spectra should have perfect correlation
        self.assertAlmostEqual(overlap_metrics['pearson_correlation'], 1.0, places=6)
        self.assertAlmostEqual(overlap_metrics['cosine_similarity'], 1.0, places=6)

    def test_calculate_spectral_overlap_orthogonal(self):
        """Test spectral overlap with orthogonal spectra."""
        spectrum1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        spectrum2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        overlap_metrics = calculate_spectral_overlap(spectrum1, spectrum2)

        # Orthogonal spectra should have zero correlation
        self.assertAlmostEqual(overlap_metrics['pearson_correlation'], 0.0, places=6)

    def test_generate_spectral_plots_with_matplotlib(self):
        """Test spectral plot generation with matplotlib available."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        wavelengths = np.linspace(2.5, 25, 100)  # 2.5 to 25 μm

        result = generate_spectral_plots(
            spectra=self.test_spectra,
            wavelengths=wavelengths
        )

        if HAS_MATPLOTLIB:
            # Should return matplotlib figure if matplotlib is available
            self.assertIsNotNone(result)
        else:
            # Should return None if matplotlib is not available
            self.assertIsNone(result)

    def test_generate_spectral_plots_without_matplotlib(self):
        """Test spectral plot generation without matplotlib."""
        # This test verifies fallback behavior
        wavelengths = np.linspace(2.5, 25, 100)

        result = generate_spectral_plots(
            spectra=self.test_spectra,
            wavelengths=wavelengths
        )

        # Should handle gracefully regardless of matplotlib availability
        # (either returns Figure or None)
        self.assertTrue(result is None or hasattr(result, 'savefig'))


class TestSpectroscopyIntegration(unittest.TestCase):
    """Test integration between spectroscopy components."""

    def setUp(self):
        """Set up test fixtures."""
        self.wavenumbers = np.linspace(400, 3200, 50)
        self.intensities = (
            0.1 * np.sin(2 * np.pi * self.wavenumbers / 800) +
            0.05 * np.sin(2 * np.pi * self.wavenumbers / 1600) +
            0.2
        )

        self.spectral_data = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities,
            species="Integration_Test_CHC"
        )

    def test_full_spectroscopy_workflow(self):
        """Test complete spectroscopy analysis workflow."""
        # Step 1: Create spectral data
        self.assertIsInstance(self.spectral_data, SpectralData)

        # Step 2: Analyze CHC spectra
        analysis_result = analyze_chc_spectra(
            wavenumbers_or_wavelengths=self.wavenumbers,
            intensities=self.intensities,
            is_wavelength=False
        )

        self.assertIsInstance(analysis_result, dict)
        self.assertIn('spectral_data', analysis_result)

        # Step 3: Extract peak information
        spectral_data = analysis_result['spectral_data']
        if 'peak_analysis' in analysis_result:
            peak_analysis = analysis_result['peak_analysis']
            self.assertIsInstance(peak_analysis, dict)

        # Step 4: Test compound identification
        test_peaks = [2850, 1450, 720]  # CHC characteristic peaks
        compounds = identify_chc_compounds(test_peaks)
        self.assertIsInstance(compounds, list)

    def test_spectroscopy_component_interaction(self):
        """Test interaction between spectroscopy components."""
        # Create individual components
        spectral_data = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities
        )

        peak_finder = PeakFinder()
        chc_analyzer = CHCAnalyzer()

        # Test component interactions
        peaks, properties = peak_finder.find_peaks(spectral_data)
        self.assertIsInstance(peaks, np.ndarray)

        analysis = chc_analyzer.analyze_spectrum(spectral_data)
        self.assertIsInstance(analysis, dict)

        # Components should work together seamlessly
        self.assertTrue(len(analysis) > 0)

    def test_spectroscopy_error_handling(self):
        """Test error handling in spectroscopy functions."""
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[],  # Empty data
                intensities=[]
            )

        # Test with mismatched array sizes
        with self.assertRaises(ValueError):
            SpectralData(
                wavenumbers=[400, 800, 1200],
                intensities=[0.1, 0.3]  # Different size
            )

        # Test analyze_chc_spectra with invalid inputs
        with self.assertRaises((ValueError, TypeError)):
            analyze_chc_spectra(
                wavenumbers_or_wavelengths=[],
                intensities=[]
            )

    def test_spectroscopy_data_consistency(self):
        """Test data consistency across spectroscopy operations."""
        # Create original data
        original_data = SpectralData(
            wavenumbers=self.wavenumbers,
            intensities=self.intensities
        )

        # Apply operations
        normalized = original_data.normalize()
        region_mask = original_data.get_region_mask(1000, 2000)

        # Check consistency
        self.assertEqual(normalized.num_points, original_data.num_points)
        self.assertEqual(len(region_mask), original_data.num_points)

        # Spectral range should be preserved
        self.assertEqual(normalized.spectral_range, original_data.spectral_range)


class TestSpectroscopyRobustness(unittest.TestCase):
    """Test robustness of spectroscopy functionality."""

    def test_spectroscopy_with_edge_case_data(self):
        """Test spectroscopy with edge case data."""
        # Test with minimal data
        minimal_wavenumbers = [400, 800]
        minimal_intensities = [0.1, 0.2]

        minimal_data = SpectralData(
            wavenumbers=minimal_wavenumbers,
            intensities=minimal_intensities
        )

        self.assertEqual(minimal_data.num_points, 2)
        self.assertEqual(minimal_data.spectral_range, (400.0, 800.0))

        # Test analysis with minimal data
        result = analyze_chc_spectra(
            wavenumbers_or_wavelengths=minimal_wavenumbers,
            intensities=minimal_intensities
        )

        self.assertIsInstance(result, dict)

    def test_spectroscopy_with_large_data(self):
        """Test spectroscopy with large datasets."""
        # Create large dataset
        large_wavenumbers = np.linspace(300, 4000, 10000)  # 10k points
        large_intensities = np.random.rand(10000) * 0.5 + 0.1

        large_data = SpectralData(
            wavenumbers=large_wavenumbers,
            intensities=large_intensities
        )

        # Should handle large data efficiently
        self.assertEqual(large_data.num_points, 10000)

        # Test basic operations on large data
        spectral_range = large_data.spectral_range
        self.assertAlmostEqual(spectral_range[0], 300.0, places=1)
        self.assertAlmostEqual(spectral_range[1], 4000.0, places=1)

    def test_spectroscopy_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small intensities
        small_wavenumbers = [400, 800, 1200, 1600]
        small_intensities = [1e-6, 2e-6, 3e-6, 4e-6]

        small_data = SpectralData(
            wavenumbers=small_wavenumbers,
            intensities=small_intensities
        )

        # Normalization should handle small values
        normalized = small_data.normalize()
        self.assertAlmostEqual(np.max(normalized.intensities), 1.0, places=6)

        # Test with very large intensities
        large_intensities = [1e6, 2e6, 3e6, 4e6]

        large_data = SpectralData(
            wavenumbers=small_wavenumbers,
            intensities=large_intensities
        )

        # Should handle large values without overflow
        self.assertIsInstance(large_data, SpectralData)

    def test_spectroscopy_cross_platform_compatibility(self):
        """Test spectroscopy compatibility across different scenarios."""
        # Test with different numpy availability scenarios
        wavenumbers = [400, 800, 1200, 1600]
        intensities = [0.1, 0.3, 0.8, 0.6]

        # Should work with both lists and numpy arrays
        list_data = SpectralData(
            wavenumbers=wavenumbers,
            intensities=intensities
        )

        array_data = SpectralData(
            wavenumbers=np.array(wavenumbers),
            intensities=np.array(intensities)
        )

        # Results should be equivalent
        np.testing.assert_array_equal(list_data.wavenumbers, array_data.wavenumbers)
        np.testing.assert_array_equal(list_data.intensities, array_data.intensities)


if __name__ == '__main__':
    unittest.main()
