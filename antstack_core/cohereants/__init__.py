"""Infrared, spectroscopy, and behavior analysis utilities for cohereAnts work."""

from .core import (
    calculate_atmospheric_transmission,
    calculate_response_time_improvement,
    calculate_wavelength_from_wavenumber,
    calculate_wavenumber_from_wavelength,
    safe_division,
    validate_numeric_inputs,
)
from .behavioral import (
    BehavioralAnalyzer,
    BehavioralData,
    StatisticalAnalyzer,
    analyze_behavioral_response,
    calculate_power_analysis,
    calculate_response_statistics,
    generate_behavioral_plots,
)
from .spectroscopy import (
    CHCAnalyzer,
    PeakFinder,
    SpectralData,
    analyze_chc_spectra,
    calculate_spectral_overlap,
    generate_spectral_plots,
    identify_chc_compounds,
)

__all__ = [
    "BehavioralAnalyzer",
    "BehavioralData",
    "CHCAnalyzer",
    "PeakFinder",
    "SpectralData",
    "StatisticalAnalyzer",
    "analyze_behavioral_response",
    "analyze_chc_spectra",
    "calculate_atmospheric_transmission",
    "calculate_power_analysis",
    "calculate_response_statistics",
    "calculate_response_time_improvement",
    "calculate_spectral_overlap",
    "calculate_wavelength_from_wavenumber",
    "calculate_wavenumber_from_wavelength",
    "generate_behavioral_plots",
    "generate_spectral_plots",
    "identify_chc_compounds",
    "safe_division",
    "validate_numeric_inputs",
]
