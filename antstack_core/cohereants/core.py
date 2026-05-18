"""Core numeric utilities for infrared cohereAnts analysis."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np


def _as_numeric_array(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric input, received {type(value).__name__}") from exc
    if arr.size == 0:
        raise ValueError("Numeric input cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Numeric input must contain only finite values")
    return arr


def _scalar_or_array(original: Any, result: np.ndarray) -> float | np.ndarray:
    if isinstance(original, Number) or np.asarray(original).ndim == 0:
        return float(np.asarray(result))
    return result


def validate_numeric_inputs(*values: Any) -> None:
    """Raise ``ValueError`` unless all values are finite numeric scalars or arrays."""
    for value in values:
        _as_numeric_array(value)


def calculate_wavelength_from_wavenumber(wavenumber_cm: Any) -> float | np.ndarray:
    """Convert wavenumber in cm^-1 to wavelength in micrometers."""
    values = _as_numeric_array(wavenumber_cm)
    if np.any(values <= 0):
        raise ValueError("Wavenumbers must be positive")
    result = 10000.0 / values
    return _scalar_or_array(wavenumber_cm, result)


def calculate_wavenumber_from_wavelength(wavelength_um: Any) -> float | np.ndarray:
    """Convert wavelength in micrometers to wavenumber in cm^-1."""
    values = _as_numeric_array(wavelength_um)
    if np.any(values <= 0):
        raise ValueError("Wavelengths must be positive")
    result = 10000.0 / values
    return _scalar_or_array(wavelength_um, result)


def calculate_atmospheric_transmission(wavelength_um: Any, distance: float = 1.0) -> float | np.ndarray:
    """Estimate clear-air infrared transmission for 1-25 micrometer wavelengths.

    This is a compact deterministic model with atmospheric windows around 3-5 um
    and 8-12 um, plus broad water/CO2 absorption bands. It is intended for
    reproducible comparative analysis, not site-specific radiative transfer.
    """
    wavelengths = _as_numeric_array(wavelength_um)
    if np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be positive")
    validate_numeric_inputs(distance)
    distance_scale = max(float(distance), 0.0) / 1000.0

    window_short = 0.72 * np.exp(-((wavelengths - 4.0) / 1.3) ** 2)
    window_long = 0.90 * np.exp(-((wavelengths - 10.0) / 2.2) ** 2)
    baseline = 0.08 + window_short + window_long
    water_absorption = 0.24 * np.exp(-((wavelengths - 6.3) / 0.9) ** 2)
    co2_absorption = 0.18 * np.exp(-((wavelengths - 15.0) / 1.4) ** 2)
    attenuation = np.exp(-0.12 * distance_scale)
    result = np.clip((baseline - water_absorption - co2_absorption) * attenuation, 0.0, 1.0)
    return _scalar_or_array(wavelength_um, result)


def calculate_response_time_improvement(traditional_time: Any, insect_time: Any) -> float | np.ndarray:
    """Return the response-time improvement factor traditional_time / insect_time."""
    return safe_division(traditional_time, insect_time)


def safe_division(numerator: Any, denominator: Any, default: float = np.inf) -> float | np.ndarray:
    """Divide while replacing zero-denominator and non-finite results with ``default``."""
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator_arr, denominator_arr)
    result = np.where(np.isfinite(result), result, default)
    if np.asarray(result).ndim == 0:
        return float(result)
    return result


__all__ = [
    "calculate_atmospheric_transmission",
    "calculate_response_time_improvement",
    "calculate_wavelength_from_wavenumber",
    "calculate_wavenumber_from_wavelength",
    "safe_division",
    "validate_numeric_inputs",
]
