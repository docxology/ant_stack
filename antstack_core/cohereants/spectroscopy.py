"""Cuticular hydrocarbon spectroscopy analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .core import calculate_wavenumber_from_wavelength

try:  # pragma: no cover - depends on optional SciPy runtime
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover
    scipy_find_peaks = None


@dataclass
class SpectralData:
    """Validated spectral intensity data indexed by wavenumber in cm^-1."""

    wavenumbers: Iterable[float]
    intensities: Iterable[float]
    species: str | None = None

    def __post_init__(self) -> None:
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=float)
        self.intensities = np.asarray(self.intensities, dtype=float)
        if self.wavenumbers.ndim != 1 or self.intensities.ndim != 1:
            raise ValueError("Spectral arrays must be one-dimensional")
        if self.wavenumbers.size == 0 or self.intensities.size == 0:
            raise ValueError("Spectral arrays cannot be empty")
        if self.wavenumbers.size != self.intensities.size:
            raise ValueError("Wavenumbers and intensities must have the same length")
        if not np.all(np.isfinite(self.wavenumbers)) or not np.all(np.isfinite(self.intensities)):
            raise ValueError("Spectral arrays must contain only finite values")
        if np.any(self.wavenumbers < 300) or np.any(self.wavenumbers > 6000):
            raise ValueError("Wavenumbers must stay within 300-6000 cm^-1")
        if np.any(self.intensities < 0):
            raise ValueError("Intensities must be non-negative")

    @property
    def num_points(self) -> int:
        return int(self.wavenumbers.size)

    @property
    def spectral_range(self) -> tuple[float, float]:
        return float(np.min(self.wavenumbers)), float(np.max(self.wavenumbers))

    @property
    def intensity_range(self) -> tuple[float, float]:
        return float(np.min(self.intensities)), float(np.max(self.intensities))

    def get_region_mask(self, min_wavenumber: float, max_wavenumber: float) -> np.ndarray:
        if min_wavenumber >= max_wavenumber:
            raise ValueError("min_wavenumber must be less than max_wavenumber")
        return (self.wavenumbers >= min_wavenumber) & (self.wavenumbers <= max_wavenumber)

    def normalize(self) -> "SpectralData":
        max_intensity = float(np.max(self.intensities))
        if max_intensity == 0:
            normalized = self.intensities.copy()
        else:
            normalized = self.intensities / max_intensity
        return SpectralData(self.wavenumbers.copy(), normalized, self.species)

    def baseline_correction(self, method: str = "linear") -> "SpectralData":
        if method != "linear":
            raise ValueError("Only linear baseline correction is supported")
        if self.num_points < 2:
            corrected = self.intensities.copy()
        else:
            baseline = np.interp(
                self.wavenumbers,
                [self.wavenumbers[0], self.wavenumbers[-1]],
                [self.intensities[0], self.intensities[-1]],
            )
            corrected = np.maximum(self.intensities - baseline + np.min(baseline), 0.0)
        return SpectralData(self.wavenumbers.copy(), corrected, self.species)


class PeakFinder:
    """Find and summarize spectral peaks."""

    def __init__(self, threshold_factor: float = 0.3, min_distance: int = 20):
        self.threshold_factor = float(threshold_factor)
        self.min_distance = int(min_distance)

    def find_peaks(self, spectral_data: SpectralData) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        intensities = spectral_data.intensities
        if intensities.size < 3 or np.allclose(intensities, intensities[0]):
            return np.array([], dtype=int), {"prominences": np.array([]), "widths": np.array([])}
        threshold = float(np.min(intensities) + self.threshold_factor * (np.max(intensities) - np.min(intensities)))
        if scipy_find_peaks is not None:
            distance = max(1, min(self.min_distance, max(1, intensities.size // 2)))
            peaks, props = scipy_find_peaks(intensities, height=threshold, distance=distance)
            props.setdefault("prominences", intensities[peaks] - np.median(intensities))
            props.setdefault("widths", np.ones(peaks.size))
            return peaks.astype(int), props

        peaks = []
        for idx in range(1, intensities.size - 1):
            if intensities[idx] > threshold and intensities[idx] > intensities[idx - 1] and intensities[idx] > intensities[idx + 1]:
                if not peaks or idx - peaks[-1] >= self.min_distance:
                    peaks.append(idx)
        peak_arr = np.asarray(peaks, dtype=int)
        return peak_arr, {
            "prominences": intensities[peak_arr] - np.median(intensities) if peak_arr.size else np.array([]),
            "widths": np.ones(peak_arr.size),
        }

    def analyze_peaks(self, spectral_data: SpectralData) -> dict[str, object]:
        peaks, props = self.find_peaks(spectral_data)
        return {
            "peak_count": int(peaks.size),
            "peak_indices": peaks,
            "peak_wavenumbers": spectral_data.wavenumbers[peaks],
            "peak_intensities": spectral_data.intensities[peaks],
            "peak_widths": np.asarray(props.get("widths", np.ones(peaks.size))),
            "peak_prominences": np.asarray(props.get("prominences", np.array([]))),
        }


CHC_GROUPS = [
    {"name": "CH stretch", "center": 2920.0, "tolerance": 120.0},
    {"name": "CH3 stretch", "center": 2850.0, "tolerance": 120.0},
    {"name": "CH2 bend", "center": 1450.0, "tolerance": 90.0},
    {"name": "CH3 bend", "center": 1375.0, "tolerance": 80.0},
    {"name": "long-chain rocking", "center": 720.0, "tolerance": 70.0},
]


class CHCAnalyzer:
    """Analyze cuticular hydrocarbon spectral signatures."""

    def __init__(self, peak_finder: PeakFinder | None = None):
        self.peak_finder = peak_finder or PeakFinder()

    def analyze_spectrum(self, spectral_data: SpectralData) -> dict[str, object]:
        peak_analysis = self.peak_finder.analyze_peaks(spectral_data)
        peak_wavenumbers = [float(x) for x in peak_analysis["peak_wavenumbers"]]
        return {
            "species": spectral_data.species,
            "peak_analysis": peak_analysis,
            "compound_identification": identify_chc_compounds(peak_wavenumbers),
            "spectral_characteristics": self.calculate_spectral_features(spectral_data),
            "quality_metrics": {
                "signal_range": float(np.ptp(spectral_data.intensities)),
                "nonzero_fraction": float(np.count_nonzero(spectral_data.intensities) / spectral_data.num_points),
            },
        }

    def identify_functional_groups(self, peak_wavenumbers: Iterable[float]) -> list[dict[str, float | str]]:
        groups = []
        for peak in peak_wavenumbers:
            for group in CHC_GROUPS:
                delta = abs(float(peak) - group["center"])
                if delta <= group["tolerance"]:
                    groups.append(
                        {
                            "name": group["name"],
                            "wavenumber": float(peak),
                            "confidence": float(max(0.0, 1.0 - delta / group["tolerance"])),
                        }
                    )
        return groups

    def calculate_spectral_features(self, spectral_data: SpectralData) -> dict[str, object]:
        peak_analysis = self.peak_finder.analyze_peaks(spectral_data)
        regions = {
            "fingerprint": int(np.sum(spectral_data.get_region_mask(600, 1500))),
            "ch_stretch": int(np.sum(spectral_data.get_region_mask(2800, 3100))),
        }
        return {
            "intensity_stats": {
                "mean": float(np.mean(spectral_data.intensities)),
                "std": float(np.std(spectral_data.intensities)),
                "max": float(np.max(spectral_data.intensities)),
            },
            "peak_stats": {
                "count": int(peak_analysis["peak_count"]),
                "max_peak": float(np.max(peak_analysis["peak_intensities"])) if peak_analysis["peak_count"] else 0.0,
            },
            "spectral_regions": regions,
        }

    def compare_spectra(self, spectrum1: SpectralData, spectrum2: SpectralData) -> dict[str, object]:
        overlap = calculate_spectral_overlap(spectrum1.intensities, spectrum2.intensities)
        peaks1 = self.peak_finder.analyze_peaks(spectrum1)["peak_wavenumbers"]
        peaks2 = self.peak_finder.analyze_peaks(spectrum2)["peak_wavenumbers"]
        return {
            "similarity_score": float(overlap["cosine_similarity"]),
            "overlap_metrics": overlap,
            "peak_differences": np.setdiff1d(np.round(peaks1, 1), np.round(peaks2, 1)).tolist(),
        }


def identify_chc_compounds(peak_wavenumbers: Iterable[float]) -> list[dict[str, object]]:
    """Identify likely CHC compound groups from characteristic peaks."""
    groups = CHCAnalyzer().identify_functional_groups(peak_wavenumbers)
    if not groups:
        return []
    grouped: dict[str, dict[str, object]] = {}
    for group in groups:
        name = str(group["name"])
        entry = grouped.setdefault(name, {"name": name, "confidence": 0.0, "matched_peaks": []})
        entry["confidence"] = max(float(entry["confidence"]), float(group["confidence"]))
        entry["matched_peaks"].append(float(group["wavenumber"]))
    return list(grouped.values())


def calculate_spectral_overlap(spectrum1: Iterable[float], spectrum2: Iterable[float]) -> dict[str, float]:
    """Calculate correlation, cosine similarity, and spectral angle between spectra."""
    a = np.asarray(spectrum1, dtype=float)
    b = np.asarray(spectrum2, dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("Spectra must be non-empty arrays of equal length")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom and abs(float(np.dot(a, b))) < 1e-12:
        pearson = 0.0
    elif np.allclose(a, a[0]) or np.allclose(b, b[0]):
        pearson = 1.0 if np.allclose(a, b) else 0.0
    else:
        pearson = float(np.corrcoef(a, b)[0, 1])
        if abs(pearson) < 1e-12:
            pearson = 0.0
    cosine = float(np.dot(a, b) / denom) if denom else 0.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        "pearson_correlation": pearson,
        "cosine_similarity": cosine,
        "spectral_angle": float(np.arccos(cosine)),
    }


def analyze_chc_spectra(
    wavenumbers_or_wavelengths: Iterable[float],
    intensities: Iterable[float],
    is_wavelength: bool = False,
) -> dict[str, object]:
    """Analyze one CHC spectrum from wavenumbers or wavelengths."""
    axis = np.asarray(wavenumbers_or_wavelengths, dtype=float)
    values = np.asarray(intensities, dtype=float)
    if axis.size == 0 or values.size == 0:
        raise ValueError("Spectral axis and intensities cannot be empty")
    wavenumbers = calculate_wavenumber_from_wavelength(axis) if is_wavelength else axis
    spectral_data = SpectralData(wavenumbers=wavenumbers, intensities=values)
    peak_finder = PeakFinder()
    analyzer = CHCAnalyzer(peak_finder)
    return {
        "spectral_data": spectral_data,
        "peak_analysis": peak_finder.analyze_peaks(spectral_data),
        "chc_analysis": analyzer.analyze_spectrum(spectral_data),
    }


def generate_spectral_plots(spectra: dict[str, Iterable[float]], wavelengths: Iterable[float]):
    """Generate an overlay plot for spectral series when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    x = np.asarray(wavelengths, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, intensity in spectra.items():
        y = np.asarray(intensity, dtype=float)
        if y.size != x.size:
            source = np.linspace(float(np.min(x)), float(np.max(x)), y.size)
            y = np.interp(x, source, y)
        ax.plot(x, y, label=name)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Intensity")
    ax.legend()
    fig.tight_layout()
    return fig


__all__ = [
    "CHCAnalyzer",
    "PeakFinder",
    "SpectralData",
    "analyze_chc_spectra",
    "calculate_spectral_overlap",
    "generate_spectral_plots",
    "identify_chc_compounds",
]
