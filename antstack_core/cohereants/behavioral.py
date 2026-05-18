"""Behavioral response analysis for cohereAnts experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # pragma: no cover - exercised by environments with SciPy
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


def _to_positive_array(values: Iterable[float], name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return arr.astype(float)


@dataclass
class BehavioralData:
    """Treatment/control response-time data with validated numeric arrays."""

    treatment_times: Iterable[float]
    control_times: Iterable[float]

    def __post_init__(self) -> None:
        self.treatment_times = _to_positive_array(self.treatment_times, "treatment_times")
        self.control_times = _to_positive_array(self.control_times, "control_times")

    @property
    def treatment_mean(self) -> float:
        return float(np.mean(self.treatment_times))

    @property
    def control_mean(self) -> float:
        return float(np.mean(self.control_times))

    @property
    def treatment_std(self) -> float:
        return float(np.std(self.treatment_times, ddof=1)) if self.treatment_times.size > 1 else 0.0

    @property
    def control_std(self) -> float:
        return float(np.std(self.control_times, ddof=1)) if self.control_times.size > 1 else 0.0

    @property
    def difference(self) -> float:
        return self.treatment_mean - self.control_mean

    @property
    def sample_sizes(self) -> dict[str, int]:
        return {"treatment": int(self.treatment_times.size), "control": int(self.control_times.size)}

    @property
    def can_perform_statistics(self) -> bool:
        return self.treatment_times.size >= 2 and self.control_times.size >= 2


class StatisticalAnalyzer:
    """Statistical tests and effect-size helpers for response-time data."""

    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def perform_t_test(self, data: BehavioralData) -> dict[str, float]:
        if not data.can_perform_statistics:
            return {"t_statistic": np.nan, "p_value": np.nan, "degrees_of_freedom": np.nan}
        degenerate = self._degenerate_t_test(data)
        if degenerate is not None:
            return degenerate
        if scipy_stats is not None:
            result = scipy_stats.ttest_ind(
                data.treatment_times,
                data.control_times,
                equal_var=False,
                nan_policy="omit",
            )
            df = self._welch_degrees_of_freedom(data)
            return {
                "t_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "degrees_of_freedom": float(df),
            }

        t_stat, df = self._welch_t_statistic(data)
        return {"t_statistic": float(t_stat), "p_value": np.nan, "degrees_of_freedom": float(df)}

    def calculate_cohens_d(self, data: BehavioralData) -> float:
        n1 = data.treatment_times.size
        n2 = data.control_times.size
        if n1 < 2 or n2 < 2:
            return float("nan")
        pooled_var = (
            ((n1 - 1) * data.treatment_std**2 + (n2 - 1) * data.control_std**2)
            / max(n1 + n2 - 2, 1)
        )
        pooled_std = float(np.sqrt(pooled_var))
        if pooled_std == 0:
            return 0.0
        return float(data.difference / pooled_std)

    def calculate_confidence_interval(self, data: BehavioralData) -> dict[str, float]:
        if not data.can_perform_statistics:
            return {"lower_bound": np.nan, "upper_bound": np.nan, "confidence_level": 1 - self.alpha}
        se = np.sqrt(
            np.var(data.treatment_times, ddof=1) / data.treatment_times.size
            + np.var(data.control_times, ddof=1) / data.control_times.size
        )
        df = self._welch_degrees_of_freedom(data)
        critical = (
            float(scipy_stats.t.ppf(1 - self.alpha / 2, df))
            if scipy_stats is not None and np.isfinite(df) and df > 0
            else 1.96
        )
        margin = critical * se
        return {
            "lower_bound": float(data.difference - margin),
            "upper_bound": float(data.difference + margin),
            "confidence_level": float(1 - self.alpha),
        }

    def _welch_t_statistic(self, data: BehavioralData) -> tuple[float, float]:
        variance = (
            np.var(data.treatment_times, ddof=1) / data.treatment_times.size
            + np.var(data.control_times, ddof=1) / data.control_times.size
        )
        if variance == 0:
            return 0.0, self._welch_degrees_of_freedom(data)
        return float(data.difference / np.sqrt(variance)), self._welch_degrees_of_freedom(data)

    @staticmethod
    def _welch_degrees_of_freedom(data: BehavioralData) -> float:
        n1 = data.treatment_times.size
        n2 = data.control_times.size
        s1 = np.var(data.treatment_times, ddof=1)
        s2 = np.var(data.control_times, ddof=1)
        numerator = (s1 / n1 + s2 / n2) ** 2
        denominator = (s1**2 / (n1**2 * max(n1 - 1, 1))) + (s2**2 / (n2**2 * max(n2 - 1, 1)))
        if denominator == 0:
            return float(n1 + n2 - 2)
        return float(numerator / denominator)

    def _degenerate_t_test(self, data: BehavioralData) -> dict[str, float] | None:
        """Return deterministic Welch-test results for zero-variance samples.

        SciPy correctly warns that identical samples lose floating-point
        precision. For validated behavioral pipelines that warning is expected
        edge-case behavior, so the public API handles it explicitly.
        """
        treatment_var = float(np.var(data.treatment_times, ddof=1))
        control_var = float(np.var(data.control_times, ddof=1))
        if treatment_var != 0.0 or control_var != 0.0:
            return None

        df = self._welch_degrees_of_freedom(data)
        if data.difference == 0:
            return {"t_statistic": 0.0, "p_value": 1.0, "degrees_of_freedom": float(df)}

        direction = -1.0 if data.difference < 0 else 1.0
        return {
            "t_statistic": float(direction * np.inf),
            "p_value": 0.0,
            "degrees_of_freedom": float(df),
        }


class BehavioralAnalyzer:
    """High-level behavioral response analyzer."""

    def __init__(self, alpha: float = 0.05):
        self.stats = StatisticalAnalyzer(alpha=alpha)

    def analyze_response(self, treatment_times: Iterable[float], control_times: Iterable[float]) -> dict[str, float]:
        data = BehavioralData(treatment_times, control_times)
        t_test = self.stats.perform_t_test(data)
        ci = self.stats.calculate_confidence_interval(data)
        cohens_d = self.stats.calculate_cohens_d(data)
        p_value = t_test["p_value"]
        significant = bool(np.isfinite(p_value) and p_value < self.stats.alpha)
        return {
            "treatment_mean": data.treatment_mean,
            "control_mean": data.control_mean,
            "difference": data.difference,
            "treatment_std": data.treatment_std,
            "control_std": data.control_std,
            "cohens_d": cohens_d,
            "significant": significant,
            **t_test,
            **ci,
        }


def analyze_behavioral_response(treatment_times: Iterable[float], control_times: Iterable[float]) -> dict[str, float]:
    """Analyze treatment/control response-time differences."""
    return BehavioralAnalyzer().analyze_response(treatment_times, control_times)


def calculate_response_statistics(response_data: Iterable[float]) -> dict[str, float]:
    """Return descriptive statistics for one response-time series."""
    arr = _to_positive_array(response_data, "response_data")
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def calculate_power_analysis(treatment_times: Iterable[float], control_times: Iterable[float]) -> dict[str, float]:
    """Estimate observed effect and approximate two-sample power."""
    data = BehavioralData(treatment_times, control_times)
    effect = abs(StatisticalAnalyzer().calculate_cohens_d(data))
    harmonic_n = 2 / (1 / data.treatment_times.size + 1 / data.control_times.size)
    noncentrality = effect * np.sqrt(harmonic_n / 2)
    if scipy_stats is not None:
        z_alpha = float(scipy_stats.norm.ppf(0.975))
        power = float(1 - scipy_stats.norm.cdf(z_alpha - noncentrality))
    else:
        power = float(min(max(noncentrality / 2.8, 0.0), 1.0))
    return {
        "effect_size": float(effect),
        "harmonic_sample_size": float(harmonic_n),
        "estimated_power": float(np.clip(power, 0.0, 1.0)),
    }


def generate_behavioral_plots(response_data: Iterable[float], time_points: Iterable[float] | None = None):
    """Generate a simple response-time line plot when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    responses = _to_positive_array(response_data, "response_data")
    times = np.arange(responses.size) if time_points is None else np.asarray(time_points, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, responses, marker="o", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Response")
    ax.set_title("Behavioral response")
    fig.tight_layout()
    return fig


__all__ = [
    "BehavioralAnalyzer",
    "BehavioralData",
    "StatisticalAnalyzer",
    "analyze_behavioral_response",
    "calculate_power_analysis",
    "calculate_response_statistics",
    "generate_behavioral_plots",
]
