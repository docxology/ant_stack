"""Statistical analysis utilities for veridical reporting.

This module provides comprehensive statistical analysis capabilities including:
- Bootstrap confidence interval calculation
- Uncertainty quantification and propagation
- Quality metrics and validation checks
- Statistical significance testing

Following .cursorrules principles:
- Rigorous statistical methods
- Comprehensive uncertainty quantification
- Bootstrap-based confidence intervals
- Professional statistical reporting
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

from scipy import stats

from .statistics import bootstrap_mean_ci, _finite_values, _validate_resampling


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for empirical evidence.

    This class provides statistical analysis capabilities including:
    - Bootstrap confidence intervals
    - Uncertainty quantification
    - Quality assessment metrics
    - Validation checks

    Attributes:
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals (default 0.95)
        random_seed: Random seed for reproducible results
    """

    def __init__(self,
                 bootstrap_samples: int = 1000,  # Increased from 1000 to match review requirements
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """Initialize statistical analyzer.

        Args:
            bootstrap_samples: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_seed: Random seed for reproducibility
        """
        _validate_resampling(bootstrap_samples, 1 - confidence_level)
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.random_seed = random_seed

    def calculate_measurement_uncertainty(self,
                                        values: List[float],
                                        method: str = "standard_error") -> float:
        """Calculate measurement uncertainty for a set of values.

        Args:
            values: List of measurement values
            method: Uncertainty calculation method ("standard_error", "std_dev")

        Returns:
            Uncertainty value

        Raises:
            ValueError: If invalid method or insufficient data
        """
        values = _finite_values(values)
        if method not in ("standard_error", "std_dev"):
            raise ValueError(f"Unknown uncertainty method: {method}")
        n = len(values)
        if n < 2:
            return float("nan")  # A sample variance cannot be estimated from one observation.
        scale = max(abs(value) for value in values)
        if scale == 0:
            return 0.0
        scaled_mean = math.fsum(value / scale for value in values) / n
        deviation = math.hypot(*(value / scale - scaled_mean for value in values)) / math.sqrt(n - 1)
        result = scale * deviation
        if method == "standard_error":
            result /= math.sqrt(n)
        if not math.isfinite(result):
            raise ValueError("Sample uncertainty exceeds the finite numeric range")
        return result

    def calculate_statistical_metrics(self,
                                    measurements: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate comprehensive statistical metrics for measurements.

        Args:
            measurements: Dictionary mapping parameter names to value lists

        Returns:
            Dictionary of statistical metrics
        """
        metrics = {}

        for param, values in measurements.items():
            if not values:
                continue
            values = _finite_values(values)

            n = len(values)

            # Basic statistics
            mean_val = math.fsum(value / n for value in values)
            metrics[f"{param}_mean"] = mean_val
            metrics[f"{param}_n"] = n
            metrics[f"{param}_std"] = None
            metrics[f"{param}_se"] = None

            if n > 1:
                # Standard deviation
                uncertainty = self.calculate_measurement_uncertainty(values, "std_dev")
                metrics[f"{param}_std"] = uncertainty

                # Standard error
                se = self.calculate_measurement_uncertainty(values, "standard_error")
                metrics[f"{param}_se"] = se

                # Range
                metrics[f"{param}_min"] = min(values)
                metrics[f"{param}_max"] = max(values)
                metrics[f"{param}_range"] = max(values) - min(values)

                # Additional statistics if numpy available
                if HAS_NUMPY:
                    try:
                        np_values = np.array(values)
                        metrics[f"{param}_median"] = float(np.median(np_values))
                        metrics[f"{param}_q25"] = float(np.percentile(np_values, 25.0))
                        metrics[f"{param}_q75"] = float(np.percentile(np_values, 75.0))
                        metrics[f"{param}_iqr"] = float(np.subtract(*np.percentile(np_values, [75.0, 25.0])))
                    except (ValueError, TypeError):
                        # Fallback if numpy percentile fails
                        sorted_values = sorted(values)
                        n = len(sorted_values)
                        metrics[f"{param}_median"] = sorted_values[n // 2]
                        q25_idx = n // 4
                        q75_idx = 3 * n // 4
                        metrics[f"{param}_q25"] = sorted_values[q25_idx]
                        metrics[f"{param}_q75"] = sorted_values[q75_idx]
                        metrics[f"{param}_iqr"] = metrics[f"{param}_q75"] - metrics[f"{param}_q25"]

        return metrics

    def calculate_confidence_intervals(self,
                                     measurements,
                                     metrics: Optional[Dict[str, Any]] = None):
        """Percentile-bootstrap intervals using identical local-RNG paths for both shapes.

        A singleton's resampling distribution is degenerate; it cannot estimate
        population uncertainty. Evidence reports suppress that interval explicitly.
        Dictionary keys encode the actual confidence percentage, e.g. ``x_99ci``.
        """
        def interval(values):
            return bootstrap_mean_ci(
                values, num_samples=self.bootstrap_samples,
                alpha=1 - self.confidence_level, seed=self.random_seed,
            )

        if not isinstance(measurements, dict):
            return interval(measurements)
        label = f"{100 * self.confidence_level:g}"
        return {
            f"{param}_{label}ci": interval(values)[1:]
            for param, values in measurements.items() if len(values) > 0
        }

    def assess_measurement_quality(self, values: List[float]) -> Dict[str, Any]:
        """Assess quality metrics for a single measurement vector."""
        values = _finite_values(values)
        mean_val = math.fsum(value / len(values) for value in values)
        std_dev = self.calculate_measurement_uncertainty(values, "std_dev") if len(values) > 1 else None
        cv = abs(std_dev / mean_val) if std_dev is not None and mean_val else None
        ci = self.calculate_confidence_intervals(values) if len(values) > 1 else None
        sample_score = min(1.0, len(values) / 10.0)
        consistency_score = 1.0 / (1.0 + cv) if cv is not None else 0.0
        quality_score = 0.5 * sample_score + 0.5 * consistency_score
        return {
            "sample_size": len(values),
            "mean": mean_val,
            "std_dev": std_dev,
            "coefficient_of_variation": cv,
            "confidence_interval": ci,
            "quality_score": quality_score,
        }

    def detect_outliers(self, values: List[float], threshold: float = 1.5) -> List[float]:
        """Detect outliers using the IQR rule."""
        values = _finite_values(values)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if len(values) < 4:
            return []
        sorted_values = sorted(values)
        if HAS_NUMPY:
            q1, q3 = np.percentile(sorted_values, [25, 75])
        else:
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(3 * len(sorted_values)) // 4]
        iqr = q3 - q1
        if iqr == 0:
            return []
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return [value for value in values if value < lower or value > upper]

    def calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d (group2 minus group1); zero variance is explicit, not no effect."""
        group1 = _finite_values(group1)
        group2 = _finite_values(group2)
        if len(group1) < 2 or len(group2) < 2:
            return float("nan")
        mean1 = sum(group1) / len(group1)
        mean2 = sum(group2) / len(group2)
        var1 = sum((x - mean1) ** 2 for x in group1) / (len(group1) - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (len(group2) - 1)
        pooled = math.sqrt(((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / (len(group1) + len(group2) - 2))
        difference = mean2 - mean1
        if pooled == 0:
            return math.copysign(float("inf"), difference) if difference else float("nan")
        return difference / pooled

    def calculate_statistical_power(self, group1: List[float], group2: List[float], alpha: float = 0.05) -> float:
        """Two-sided normal-approximation power using observed pooled effect size.

        This is not an independent validation or a prospective power design.
        """
        _validate_resampling(1, alpha)
        effect = abs(self.calculate_effect_size(group1, group2))
        if math.isnan(effect):
            return float("nan")
        signal = effect * math.sqrt(len(group1) * len(group2) / (len(group1) + len(group2)))
        critical = stats.norm.isf(alpha / 2)
        return float(stats.norm.sf(critical - signal) + stats.norm.cdf(-critical - signal))

    def perform_validation_checks(self,
                                  measurements: Dict[str, List[float]],
                                  uncertainty: Dict[str, float]) -> Dict[str, Optional[bool]]:
        """Descriptive data checks; power and independent replication are unperformed."""
        nonempty = bool(measurements) and all(len(values) > 0 for values in measurements.values())
        finite = nonempty and all(
            all(math.isfinite(value) for value in values)
            for values in measurements.values()
        )
        precision = None
        if nonempty and all(name in uncertainty for name in measurements):
            precision = all(
                math.isfinite(uncertainty[name])
                and uncertainty[name] <= 0.1 * abs(sum(values) / len(values))
                for name, values in measurements.items()
            )
        return {
            "sufficient_data_points": nonempty and all(len(values) >= 3 for values in measurements.values()),
            "statistical_power": None,
            "measurement_precision": precision,
            "reproducibility": None,
            "data_consistency": bool(finite),
        }

    def calculate_quality_metrics(self,
                                measurements: Dict[str, List[float]],
                                checks: Dict[str, Optional[bool]]) -> Dict[str, float]:
        """Calculate quality metrics for the analysis.

        Args:
            measurements: Dictionary of measurement data
            checks: Dictionary of validation check results

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Data completeness
        total_params = len(measurements)
        complete_params = sum(1 for values in measurements.values() if values)
        metrics["data_completeness"] = complete_params / total_params if total_params > 0 else 0.0

        # Measurement consistency (inverse of average coefficient of variation)
        cv_values = []
        for param, values in measurements.items():
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                if abs(mean_val) > 0:
                    std_val = self.calculate_measurement_uncertainty(values, "std_dev")
                    cv = std_val / abs(mean_val)
                    cv_values.append(cv)

        if cv_values:
            avg_cv = sum(cv_values) / len(cv_values)
            metrics["measurement_consistency"] = 1.0 / (1.0 + avg_cv)  # Higher is better
        else:
            metrics["measurement_consistency"] = 0.5  # Neutral score

        # A descriptive check-pass fraction, not an inferential robustness estimate.
        performed = [value for value in checks.values() if value is not None]
        metrics["statistical_robustness"] = sum(value is True for value in performed) / len(performed) if performed else 0.0

        # Data quality score (weighted combination)
        weights = {
            "data_completeness": 0.3,
            "measurement_consistency": 0.4,
            "statistical_robustness": 0.3
        }

        quality_score = sum(
            metrics.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )
        metrics["overall_quality"] = quality_score

        return metrics

    def analyze_measurement_quality(self,
                                  measurements: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform complete measurement quality analysis.

        Args:
            measurements: Dictionary of measurement data

        Returns:
            Complete analysis results
        """
        if not measurements:
            return {
                "overall_quality": 0.0,
                "individual_metrics": {},
                "recommendations": ["Collect measurements"],
                "uncertainty": {},
                "statistical_metrics": {},
                "confidence_intervals": {},
                "validation_checks": {
                    "sufficient_data_points": False,
                    "statistical_power": None,
                    "measurement_precision": None,
                    "reproducibility": None,
                    "data_consistency": False,
                },
                "quality_metrics": {"overall_quality": 0.0},
            }
        measurements = {
            name: _finite_values(values) if len(values) else []
            for name, values in measurements.items()
        }

        # Calculate uncertainties
        uncertainty = {}
        for param, values in measurements.items():
            if len(values) > 1:
                uncertainty[param] = self.calculate_measurement_uncertainty(values)

        # Calculate statistical metrics
        metrics = self.calculate_statistical_metrics(measurements)

        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(
            {name: values for name, values in measurements.items() if len(values) > 1}, metrics
        )

        # Perform validation checks
        validation_checks = self.perform_validation_checks(measurements, uncertainty)

        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(measurements, validation_checks)

        individual_metrics = {
            name: self.assess_measurement_quality(values)
            for name, values in measurements.items()
            if values
        }
        overall_quality = (
            sum(m["quality_score"] for m in individual_metrics.values()) / len(individual_metrics)
            if individual_metrics else 0.0
        )
        return {
            "overall_quality": overall_quality,
            "individual_metrics": individual_metrics,
            "recommendations": [] if overall_quality >= 0.7 else ["Increase sample size or reduce measurement variance"],
            "uncertainty": uncertainty,
            "statistical_metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "validation_checks": validation_checks,
            "quality_metrics": quality_metrics
        }

    def perform_k_fold_cross_validation(self,
                                       measurements: Dict[str, List[float]],
                                       k_folds: int = 5) -> Dict[str, Any]:
        """Balanced, nonshuffled folds; every observation is held out exactly once."""
        if isinstance(k_folds, bool) or not isinstance(k_folds, int) or k_folds < 2:
            raise ValueError("k_folds must be an integer of at least 2")
        results = {}
        for param, values in measurements.items():
            values = _finite_values(values)
            n = len(values)
            if n < k_folds:
                raise ValueError("k_folds cannot exceed the number of observations")
            base, remainder = divmod(n, k_folds)
            scores = []
            test_indices = []
            start = 0
            for fold in range(k_folds):
                end = start + base + (fold < remainder)
                test = values[start:end]
                train = values[:start] + values[end:]
                scores.append(abs(sum(test) / len(test) - sum(train) / len(train)))
                test_indices.append(list(range(start, end)))
                start = end
            results[param] = {
                "mean_cv_error": sum(scores) / len(scores),
                "std_cv_error": self.calculate_measurement_uncertainty(scores, "std_dev"),
                "cv_scores": scores,
                "test_indices": test_indices,
                "shuffle": False,
            }
        return results

    def detect_statistical_significance(self,
                                      measurements: Dict[str, List[float]],
                                      baseline: Optional[float] = None,
                                      alpha: float = 0.05) -> Dict[str, Any]:
        """Two-sided one-sample Student t tests against baseline (default zero)."""
        _validate_resampling(1, alpha)
        test_baseline = 0.0 if baseline is None else baseline
        if not math.isfinite(test_baseline):
            raise ValueError("baseline must be finite")
        results = {}
        for param, values in measurements.items():
            if len(values) < 2:
                if values:
                    _finite_values(values)
                results[param] = {"testable": False, "reason": "insufficient_data", "significant": False, "p_value": None}
                continue
            values = _finite_values(values)
            mean_val = sum(values) / len(values)
            se = self.calculate_measurement_uncertainty(values)
            difference = mean_val - test_baseline
            if se == 0:
                t_statistic = math.copysign(float("inf"), difference) if difference else 0.0
                p_value = 0.0 if difference else 1.0
                df = None
                margin = 0.0
            else:
                df = len(values) - 1
                t_statistic = difference / se
                p_value = float(2 * stats.t.sf(abs(t_statistic), df))
                margin = float(stats.t.isf(alpha / 2, df) * se)
            results[param] = {
                "testable": True,
                "status": "degenerate_zero_variance" if se == 0 else "ok",
                "mean": mean_val,
                "baseline": test_baseline,
                "t_statistic": t_statistic,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "significant": p_value < alpha,
                "alpha": alpha,
                "confidence_interval": (mean_val - margin, mean_val + margin),
                "method": "one_sample_student_t",
            }
        return results


def analyze_measurement_quality(measurements: Dict[str, List[float]],
                              bootstrap_samples: int = 1000) -> Dict[str, Any]:
    """Convenience function for measurement quality analysis.

    Args:
        measurements: Dictionary of measurement data
        bootstrap_samples: Number of bootstrap samples

    Returns:
        Complete quality analysis results
    """
    analyzer = StatisticalAnalyzer(bootstrap_samples=bootstrap_samples)
    return analyzer.analyze_measurement_quality(measurements)


def detect_significance(measurements: Dict[str, List[float]],
                       baseline: Optional[float] = None,
                       alpha: float = 0.05) -> Dict[str, Any]:
    """Convenience function for significance testing.

    Args:
        measurements: Dictionary of measurement data
        baseline: Baseline value for comparison
        alpha: Significance level

    Returns:
        Significance test results
    """
    analyzer = StatisticalAnalyzer()
    _validate_resampling(1, alpha)
    for values in measurements.values():
        if values:
            _finite_values(values)
    names = list(measurements)
    significant_differences: Dict[str, Dict[str, Any]] = {}
    effect_sizes: Dict[str, float] = {}
    power_analysis: Dict[str, float] = {}

    if baseline is not None:
        baseline_results = analyzer.detect_statistical_significance(measurements, baseline, alpha)
        for name, result in baseline_results.items():
            significant_differences[f"{name}_vs_baseline"] = result

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            group1 = measurements.get(left, [])
            group2 = measurements.get(right, [])
            key = f"{left}_vs_{right}"
            if len(group1) < 2 or len(group2) < 2:
                effect = float("nan")
                power = float("nan")
                result = {"testable": False, "reason": "insufficient_data", "p_value": None, "significant": False, "alpha": alpha}
            else:
                effect = analyzer.calculate_effect_size(group1, group2)
                power = analyzer.calculate_statistical_power(group1, group2, alpha=alpha)
                mean1 = sum(group1) / len(group1)
                mean2 = sum(group2) / len(group2)
                v1 = analyzer.calculate_measurement_uncertainty(group1) ** 2
                v2 = analyzer.calculate_measurement_uncertainty(group2) ** 2
                se = math.sqrt(v1 + v2)
                difference = mean2 - mean1
                if se == 0:
                    t_stat = math.copysign(float("inf"), difference) if difference else 0.0
                    p_value = 0.0 if difference else 1.0
                    df = None
                    margin = 0.0
                else:
                    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(group1) - 1) + v2 ** 2 / (len(group2) - 1))
                    t_stat = difference / se
                    p_value = float(2 * stats.t.sf(abs(t_stat), df))
                    margin = float(stats.t.isf(alpha / 2, df) * se)
                result = {
                    "testable": True,
                    "status": "degenerate_zero_variance" if se == 0 else "ok",
                    "method": "welch_t",
                    "t_statistic": t_stat,
                    "degrees_of_freedom": df,
                    "p_value": p_value,
                    "significant": p_value < alpha,
                    "alpha": alpha,
                    "confidence_interval": (difference - margin, difference + margin),
                }
            significant_differences[key] = result
            effect_sizes[key] = effect
            power_analysis[key] = power

    return {
        "significant_differences": significant_differences,
        "effect_sizes": effect_sizes,
        "power_analysis": power_analysis,
    }


def perform_k_fold_cross_validation(measurements: Dict[str, List[float]],
                                   k_folds: int = 5) -> Dict[str, Any]:
    """Convenience function for k-fold cross-validation.

    Args:
        measurements: Dictionary of measurement data
        k_folds: Number of folds for cross-validation

    Returns:
        Cross-validation results
    """
    analyzer = StatisticalAnalyzer()
    return analyzer.perform_k_fold_cross_validation(measurements, k_folds)
