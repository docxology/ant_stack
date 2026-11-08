"""Statistical analysis utilities for scientific computing.

Comprehensive statistical methods for quantitative research including:
- Percentile bootstrap confidence intervals
- Scaling relationship analysis and power law detection
- Uncertainty quantification and error propagation
- Statistical significance testing
- Theoretical limit calculations

Following .cursorrules principles:
- Real methods only (no mocks)
- Comprehensive validation and error handling
- Professional implementation with proper statistical rigor
- Well-documented with appropriate references

References:
- Bootstrap methods: https://doi.org/10.1214/aos/1176344552
- Scaling laws analysis: https://doi.org/10.1126/science.1062081
- Statistical computing: https://doi.org/10.1201/9781315117270
"""

from __future__ import annotations
from typing import Sequence, Optional, Tuple, Dict, Any, Union
import math
import random


def bootstrap_mean_ci(
    values: Sequence[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval for the mean.
    
    Uses bootstrap resampling to estimate confidence intervals for the
    sample mean without assuming specific distributions.
    
    Args:
        values: Sample data values
        num_samples: Number of bootstrap samples
        alpha: Significance level (default: 0.05 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
        
    References:
        - Efron & Tibshirani bootstrap methods: https://doi.org/10.1201/9780429246593
        - Bootstrap confidence intervals: https://doi.org/10.1214/aos/1176344552
    """
    data = _finite_values(values)
    _validate_resampling(num_samples, alpha)
    rng = random.Random(seed)
    n = len(data)
    original_mean = math.fsum(value / n for value in data)
    if n == 1 or all(value == data[0] for value in data):
        return original_mean, original_mean, original_mean
    bootstrap_means = sorted(
        math.fsum(rng.choice(data) / n for _ in range(n))
        for _ in range(num_samples)
    )
    return (
        original_mean,
        _percentile(bootstrap_means, alpha / 2),
        _percentile(bootstrap_means, 1 - alpha / 2),
    )


def _finite_values(values: Sequence[float]) -> list[float]:
    """Validate a nonempty finite numeric vector without changing global RNGs."""
    try:
        data = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Data must be a finite numeric sequence") from exc
    if not data or not all(math.isfinite(value) for value in data):
        raise ValueError("Data must be a nonempty finite numeric sequence")
    return data


def _validate_resampling(num_samples: int, alpha: float) -> None:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolated empirical quantile (NumPy's default convention)."""
    position = (len(sorted_values) - 1) * probability
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    weight = position - left
    return (1 - weight) * sorted_values[left] + weight * sorted_values[right]


def analyze_scaling_relationship(
    x_values: Sequence[float], 
    y_values: Sequence[float]
) -> Dict[str, Any]:
    """Analyze scaling relationships and detect power laws.
    
    Performs comprehensive scaling analysis including:
    - Linear regression in log-log space
    - Power law exponent estimation
    - Goodness-of-fit metrics (R-squared)
    - Classification of scaling regimes
    
    Args:
        x_values: Independent variable values
        y_values: Dependent variable values
        
    Returns:
        Dictionary containing scaling analysis results
        
    References:
        - Power law detection: https://doi.org/10.1137/070710111
        - Scaling in complex systems: https://doi.org/10.1126/science.1062081
    """
    if len(x_values) == 0 or len(y_values) == 0:
        return {"valid": False, "error": "Empty input data"}
    if len(x_values) != len(y_values):
        return {"valid": False, "error": "Mismatched data lengths"}
    if len(x_values) < 3:
        return {"valid": False, "error": "Insufficient data points for scaling analysis"}
    try:
        x_data = _finite_values(x_values)
        y_data = _finite_values(y_values)
        if any(value <= 0 for value in x_data + y_data):
            return {"valid": False, "error": "Cannot analyze scaling with non-positive values"}
        log_x = [math.log10(value) for value in x_data]
        log_y = [math.log10(value) for value in y_data]
        n = len(log_x)
        mean_x = math.fsum(log_x) / n
        mean_y = math.fsum(log_y) / n
        centered_x = [value - mean_x for value in log_x]
        centered_y = [value - mean_y for value in log_y]
        ss_x = math.fsum(value * value for value in centered_x)
        if ss_x == 0:
            return {"valid": False, "error": "Singular matrix in regression"}
        scaling_exponent = math.fsum(x * y for x, y in zip(centered_x, centered_y)) / ss_x
        log_intercept = mean_y - scaling_exponent * mean_x
        intercept = 10 ** log_intercept
        ss_total = math.fsum(value * value for value in centered_y)
        ss_residual = math.fsum(
            (y - (scaling_exponent * x + log_intercept)) ** 2
            for x, y in zip(log_x, log_y)
        )
        # A constant response has undefined R²; it does not demonstrate fit quality.
        r_squared = 1 - ss_residual / ss_total if ss_total > 0 else None
        if all(value == y_data[0] for value in y_data):
            scaling_exponent = 0.0
            log_intercept = log_y[0]
            intercept = y_data[0]
            r_squared = None
        if intercept <= 0:
            return {"valid": False, "error": "Regression intercept underflow"}
        if not all(math.isfinite(value) for value in (scaling_exponent, intercept, log_intercept)):
            return {"valid": False, "error": "Nonfinite regression result"}
        if r_squared is not None and not math.isfinite(r_squared):
            return {"valid": False, "error": "Nonfinite regression result"}
        if abs(scaling_exponent) < 0.1:
            scaling_regime = "constant"
        elif abs(scaling_exponent - 1.0) < 0.1:
            scaling_regime = "linear"
        elif abs(scaling_exponent - 2.0) < 0.1:
            scaling_regime = "quadratic"
        elif abs(scaling_exponent - 3.0) < 0.1:
            scaling_regime = "cubic"
        elif scaling_exponent < 0:
            scaling_regime = "inverse"
        elif scaling_exponent < 1:
            scaling_regime = "sub-linear"
        elif scaling_exponent > 2:
            scaling_regime = "super-quadratic"
        else:
            scaling_regime = "super-linear"
        return {
            "valid": True,
            "scaling_exponent": scaling_exponent,
            "intercept": intercept,
            "r_squared": r_squared,
            "scaling_regime": scaling_regime,
            "sample_size": n,
            "log_intercept": log_intercept,
        }
    except (ArithmeticError, TypeError, ValueError) as exc:
        return {"valid": False, "error": f"Scaling analysis failed: {exc}"}


def estimate_theoretical_limits(
    workload_metrics: Dict[str, float],
    temperature_k: float = 300.0
) -> Dict[str, float]:
    """Estimate theoretical energy limits for computational workloads.
    
    Calculates fundamental physical limits including:
    - Landauer limit for irreversible computation
    - Thermodynamic efficiency bounds
    - Quantum computational limits
    - Practical efficiency estimates
    
    Args:
        workload_metrics: Dictionary of workload characteristics
        temperature_k: Operating temperature in Kelvin
        
    Returns:
        Dictionary of theoretical limit estimates
        
    References:
        - Landauer's principle: https://doi.org/10.1147/rd.53.0183
        - Quantum limits: https://doi.org/10.1103/PhysRevLett.85.441
        - Thermodynamic computing: https://doi.org/10.1038/nature23460
    """
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature_k must be finite and positive")
    for name, value in workload_metrics.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if "hot_temperature_k" in workload_metrics and workload_metrics["hot_temperature_k"] <= temperature_k:
        raise ValueError("hot_temperature_k must exceed temperature_k")
    k_boltzmann = 1.380649e-23  # J/K (Boltzmann constant)
    
    limits = {}
    
    # Landauer limit for bit erasure
    if "bits_erased" in workload_metrics:
        bits = workload_metrics["bits_erased"]
        landauer_limit = bits * k_boltzmann * temperature_k * math.log(2)
        limits["landauer_limit_j"] = landauer_limit
    elif "flops" in workload_metrics:
        # Rough approximation: assume 1 bit erased per FLOP
        estimated_bits = workload_metrics["flops"]
        landauer_limit = estimated_bits * k_boltzmann * temperature_k * math.log(2)
        limits["landauer_limit_j"] = landauer_limit
    
    # Thermal energy scale
    thermal_energy = k_boltzmann * temperature_k
    limits["thermal_energy_j"] = thermal_energy
    
    # Mechanical work limits (if applicable)
    if "mechanical_work_j" in workload_metrics:
        mechanical_work = workload_metrics["mechanical_work_j"]
        limits["mechanical_work_j"] = mechanical_work
        
        # Total theoretical minimum
        theoretical_min = limits.get("landauer_limit_j", 0) + mechanical_work
        limits["total_theoretical_j"] = theoretical_min
        
        # Carnot efficiency bound for heat engines
        if "hot_temperature_k" in workload_metrics:
            t_hot = workload_metrics["hot_temperature_k"]
            carnot_efficiency = 1 - (temperature_k / t_hot)
            limits["carnot_efficiency"] = max(0, carnot_efficiency)
    else:
        limits["total_theoretical_j"] = limits.get("landauer_limit_j", thermal_energy)
    
    # Quantum computation limits (rough estimates)
    if "quantum_gates" in workload_metrics:
        gates = workload_metrics["quantum_gates"]
        # Rough estimate based on energy-time uncertainty principle
        quantum_limit = gates * thermal_energy * 1e-6  # Very rough approximation
        limits["quantum_limit_j"] = quantum_limit
    
    return limits


def calculate_energy_efficiency_metrics(
    energy_values: list[float], 
    performance_values: list[float]
) -> Dict[str, float]:
    """Calculate comprehensive energy efficiency metrics.
    
    Args:
        energy_values: Energy consumption values (J)
        performance_values: Performance metrics (operations/s, accuracy, etc.)
        
    Returns:
        Dictionary of efficiency metrics
    """
    if not energy_values or not performance_values:
        return {"error": "Empty input data"}
    
    if len(energy_values) != len(performance_values):
        return {"error": "Mismatched data lengths"}
    try:
        energy_values = _finite_values(energy_values)
        performance_values = _finite_values(performance_values)
    except ValueError as exc:
        return {"error": str(exc)}
    if any(value < 0 for value in energy_values + performance_values):
        return {"error": "Energy and performance must be nonnegative"}
    
    # Basic efficiency metrics
    avg_energy = sum(energy_values) / len(energy_values)
    avg_performance = sum(performance_values) / len(performance_values)
    
    performance_per_joule = avg_performance / avg_energy if avg_energy > 0 else 0
    energy_per_performance = avg_energy / avg_performance if avg_performance > 0 else float('inf')
    
    # Energy-delay product (lower is better)
    # Approximate delay as inverse of performance
    delays = [1.0 / p if p > 0 else float('inf') for p in performance_values]
    edp_values = [e * d for e, d in zip(energy_values, delays)]
    avg_edp = sum(edp_values) / len(edp_values)
    
    return {
        "average_energy_j": avg_energy,
        "average_performance": avg_performance,
        "performance_per_joule": performance_per_joule,
        "energy_per_performance": energy_per_performance,
        "energy_delay_product": avg_edp,
        "efficiency_score": performance_per_joule  # Higher is better
    }
