"""
Scaling Analysis Module

Provides comprehensive scaling analysis capabilities for Ant Stack modules,
including power law detection, regime classification, and multi-parameter analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    stats = None
    HAS_SCIPY = False

from .statistics import analyze_scaling_relationship, _percentile, _validate_resampling


@dataclass
class ScalingResult:
    """Container for scaling analysis results."""

    parameter_name: str
    parameter_values: List[float]
    response_values: List[float]

    scaling_exponent: Optional[float] = None
    intercept: Optional[float] = None
    r_squared: Optional[float] = None
    regime: Optional[str] = None

    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None

    valid: bool = False
    error: Optional[str] = None
    bootstrap_valid_samples: int = 0
    inference_method: str = "paired_percentile_bootstrap_descriptive"
    confidence_level: float = 0.95


@dataclass
class MultiParameterScaling:
    """Multi-parameter scaling analysis results."""

    primary_parameter: str
    secondary_parameters: Dict[str, List[float]]
    scaling_results: Dict[str, ScalingResult]

    interaction_effects: Dict[str, float] = None
    parameter_correlations: Optional[Dict[str, float]] = None


class ScalingAnalyzer:
    """Advanced scaling analyzer for Ant Stack modules.

    Provides comprehensive scaling analysis including:
    - Descriptive power-law fitting
    - Marginal multi-parameter scaling analysis
    - Paired-bootstrap exponent intervals
    - Descriptive parameter correlations (not interaction estimates)
    """

    def __init__(self, bootstrap_samples: int = 1000, confidence_level: float = 0.95,
                 random_seed: Optional[int] = 42):
        """Configure reproducible paired resampling without mutating ambient RNGs."""
        _validate_resampling(bootstrap_samples, 1 - confidence_level)
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.backend_capabilities = {
            "numpy": HAS_NUMPY,
            "scipy": HAS_SCIPY,
        }
        self.last_result: Optional[ScalingResult] = None

    def analyze_single_parameter_scaling(self,
                                       parameter_values: List[float],
                                       response_values: List[float],
                                       parameter_name: str = "parameter",
                                       response_name: str = "response") -> ScalingResult:
        """Analyze scaling relationship for single parameter.

        Args:
            parameter_values: Parameter values (x-axis)
            response_values: Response values (y-axis)
            parameter_name: Name of parameter for reporting
            response_name: Name of response for reporting

        Returns:
            ScalingResult with comprehensive analysis
        """
        scaling_result = analyze_scaling_relationship(parameter_values, response_values)
        analysis_valid = scaling_result["valid"]
        confidence_interval = None
        exponents = []
        if analysis_valid:
            rng = random.Random(self.random_seed)
            for _ in range(self.bootstrap_samples):
                indices = [rng.randrange(len(parameter_values)) for _ in parameter_values]
                boot_result = analyze_scaling_relationship(
                    [parameter_values[index] for index in indices],
                    [response_values[index] for index in indices],
                )
                if boot_result["valid"]:
                    exponents.append(boot_result["scaling_exponent"])
            if exponents:
                exponents.sort()
                alpha = 1 - self.confidence_level
                confidence_interval = (_percentile(exponents, alpha / 2), _percentile(exponents, 1 - alpha / 2))
        # Replicate count is Monte Carlo precision, not inferential degrees of freedom.
        # No residual/noise model is supplied, so no hypothesis-test p-value is claimed.

        result = ScalingResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            response_values=response_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            intercept=scaling_result.get("intercept"),
            r_squared=scaling_result.get("r_squared"),
            regime=scaling_result.get("scaling_regime"),
            confidence_interval=confidence_interval,
            p_value=None,
            valid=analysis_valid,
            error=scaling_result.get("error"),
            bootstrap_valid_samples=len(exponents),
            confidence_level=self.confidence_level,
        )
        self.last_result = result
        return result

    def analyze_multi_parameter_scaling(self,
                                      primary_param: str,
                                      primary_values: List[float],
                                      secondary_params: Dict[str, List[float]],
                                      response_values: List[float]) -> MultiParameterScaling:
        """Analyze scaling with multiple parameters.

        Args:
            primary_param: Name of primary parameter
            primary_values: Values of primary parameter
            secondary_params: Dictionary of secondary parameter names to values
            response_values: Response values

        Returns:
            MultiParameterScaling with interaction analysis
        """
        scaling_results = {}

        # Analyze primary parameter scaling
        primary_result = self.analyze_single_parameter_scaling(
            primary_values, response_values, primary_param, "response"
        )
        scaling_results[primary_param] = primary_result

        # Analyze secondary parameters
        for param_name, param_values in secondary_params.items():
            if len(param_values) == len(response_values):
                result = self.analyze_single_parameter_scaling(
                    param_values, response_values, param_name, "response"
                )
                scaling_results[param_name] = result

        # Marginal fits and parameter correlation cannot identify interaction terms.
        correlations = {}
        if primary_result.valid:
            for name, result in scaling_results.items():
                if name == primary_param or not result.valid:
                    continue
                left_mean = sum(primary_values) / len(primary_values)
                right_mean = sum(result.parameter_values) / len(result.parameter_values)
                left = [value - left_mean for value in primary_values]
                right = [value - right_mean for value in result.parameter_values]
                scale = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
                if scale > 0 and math.isfinite(scale):
                    correlations[name] = sum(x * y for x, y in zip(left, right)) / scale

        return MultiParameterScaling(
            primary_parameter=primary_param,
            secondary_parameters=secondary_params,
            scaling_results=scaling_results,
            interaction_effects=None,
            parameter_correlations=correlations,
        )

    def detect_scaling_regime(self, scaling_result: ScalingResult) -> Dict[str, Any]:
        """Detect and characterize scaling regime.

        Args:
            scaling_result: ScalingResult to analyze

        Returns:
            Dictionary with regime characterization
        """
        if (not scaling_result.valid or scaling_result.scaling_exponent is None
                or not math.isfinite(scaling_result.scaling_exponent)):
            return {"regime": "invalid", "confidence": None}
        exponent = scaling_result.scaling_exponent
        if abs(exponent) < 0.1:
            regime = "constant"
        elif abs(exponent - 1) < 0.1:
            regime = "linear"
        elif abs(exponent - 2) < 0.1:
            regime = "quadratic"
        elif abs(exponent - 3) < 0.1:
            regime = "cubic"
        elif exponent < 0:
            regime = "inverse"
        elif exponent < 1:
            regime = "sub-linear"
        elif exponent > 2:
            regime = "super-quadratic"
        else:
            regime = "super-linear"
        return {
            "regime": regime,
            "confidence": None,
            "r_squared": scaling_result.r_squared,
            "classification_tolerance": 0.1,
            "interpretation": self._interpret_regime(regime, exponent),
        }

    def _interpret_regime(self, regime: str, exponent: float) -> str:
        """Provide human-readable interpretation of scaling regime.

        Args:
            regime: Scaling regime classification
            exponent: Scaling exponent value

        Returns:
            Human-readable interpretation
        """
        interpretations = {
            "constant": f"Approximately constant response (exponent {exponent:.3f}); "
                        "the parameter has negligible effect on the response",
            "linear": f"Linear scaling with exponent {exponent:.3f}; response grows "
                      "proportionally with the parameter",
            "quadratic": f"Quadratic scaling with exponent {exponent:.3f}; response grows "
                         "with the square of the parameter",
            "cubic": f"Cubic scaling with exponent {exponent:.3f}; response grows with the "
                     "cube of the parameter",
            "sub-linear": f"Sub-linear scaling with exponent {exponent:.3f}; economies of "
                          "scale with increasing parameter",
            "super-linear": f"Super-linear scaling with exponent {exponent:.3f}; response "
                            "outpaces proportional growth"
        }

        return interpretations.get(regime, f"Unknown scaling regime with exponent {exponent:.3f}")

    def compare_scaling_regimes(self,
                               scaling_results: Dict[str, ScalingResult]) -> Dict[str, Any]:
        """Compare scaling regimes across different analyses.

        Args:
            scaling_results: Dictionary of scaling results to compare

        Returns:
            Dictionary with comparative analysis
        """
        if not scaling_results:
            return {"error": "No scaling results to compare"}

        valid_results = {
            name: result for name, result in scaling_results.items()
            if result.valid and result.scaling_exponent is not None
            and math.isfinite(result.scaling_exponent)
        }
        if not valid_results:
            return {"error": "No valid scaling results"}
        exponents = [result.scaling_exponent for result in valid_results.values()]
        r_squared_values = [
            result.r_squared for result in valid_results.values()
            if result.r_squared is not None and math.isfinite(result.r_squared)
        ]
        comparison = {
            "num_valid_results": len(valid_results),
            "exponent_range": (min(exponents), max(exponents)),
            "mean_r_squared": sum(r_squared_values) / len(r_squared_values) if r_squared_values else None,
            "regime_distribution": {},
        }
        for result in valid_results.values():
            regime = result.regime or "unknown"
            comparison["regime_distribution"][regime] = comparison["regime_distribution"].get(regime, 0) + 1
        name, result = max(valid_results.items(), key=lambda pair: abs(pair[1].scaling_exponent))
        comparison["most_challenging"] = {
            "parameter": name,
            "exponent": result.scaling_exponent,
            "regime": result.regime,
        }

        return comparison

    def generate_scaling_report(self, scaling_result: ScalingResult) -> str:
        """Generate comprehensive scaling analysis report.

        Args:
            scaling_result: ScalingResult to report on

        Returns:
            Formatted report string
        """
        if not scaling_result.valid:
            return f"Scaling analysis for {scaling_result.parameter_name}: INVALID ({scaling_result.error or 'insufficient data or invalid domain'})"

        report_lines = [
            f"Scaling Analysis Report: {scaling_result.parameter_name}",
            "=" * 60,
            f"Parameter: {scaling_result.parameter_name}",
            f"Data points: {len(scaling_result.parameter_values)}",
            f"Exponent: {scaling_result.scaling_exponent:.6f}",
            f"R²: {scaling_result.r_squared:.6f}" if scaling_result.r_squared is not None else "R²: undefined (constant response)",
            f"Intercept: {scaling_result.intercept:.6f}",
            f"Regime: {scaling_result.regime}",
        ]

        if scaling_result.confidence_interval:
            ci_low, ci_high = scaling_result.confidence_interval
            report_lines.append(f"{100 * scaling_result.confidence_level:g}% CI (paired percentile bootstrap): [{ci_low:.6f}, {ci_high:.6f}]")
        if scaling_result.p_value is not None:
            report_lines.append(f"P-value (supplied): {scaling_result.p_value:.6f}")
        else:
            report_lines.append("Hypothesis test: not performed; bootstrap intervals are descriptive resampling summaries.")
        report_lines.extend([
            "",
            "Interpretation:",
            self._interpret_regime(scaling_result.regime or "unknown",
                                  scaling_result.scaling_exponent or 0),
            "",
            "Recommendations:",
            self._generate_scaling_recommendations(scaling_result)
        ])

        return "\n".join(report_lines)

    def _generate_scaling_recommendations(self, scaling_result: ScalingResult) -> str:
        """Generate scaling recommendations based on analysis results.

        Args:
            scaling_result: ScalingResult to generate recommendations for

        Returns:
            Recommendation string
        """
        if not scaling_result.valid or scaling_result.scaling_exponent is None:
            return "Insufficient data for recommendations"

        exponent = scaling_result.scaling_exponent
        regime = scaling_result.regime

        recommendations = []

        if regime == "constant":
            recommendations.append("Parameter has minimal impact on performance - low optimization priority")
        elif regime == "linear":
            recommendations.append("Linear scaling suggests balanced resource allocation")
        elif regime == "sub-linear":
            recommendations.append("Sub-linear scaling indicates efficiency gains with scale - consider scaling up")
        elif regime == "super-linear":
            recommendations.append("Super-linear scaling indicates potential bottlenecks - high optimization priority")
        elif regime == "quadratic":
            recommendations.append("Quadratic scaling suggests algorithmic improvements needed")
        elif regime == "cubic":
            recommendations.append("Cubic scaling indicates fundamental algorithmic redesign required")

        if exponent > 2.0:
            recommendations.append("High-order power-law scaling may limit practical applications; it is not an exponential fit")

        if scaling_result.r_squared and scaling_result.r_squared < 0.8:
            recommendations.append("Low R² indicates complex relationship - consider additional parameters")

        return "\n".join(f"- {rec}" for rec in recommendations)
