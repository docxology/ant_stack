"""
Scaling Analysis Module

Provides comprehensive scaling analysis capabilities for Ant Stack modules,
including power law detection, regime classification, and multi-parameter analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import math
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

from .statistics import analyze_scaling_relationship


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


@dataclass
class MultiParameterScaling:
    """Multi-parameter scaling analysis results."""

    primary_parameter: str
    secondary_parameters: Dict[str, List[float]]
    scaling_results: Dict[str, ScalingResult]

    interaction_effects: Dict[str, float] = None


class ScalingAnalyzer:
    """Advanced scaling analyzer for Ant Stack modules.

    Provides comprehensive scaling analysis including:
    - Power law detection with statistical validation
    - Multi-parameter scaling analysis
    - Regime classification and confidence intervals
    - Interaction effect detection
    """

    def __init__(self):
        """Initialize scaling analyzer."""
        pass

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
        if len(parameter_values) < 3 or len(response_values) < 3:
            return ScalingResult(
                parameter_name=parameter_name,
                parameter_values=parameter_values,
                response_values=response_values,
                valid=False
            )

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(parameter_values, response_values)

        # Calculate confidence interval for scaling exponent
        confidence_interval = None
        p_value = None

        # Check if scaling analysis was successful
        analysis_valid = "scaling_exponent" in scaling_result and "intercept" in scaling_result

        if analysis_valid:
            try:
                # Calculate confidence interval using bootstrap
                if HAS_NUMPY and HAS_SCIPY:
                    exponents = []
                    n_bootstrap = 1000

                    for _ in range(n_bootstrap):
                        # Bootstrap sample
                        indices = np.random.choice(len(parameter_values), len(parameter_values), replace=True)
                        boot_x = [parameter_values[i] for i in indices]
                        boot_y = [response_values[i] for i in indices]

                        # Calculate scaling for bootstrap sample
                        boot_result = analyze_scaling_relationship(boot_x, boot_y)
                        if "scaling_exponent" in boot_result:
                            exponents.append(boot_result.get("scaling_exponent", 0))

                    if exponents:
                        try:
                            ci_low = np.percentile(exponents, 2.5)
                            ci_high = np.percentile(exponents, 97.5)
                            confidence_interval = (ci_low, ci_high)

                            # Calculate p-value (test against null hypothesis of no scaling)
                            t_stat = scaling_result.get("scaling_exponent", 0) / (np.std(exponents) or 1)
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(exponents) - 1))
                        except (ValueError, TypeError):
                            # Fallback if numpy percentile fails
                            exponents.sort()
                            n_exp = len(exponents)
                            ci_low_idx = max(0, int(0.025 * n_exp))
                            ci_high_idx = min(n_exp - 1, int(0.975 * n_exp))
                            confidence_interval = (exponents[ci_low_idx], exponents[ci_high_idx])

            except Exception:
                pass  # Keep confidence_interval as None if calculation fails

        return ScalingResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            response_values=response_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            intercept=scaling_result.get("intercept"),
            r_squared=scaling_result.get("r_squared"),
            regime=scaling_result.get("scaling_regime"),
            confidence_interval=confidence_interval,
            p_value=p_value,
            valid=analysis_valid
        )

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

        # Calculate interaction effects (simplified)
        interaction_effects = {}
        if len(scaling_results) > 1 and HAS_NUMPY:
            for param_name, result in scaling_results.items():
                if result.valid and result.scaling_exponent:
                    # Simple interaction measure based on correlation
                    correlation = np.corrcoef(primary_values, result.parameter_values)[0, 1]
                    interaction_effects[param_name] = abs(correlation) * result.scaling_exponent

        return MultiParameterScaling(
            primary_parameter=primary_param,
            secondary_parameters=secondary_params,
            scaling_results=scaling_results,
            interaction_effects=interaction_effects
        )

    def detect_scaling_regime(self, scaling_result: ScalingResult) -> Dict[str, Any]:
        """Detect and characterize scaling regime.

        Args:
            scaling_result: ScalingResult to analyze

        Returns:
            Dictionary with regime characterization
        """
        if not scaling_result.valid or scaling_result.scaling_exponent is None:
            return {"regime": "invalid", "confidence": 0.0}

        exponent = scaling_result.scaling_exponent

        # Classify regime
        if abs(exponent) < 0.1:
            regime = "constant"
            confidence = 0.9
        elif abs(exponent - 1.0) < 0.1:
            regime = "linear"
            confidence = 0.9
        elif abs(exponent - 2.0) < 0.1:
            regime = "quadratic"
            confidence = 0.8
        elif abs(exponent - 3.0) < 0.1:
            regime = "cubic"
            confidence = 0.7
        elif exponent > 1.0:
            regime = "super-linear"
            confidence = 0.8
        elif exponent < 1.0 and exponent > 0:
            regime = "sub-linear"
            confidence = 0.8
        else:
            regime = "unknown"
            confidence = 0.5

        # Adjust confidence based on r-squared
        if scaling_result.r_squared:
            r2_confidence = scaling_result.r_squared * confidence
        else:
            r2_confidence = confidence

        return {
            "regime": regime,
            "confidence": r2_confidence,
            "exponent_range": (exponent - 0.1, exponent + 0.1),
            "interpretation": self._interpret_regime(regime, exponent)
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
            "constant": ".3f",
            "linear": ".3f",
            "quadratic": ".3f",
            "cubic": ".3f",
            "sub-linear": ".3f",
            "super-linear": ".3f"
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

        # Extract valid results
        valid_results = {k: v for k, v in scaling_results.items() if v.valid}

        if not valid_results:
            return {"error": "No valid scaling results"}

        # Compare exponents
        exponents = [r.scaling_exponent for r in valid_results.values() if r.scaling_exponent]
        r_squared_values = [r.r_squared for r in valid_results.values() if r.r_squared]

        comparison = {
            "num_valid_results": len(valid_results),
            "exponent_range": (min(exponents), max(exponents)) if exponents else None,
            "mean_r_squared": None,
            "regime_distribution": {}
        }

        # Calculate mean r_squared
        if r_squared_values and HAS_NUMPY:
            comparison["mean_r_squared"] = np.mean(r_squared_values)

        # Count regime distribution
        for result in valid_results.values():
            regime = result.regime or "unknown"
            comparison["regime_distribution"][regime] = \
                comparison["regime_distribution"].get(regime, 0) + 1

        # Identify most challenging scaling
        if exponents:
            if HAS_NUMPY:
                max_exponent_idx = np.argmax(np.abs(exponents))
            else:
                # Fallback without numpy
                max_exponent_idx = 0
                max_abs_exp = 0
                for i, exp in enumerate(exponents):
                    if abs(exp) > max_abs_exp:
                        max_abs_exp = abs(exp)
                        max_exponent_idx = i

            max_exponent_param = list(valid_results.keys())[max_exponent_idx]
            comparison["most_challenging"] = {
                "parameter": max_exponent_param,
                "exponent": exponents[max_exponent_idx],
                "regime": list(valid_results.values())[max_exponent_idx].regime
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
            return f"Scaling analysis for {scaling_result.parameter_name}: INVALID (insufficient data)"

        report_lines = [
            f"Scaling Analysis Report: {scaling_result.parameter_name}",
            "=" * 60,
            f"Parameter: {scaling_result.parameter_name}",
            f"Data points: {len(scaling_result.parameter_values)}",
            f"Exponent: {scaling_result.scaling_exponent:.6f}",
            f"R²: {scaling_result.r_squared:.6f}",
            f"Intercept: {scaling_result.intercept:.6f}",
            f"Regime: {scaling_result.regime}",
        ]

        if scaling_result.confidence_interval:
            ci_low, ci_high = scaling_result.confidence_interval
            report_lines.append(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
        if scaling_result.p_value:
            report_lines.append(f"P-value: {scaling_result.p_value:.6f}")
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
        if not scaling_result.valid or not scaling_result.scaling_exponent:
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
            recommendations.append("WARNING: Exponential scaling detected - may limit practical applications")

        if scaling_result.r_squared and scaling_result.r_squared < 0.8:
            recommendations.append("Low R² indicates complex relationship - consider additional parameters")

        return "\n".join(f"- {rec}" for rec in recommendations)
