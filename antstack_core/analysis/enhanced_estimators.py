"""
Enhanced Energy Estimation Module

Provides comprehensive energy estimation with detailed breakdowns,
scaling analysis, and theoretical limit comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import statistics

from .energy import EnergyCoefficients, ComputeLoad, estimate_detailed_energy, _require_nonnegative, _require_positive
from .statistics import bootstrap_mean_ci, analyze_scaling_relationship
from .workloads import (
    calculate_contact_complexity,
    calculate_sparse_neural_complexity,
    calculate_active_inference_complexity
)

_AGGREGATION_CONTEXT = (
    "Analytical sum of isolated module sweep means, including idle energy once per module "
    "and the body's modeled mechanical work; not whole-device energy or an empirical measurement"
)



@dataclass
class ModuleScalingData:
    """Container for module-specific scaling analysis results."""

    module_name: str
    parameter_name: str
    parameter_values: List[float]
    energy_values: List[float]
    flops_values: List[float]
    memory_values: List[float]

    scaling_exponent: Optional[float] = None
    r_squared: Optional[float] = None
    scaling_regime: Optional[str] = None

    efficiency_ratio: Optional[float] = None
    theoretical_limit_j: Optional[float] = None
    mechanical_work_values: List[float] = field(default_factory=list)
    duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        count = len(self.parameter_values)
        for name in ('energy_values', 'flops_values', 'memory_values', 'mechanical_work_values'):
            values = getattr(self, name)
            if name != 'mechanical_work_values' or values:
                if len(values) != count:
                    raise ValueError(f"{name} must be paired with parameter_values")
            for value in values:
                _require_nonnegative(name, value)
        if self.duration_s is not None:
            _require_positive("duration_s", self.duration_s)


@dataclass
class ComprehensiveEnergyAnalysis:
    """Comprehensive energy analysis results for all modules."""

    body_analysis: ModuleScalingData
    brain_analysis: ModuleScalingData
    mind_analysis: ModuleScalingData

    system_efficiency: Optional[float] = None  # Sum of matched bounds / sum of modeled energies.
    total_energy_per_decision_j: float = 0.0
    cost_of_transport: Optional[float] = None

    key_numbers: Dict[str, Any] = field(default_factory=dict)
    aggregation_context: str = _AGGREGATION_CONTEXT


class EnhancedEnergyEstimator:
    """Enhanced energy estimator with comprehensive analysis capabilities.

    Provides detailed energy estimation, scaling analysis, and theoretical
    limit comparisons for all Ant Stack modules.
    """

    def __init__(self, coefficients: EnergyCoefficients):
        """Initialize estimator with energy coefficients.

        Args:
            coefficients: EnergyCoefficients instance with device parameters
        """
        self.coefficients = coefficients

    def analyze_body_scaling(self, j_values: List[int],
                           base_params: Dict[str, Any]) -> ModuleScalingData:
        """Analyze AntBody energy scaling across joint counts.

        Args:
            j_values: List of joint counts to analyze
            base_params: Base parameters for body analysis

        Returns:
            ModuleScalingData with scaling analysis results
        """
        from .workloads import enhanced_body_workload_closed_form

        energy_values = []
        flops_values = []
        memory_values = []

        mechanical_work_values = []
        for j in j_values:
            params = base_params.copy()
            params['J'] = j

            # Use enhanced closed-form model for consistent analysis
            dt = float(params.get("dt", 0.01))
            load = enhanced_body_workload_closed_form(dt, params)
            velocity = float(params.get("v", 1.0))
            mass = float(params.get("m", 0.001))
            length = float(params.get("L", 0.01))
            gravity = float(params.get("g", 9.81))
            for name, value in (("v", velocity), ("m", mass), ("L", length), ("g", gravity)):
                _require_nonnegative(name, value)
            # Ideal per-decision lift plus acceleration proxy, in joules:
            # J joints * m kg/joint * (g m/s^2 * L m + v^2 m^2/s^2 / 2).
            # Assumes each joint lifts L and accelerates from rest to v once;
            # no regeneration or calibrated biomechanical efficiency is implied.
            actuation_energy = j * mass * (gravity * length + 0.5 * velocity ** 2)
            mechanical_work_values.append(actuation_energy)
            energy_breakdown = estimate_detailed_energy(
                load, self.coefficients, duration_s=dt, actuation_energy=actuation_energy
            )
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(j_values, energy_values)

        return ModuleScalingData(
            module_name="AntBody",
            parameter_name="joints",
            parameter_values=j_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("scaling_regime"),
            mechanical_work_values=mechanical_work_values,
            duration_s=float(base_params.get("dt", 0.01)),
        )

    def analyze_brain_scaling(self, k_values: List[int],
                            base_params: Dict[str, Any]) -> ModuleScalingData:
        """Analyze AntBrain energy scaling across sensory channels.

        Args:
            k_values: List of AL input channel counts to analyze
            base_params: Base parameters for brain analysis

        Returns:
            ModuleScalingData with scaling analysis results
        """
        from .workloads import enhanced_brain_workload_closed_form

        energy_values = []
        flops_values = []
        memory_values = []

        for k in k_values:
            params = base_params.copy()
            params['K'] = k

            # Use enhanced closed-form model
            dt = float(params.get("dt", 0.01))
            load = enhanced_brain_workload_closed_form(dt, params)
            energy_breakdown = estimate_detailed_energy(load, self.coefficients, duration_s=dt)
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(k_values, energy_values)

        return ModuleScalingData(
            module_name="AntBrain",
            parameter_name="channels",
            parameter_values=k_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("scaling_regime"),
            duration_s=float(base_params.get("dt", 0.01)),
        )

    def analyze_mind_scaling(self, h_p_values: List[int],
                           base_params: Dict[str, Any]) -> ModuleScalingData:
        """Analyze AntMind energy scaling across policy horizons.

        Args:
            h_p_values: List of policy horizon values to analyze
            base_params: Base parameters for mind analysis

        Returns:
            ModuleScalingData with scaling analysis results
        """
        from .workloads import enhanced_mind_workload_closed_form

        energy_values = []
        flops_values = []
        memory_values = []

        for h_p in h_p_values:
            params = base_params.copy()
            params['H_p'] = h_p

            # Use enhanced closed-form model
            dt = float(params.get("dt", 0.01))
            load = enhanced_mind_workload_closed_form(dt, params)
            energy_breakdown = estimate_detailed_energy(load, self.coefficients, duration_s=dt)
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(h_p_values, energy_values)

        return ModuleScalingData(
            module_name="AntMind",
            parameter_name="horizon",
            parameter_values=h_p_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("scaling_regime"),
            duration_s=float(base_params.get("dt", 0.01)),
        )

    def calculate_theoretical_limits(self, module_data: ModuleScalingData) -> Dict[str, Any]:
        """Calculate theoretical energy limits for module.

        Args:
            module_data: Module scaling data

        Returns:
            Dictionary with theoretical limit calculations
        """
        from .statistics import estimate_theoretical_limits

        # A conditional scenario, not a universal FLOP-to-erasure conversion:
        # one bit is irreversibly erased per modeled FLOP at 300 K.
        if not module_data.flops_values:
            raise ValueError("Theoretical comparison requires workload observations")
        module_params = {
            'bits_erased': statistics.mean(module_data.flops_values),
            'mechanical_work_j': statistics.mean(module_data.mechanical_work_values) if module_data.mechanical_work_values else 0.0,
        }
        limits = estimate_theoretical_limits(module_params)
        limits['erasure_assumption'] = 'One erased bit per modeled FLOP at 300 K'
        limits['total_bound_accounting'] = 'Sum of separate mechanical work and digital erasure costs; no overlap assumed'
        return limits

    def calculate_efficiency_metrics(
        self,
        body_data: ModuleScalingData,
        brain_data: ModuleScalingData,
        mind_data: ModuleScalingData,
    ) -> Dict[str, Any]:
        """Calculate system efficiency metrics across module scaling results."""
        means = {
            "body": statistics.mean(body_data.energy_values) if body_data.energy_values else 0.0,
            "brain": statistics.mean(brain_data.energy_values) if brain_data.energy_values else 0.0,
            "mind": statistics.mean(mind_data.energy_values) if mind_data.energy_values else 0.0,
        }
        total = sum(means.values())
        modules = (body_data, brain_data, mind_data)
        durations = {module.duration_s for module in modules}
        if len(durations) != 1 or None in durations:
            raise ValueError("System accounting requires a common known decision duration")
        common_duration = next(iter(durations))
        _require_positive("decision duration", common_duration)
        total_bound = sum(self.calculate_theoretical_limits(module)['total_theoretical_j'] for module in modules)
        fraction = total_bound / total if total > 0 else None
        distribution = {name: (value / total if total > 0 else 0.0) for name, value in means.items()}
        bottlenecks = [name for name, share in distribution.items() if share == max(distribution.values(), default=0.0)]
        return {
            "system_efficiency": fraction,
            "valid": fraction is not None and fraction <= 1,
            "aggregation": _AGGREGATION_CONTEXT,
            "energy_distribution": distribution,
            "bottlenecks": bottlenecks,
        }

    def compare_with_theoretical_limits(self, module_data: ModuleScalingData) -> Dict[str, Any]:
        """Compare empirical module energy against simple theoretical limits."""
        limits = self.calculate_theoretical_limits(module_data)
        actual = statistics.mean(module_data.energy_values) if module_data.energy_values else 0.0
        landauer = limits['landauer_limit_j']
        thermodynamic = limits['mechanical_work_j']
        return {
            "landauer_efficiency": landauer / actual if actual > 0 else None,
            "thermodynamic_efficiency": thermodynamic / actual if actual > 0 else None,
            "landauer_overhead_ratio": actual / landauer if landauer > 0 else None,
            "comparison_scope": "Component bounds divided by total modeled energy; not component conversion efficiencies",
            "fundamental_limits": limits,
        }

    def generate_scaling_report(self, analysis: ComprehensiveEnergyAnalysis) -> str:
        """Generate a concise text report for comprehensive scaling analysis."""
        lines = [
            "Ant Stack Energy Scaling Report",
            "================================",
        ]
        for module in [analysis.body_analysis, analysis.brain_analysis, analysis.mind_analysis]:
            lines.extend([
                f"{module.module_name} scaling",
                f"- parameter: {module.parameter_name}",
                f"- samples: {len(module.parameter_values)}",
                f"- exponent: {module.scaling_exponent if module.scaling_exponent is not None else 'n/a'}",
                f"- regime: {module.scaling_regime or 'unknown'}",
            ])
        lines.extend([
            f"Modeled aggregate bound/energy fraction: {analysis.system_efficiency if analysis.system_efficiency is not None else 'undefined'}",
            f"Sum of isolated module mean energies per decision (J): {analysis.total_energy_per_decision_j:.6g}",
            analysis.aggregation_context,
        ])
        return "\n".join(lines)

    def perform_comprehensive_analysis(self,
                                     body_params: Dict[str, Any],
                                     brain_params: Dict[str, Any],
                                     mind_params: Dict[str, Any],
                                     j_values: Optional[List[int]] = None,
                                     k_values: Optional[List[int]] = None,
                                     h_p_values: Optional[List[int]] = None) -> ComprehensiveEnergyAnalysis:
        """Perform comprehensive energy analysis across all modules.

        Args:
            body_params: Base parameters for body analysis
            brain_params: Base parameters for brain analysis
            mind_params: Base parameters for mind analysis
            j_values: Joint count values for scaling (default: [6, 12, 18, 24, 30])
            k_values: AL channel values for scaling (default: [64, 128, 256, 512])
            h_p_values: Policy horizon values for scaling (default: [5, 10, 15, 20])

        Returns:
            ComprehensiveEnergyAnalysis with all results
        """
        # Set defaults
        j_values = j_values or [6, 12, 18, 24, 30]
        k_values = k_values or [64, 128, 256, 512, 1024]
        h_p_values = h_p_values or [5, 10, 15, 20]

        # Analyze each module
        body_analysis = self.analyze_body_scaling(j_values, body_params)
        brain_analysis = self.analyze_brain_scaling(k_values, brain_params)
        mind_analysis = self.analyze_mind_scaling(h_p_values, mind_params)

        # Calculate theoretical limits
        body_limits = self.calculate_theoretical_limits(body_analysis)
        brain_limits = self.calculate_theoretical_limits(brain_analysis)
        mind_limits = self.calculate_theoretical_limits(mind_analysis)

        # Calculate efficiency ratios
        for module, limits in ((body_analysis, body_limits), (brain_analysis, brain_limits), (mind_analysis, mind_limits)):
            bound = limits['total_theoretical_j']
            module.efficiency_ratio = statistics.mean(module.energy_values) / bound if bound > 0 else None

        # Store theoretical limits
        body_analysis.theoretical_limit_j = body_limits.get('total_theoretical_j')
        brain_analysis.theoretical_limit_j = brain_limits.get('total_theoretical_j')
        mind_analysis.theoretical_limit_j = mind_limits.get('total_theoretical_j')

        # Check decision windows before summing isolated module sweep means.
        system_efficiency = self.calculate_efficiency_metrics(body_analysis, brain_analysis, mind_analysis)['system_efficiency']
        total_energy = (statistics.mean(body_analysis.energy_values) +
                       statistics.mean(brain_analysis.energy_values) +
                       statistics.mean(mind_analysis.energy_values))

        # Generate key numbers dictionary
        key_numbers = self._generate_key_numbers_dict(
            body_analysis, brain_analysis, mind_analysis,
            total_energy, system_efficiency,
            float(body_params.get('dt', 0.01)), float(body_params.get('hz', 100)),
        )

        return ComprehensiveEnergyAnalysis(
            body_analysis=body_analysis,
            brain_analysis=brain_analysis,
            mind_analysis=mind_analysis,
            system_efficiency=system_efficiency,
            total_energy_per_decision_j=total_energy,
            key_numbers=key_numbers
        )

    def _generate_key_numbers_dict(self,
                                 body_analysis: ModuleScalingData,
                                 brain_analysis: ModuleScalingData,
                                 mind_analysis: ModuleScalingData,
                                 total_energy: float,
                                 system_efficiency: Optional[float],
                                 duration_s: float,
                                 control_frequency_hz: float) -> Dict[str, Any]:
        """Generate comprehensive key numbers dictionary.

        Args:
            body_analysis: Body scaling analysis results
            brain_analysis: Brain scaling analysis results
            mind_analysis: Mind scaling analysis results
            total_energy: Sum of isolated module mean energies per decision

        Returns:
            Dictionary with all key numbers
        """
        return {
            "per_decision_energy": {
                "body_mj": statistics.mean(body_analysis.energy_values) * 1000,
                "brain_mj": statistics.mean(brain_analysis.energy_values) * 1000,
                "mind_mj": statistics.mean(mind_analysis.energy_values) * 1000,
                "total_mj": total_energy * 1000
            },
            "computational_load": {
                "body_flops": statistics.mean(body_analysis.flops_values),
                "brain_flops": statistics.mean(brain_analysis.flops_values),
                "mind_flops": statistics.mean(mind_analysis.flops_values),
                "body_memory_kb": statistics.mean(body_analysis.memory_values) / 1024,
                "brain_memory_kb": statistics.mean(brain_analysis.memory_values) / 1024,
                "mind_memory_kb": statistics.mean(mind_analysis.memory_values) / 1024
            },
            "scaling_exponents": {
                "body_energy": body_analysis.scaling_exponent,
                "body_r_squared": body_analysis.r_squared,
                "body_regime": body_analysis.scaling_regime or "unknown",
                "brain_energy": brain_analysis.scaling_exponent,
                "brain_r_squared": brain_analysis.r_squared,
                "brain_regime": brain_analysis.scaling_regime or "unknown",
                "mind_energy": mind_analysis.scaling_exponent,
                "mind_r_squared": mind_analysis.r_squared,
                "mind_regime": mind_analysis.scaling_regime or "unknown"
            },
            "system_parameters": {
                "control_frequency_hz": control_frequency_hz,
                "decision_period_ms": duration_s * 1000,
                "baseline_power_mw": self.coefficients.baseline_w * 1000,
                "total_power_w": total_energy / duration_s,
                "cost_of_transport": None,
                "total_power_boundary": _AGGREGATION_CONTEXT,
                "system_efficiency": system_efficiency * 100 if system_efficiency is not None else None
            },
            "theoretical_limits": {
                "body_efficiency_ratio": body_analysis.efficiency_ratio,
                "brain_efficiency_ratio": brain_analysis.efficiency_ratio,
                "mind_efficiency_ratio": mind_analysis.efficiency_ratio,
                "body_theoretical_j": body_analysis.theoretical_limit_j,
                "brain_theoretical_j": brain_analysis.theoretical_limit_j,
                "mind_theoretical_j": mind_analysis.theoretical_limit_j
            },
            "aggregation_context": _AGGREGATION_CONTEXT,
        }
