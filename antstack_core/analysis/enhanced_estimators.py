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

from .energy import EnergyCoefficients, ComputeLoad, estimate_detailed_energy
from .statistics import bootstrap_mean_ci, analyze_scaling_relationship
from .workloads import (
    calculate_contact_complexity,
    calculate_sparse_neural_complexity,
    calculate_active_inference_complexity
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


@dataclass
class ComprehensiveEnergyAnalysis:
    """Comprehensive energy analysis results for all modules."""

    body_analysis: ModuleScalingData
    brain_analysis: ModuleScalingData
    mind_analysis: ModuleScalingData

    system_efficiency: float = 0.0
    total_energy_per_decision_j: float = 0.0
    cost_of_transport: float = 0.0

    key_numbers: Dict[str, Any] = field(default_factory=dict)


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

        for j in j_values:
            params = base_params.copy()
            params['J'] = j

            # Use enhanced closed-form model for consistent analysis
            load = enhanced_body_workload_closed_form(0.01, params)  # 10ms decision
            energy_breakdown = estimate_detailed_energy(load, self.coefficients)
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(j_values, energy_values)

        return ModuleScalingData(
            module_name="body",
            parameter_name="J",
            parameter_values=j_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("regime")
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
            load = enhanced_brain_workload_closed_form(0.01, params)  # 10ms decision
            energy_breakdown = estimate_detailed_energy(load, self.coefficients)
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(k_values, energy_values)

        return ModuleScalingData(
            module_name="brain",
            parameter_name="K",
            parameter_values=k_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("regime")
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
            load = enhanced_mind_workload_closed_form(0.01, params)  # 10ms decision
            energy_breakdown = estimate_detailed_energy(load, self.coefficients)
            energy = energy_breakdown.total
            energy_values.append(energy)
            flops_values.append(load.flops)
            memory_values.append(load.sram_bytes + load.dram_bytes)

        # Perform scaling analysis
        scaling_result = analyze_scaling_relationship(h_p_values, energy_values)

        return ModuleScalingData(
            module_name="mind",
            parameter_name="H_p",
            parameter_values=h_p_values,
            energy_values=energy_values,
            flops_values=flops_values,
            memory_values=memory_values,
            scaling_exponent=scaling_result.get("scaling_exponent"),
            r_squared=scaling_result.get("r_squared"),
            scaling_regime=scaling_result.get("regime")
        )

    def calculate_theoretical_limits(self, module_data: ModuleScalingData) -> Dict[str, float]:
        """Calculate theoretical energy limits for module.

        Args:
            module_data: Module scaling data

        Returns:
            Dictionary with theoretical limit calculations
        """
        from .statistics import estimate_theoretical_limits

        # Estimate typical module parameters for theoretical limit calculation
        if module_data.module_name == "body":
            module_params = {
                'flops': statistics.mean(module_data.flops_values),
                'bits_processed': statistics.mean(module_data.flops_values) * 64,
                'mechanical_work_j': 0.00036  # Typical actuation energy
            }
        elif module_data.module_name == "brain":
            module_params = {
                'flops': statistics.mean(module_data.flops_values),
                'bits_processed': statistics.mean(module_data.flops_values) * 32,
                'mechanical_work_j': 0.0
            }
        else:  # mind
            module_params = {
                'flops': statistics.mean(module_data.flops_values),
                'bits_processed': statistics.mean(module_data.flops_values) * 64,
                'mechanical_work_j': 0.0
            }

        return estimate_theoretical_limits(module_params)

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
        body_analysis.efficiency_ratio = (statistics.mean(body_analysis.energy_values) /
                                        body_limits.get('total_theoretical_j', 1.0))
        brain_analysis.efficiency_ratio = (statistics.mean(brain_analysis.energy_values) /
                                         brain_limits.get('total_theoretical_j', 1.0))
        mind_analysis.efficiency_ratio = (statistics.mean(mind_analysis.energy_values) /
                                        mind_limits.get('total_theoretical_j', 1.0))

        # Store theoretical limits
        body_analysis.theoretical_limit_j = body_limits.get('total_theoretical_j')
        brain_analysis.theoretical_limit_j = brain_limits.get('total_theoretical_j')
        mind_analysis.theoretical_limit_j = mind_limits.get('total_theoretical_j')

        # Calculate system-level metrics
        total_energy = (statistics.mean(body_analysis.energy_values) +
                       statistics.mean(brain_analysis.energy_values) +
                       statistics.mean(mind_analysis.energy_values))

        # Cost of transport (assuming typical mass and distance)
        mass_kg = 0.02
        distance_m = 1.0
        gravity = 9.81
        cost_of_transport = total_energy / (mass_kg * gravity * distance_m)

        # System efficiency (harmonic mean of module efficiencies)
        efficiencies = [body_analysis.efficiency_ratio or 1.0,
                       brain_analysis.efficiency_ratio or 1.0,
                       mind_analysis.efficiency_ratio or 1.0]
        system_efficiency = len(efficiencies) / sum(1/e for e in efficiencies if e > 0)

        # Generate key numbers dictionary
        key_numbers = self._generate_key_numbers_dict(
            body_analysis, brain_analysis, mind_analysis,
            total_energy, cost_of_transport
        )

        return ComprehensiveEnergyAnalysis(
            body_analysis=body_analysis,
            brain_analysis=brain_analysis,
            mind_analysis=mind_analysis,
            system_efficiency=system_efficiency,
            total_energy_per_decision_j=total_energy,
            cost_of_transport=cost_of_transport,
            key_numbers=key_numbers
        )

    def _generate_key_numbers_dict(self,
                                 body_analysis: ModuleScalingData,
                                 brain_analysis: ModuleScalingData,
                                 mind_analysis: ModuleScalingData,
                                 total_energy: float,
                                 cost_of_transport: float) -> Dict[str, Any]:
        """Generate comprehensive key numbers dictionary.

        Args:
            body_analysis: Body scaling analysis results
            brain_analysis: Brain scaling analysis results
            mind_analysis: Mind scaling analysis results
            total_energy: Total energy per decision
            cost_of_transport: Cost of transport metric

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
                "body_energy": body_analysis.scaling_exponent or 0,
                "body_r_squared": body_analysis.r_squared or 0,
                "body_regime": body_analysis.scaling_regime or "unknown",
                "brain_energy": brain_analysis.scaling_exponent or 0,
                "brain_r_squared": brain_analysis.r_squared or 0,
                "brain_regime": brain_analysis.scaling_regime or "unknown",
                "mind_energy": mind_analysis.scaling_exponent or 0,
                "mind_r_squared": mind_analysis.r_squared or 0,
                "mind_regime": mind_analysis.scaling_regime or "unknown"
            },
            "system_parameters": {
                "control_frequency_hz": 100,
                "decision_period_ms": 10,
                "baseline_power_mw": self.coefficients.baseline_w * 1000,
                "total_power_w": total_energy * 100,
                "cost_of_transport": cost_of_transport,
                "system_efficiency": (body_analysis.efficiency_ratio or 0) * 100
            },
            "theoretical_limits": {
                "body_efficiency_ratio": body_analysis.efficiency_ratio,
                "brain_efficiency_ratio": brain_analysis.efficiency_ratio,
                "mind_efficiency_ratio": mind_analysis.efficiency_ratio,
                "body_theoretical_j": body_analysis.theoretical_limit_j,
                "brain_theoretical_j": brain_analysis.theoretical_limit_j,
                "mind_theoretical_j": mind_analysis.theoretical_limit_j
            }
        }
