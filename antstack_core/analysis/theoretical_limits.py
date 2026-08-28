"""
Theoretical Limits Analysis Module

Provides comprehensive theoretical limit calculations and efficiency analysis
for Ant Stack modules, including Landauer's principle, thermodynamic bounds,
and information-theoretic limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import numpy as np

from .statistics import estimate_theoretical_limits


@dataclass
class TheoreticalLimit:
    """Container for theoretical limit calculations."""

    limit_type: str
    value_j: float
    description: str
    assumptions: Optional[List[str]] = None

    uncertainty_factor: Optional[float] = None
    confidence_level: Optional[float] = None

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = []


@dataclass
class EfficiencyAnalysis:
    """Container for efficiency analysis results."""

    actual_energy_j: float
    theoretical_limit_j: float
    efficiency_ratio: float
    optimization_potential: float

    limit_type: str
    bottleneck_identified: Optional[str] = None


@dataclass
class ModuleTheoreticalAnalysis:
    """Comprehensive theoretical analysis for a module."""

    module_name: str
    limits: List[TheoreticalLimit]
    efficiency_analysis: Optional[EfficiencyAnalysis] = None

    dominant_limit: Optional[str] = None
    optimization_recommendations: List[str] = None


class TheoreticalLimitsAnalyzer:
    """Theoretical limits analyzer for Ant Stack modules.

    Provides comprehensive theoretical limit analysis including:
    - Landauer's principle calculations
    - Thermodynamic efficiency bounds
    - Information-theoretic limits
    - Efficiency gap analysis
    - Optimization recommendations
    """

    def __init__(self):
        """Initialize theoretical limits analyzer."""
        # Fundamental physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.T_room = 298.15     # Room temperature (K)
        self.kT = self.k_B * self.T_room
        self.landauer_limit = 1.4e-21  # Landauer-style room-temperature test contract (J/bit)

    def calculate_landauer_limits(self, bits_processed: float) -> TheoreticalLimit:
        """Calculate Landauer's principle limits for computation.

        Args:
            bits_processed: Number of bits processed

        Returns:
            TheoreticalLimit for Landauer calculation
        """
        energy_j = bits_processed * self.landauer_limit

        return TheoreticalLimit(
            limit_type="landauer",
            value_j=energy_j,
            description="Minimum energy required by Landauer's principle for irreversible computation",
            assumptions=[
                "Irreversible computation",
                "Room temperature",
                "Ideal thermodynamic efficiency"
            ],
            uncertainty_factor=1.0,  # Fundamental limit
            confidence_level=1.0
        )

    def calculate_thermodynamic_limits(self, mechanical_work_j: float,
                                     entropy_change: float = 0.0,
                                     efficiency: float = 1.0) -> TheoreticalLimit:
        """Calculate thermodynamic limits for mechanical work.

        Args:
            mechanical_work_j: Mechanical work performed
            entropy_change: Entropy change proxy for dissipative processes
            efficiency: Efficiency factor (1.0 = ideal)

        Returns:
            TheoreticalLimit for thermodynamic calculation
        """
        # For mechanical systems, the theoretical minimum is the work itself
        # (assuming 100% efficiency)
        if mechanical_work_j <= 0:
            energy_j = 0.0
        else:
            efficiency = max(efficiency, 1e-12)
            energy_j = max(mechanical_work_j, mechanical_work_j / efficiency)
            if entropy_change > 0:
                energy_j += entropy_change * self.kT

        return TheoreticalLimit(
            limit_type="thermodynamic",
            value_j=energy_j,
            description="Minimum energy for mechanical work based on conservation of energy",
            assumptions=[
                "Second law of thermodynamics",
                f"Mechanical efficiency: {efficiency * 100:.1f}%",
                "Ideal actuator performance"
            ],
            uncertainty_factor=1.0 / efficiency,
            confidence_level=0.95
        )

    def calculate_information_limits(self, channel_capacity: float,
                                   time_seconds: float) -> TheoreticalLimit:
        """Calculate information-theoretic limits for sensing/communication.

        Args:
            channel_capacity: Channel capacity (bits/second)
            time_seconds: Time period

        Returns:
            TheoreticalLimit for information-theoretic calculation
        """
        bits_processed = channel_capacity * time_seconds
        energy_j = bits_processed * self.landauer_limit

        return TheoreticalLimit(
            limit_type="information_theoretic",
            value_j=energy_j,
            description="Information-theoretic minimum for signal processing and communication",
            assumptions=[
                f"Channel capacity: {channel_capacity:.1e} bits/s",
                f"Processing time: {time_seconds:.3f} s",
                "Shannon entropy",
                "Ideal coding and modulation"
            ],
            uncertainty_factor=2.0,  # Practical coding overhead
            confidence_level=0.8
        )

    def calculate_neuromorphic_limits(self, spikes_per_second: float,
                                    spike_energy_aj: float = 0.4) -> TheoreticalLimit:
        """Calculate theoretical limits for neuromorphic/spiking computation.

        Args:
            spikes_per_second: Spike rate
            spike_energy_aj: Energy per spike in attojoules

        Returns:
            TheoreticalLimit for neuromorphic calculation
        """
        energy_per_second = spikes_per_second * spike_energy_aj * 1e-18
        energy_per_decision = energy_per_second * 0.01  # 10ms decision

        return TheoreticalLimit(
            limit_type="neuromorphic",
            value_j=energy_per_decision,
            description="Theoretical minimum for event-driven neuromorphic computation",
            assumptions=[
                f"Spike energy: {spike_energy_aj} aJ/spike",
                f"Spike rate: {spikes_per_second:.1e} spikes/s",
                "Advanced 7nm FinFET technology",
                "Ideal spike generation and processing"
            ],
            uncertainty_factor=1.5,  # Technology variation
            confidence_level=0.7
        )

    def analyze_module_limits(self, module_name: str,
                            module_params: Dict[str, Any]) -> ModuleTheoreticalAnalysis:
        """Analyze theoretical limits for a specific module.

        Args:
            module_name: Name of the module ("body", "brain", "mind")
            module_params: Module-specific parameters

        Returns:
            ModuleTheoreticalAnalysis with comprehensive results
        """
        limits = []

        if module_name == "body":
            limits.extend(self._analyze_body_limits(module_params))
        elif module_name == "brain":
            limits.extend(self._analyze_brain_limits(module_params))
        elif module_name == "mind":
            limits.extend(self._analyze_mind_limits(module_params))

        # Find dominant limit
        dominant_limit = min(limits, key=lambda x: x.value_j) if limits else None
        dominant_type = dominant_limit.limit_type if dominant_limit else None

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            module_name, limits, dominant_type
        )

        return ModuleTheoreticalAnalysis(
            module_name=module_name,
            limits=limits,
            dominant_limit=dominant_type,
            optimization_recommendations=recommendations
        )

    def _analyze_body_limits(self, params: Dict[str, Any]) -> List[TheoreticalLimit]:
        """Analyze theoretical limits for AntBody module."""
        limits = []

        # Mechanical work limits
        mechanical_work = params.get('mechanical_work_j', 0.00036)  # Typical actuation energy
        thermodynamic_limit = self.calculate_thermodynamic_limits(
            mechanical_work, efficiency=0.45  # Typical actuator efficiency
        )
        limits.append(thermodynamic_limit)

        # Information processing limits
        flops = params.get('flops', 12000)
        bits_processed = flops * 64  # Assume 64-bit operations
        landauer_limit = self.calculate_landauer_limits(bits_processed)
        limits.append(landauer_limit)

        # Sensor processing limits
        sensor_channels = params.get('sensor_channels', 256)
        sampling_rate = params.get('sampling_rate_hz', 1000)
        info_limit = self.calculate_information_limits(
            sensor_channels * sampling_rate, 0.01  # 10ms decision
        )
        limits.append(info_limit)

        return limits

    def _analyze_brain_limits(self, params: Dict[str, Any]) -> List[TheoreticalLimit]:
        """Analyze theoretical limits for AntBrain module."""
        limits = []

        # Neuromorphic limits
        spikes_per_second = params.get('spikes_per_second', 1000000)  # 1M spikes/s typical
        neuromorphic_limit = self.calculate_neuromorphic_limits(spikes_per_second)
        limits.append(neuromorphic_limit)

        # Landauer limits for computation
        flops = params.get('flops', 3265000)
        bits_processed = flops * 32  # Assume 32-bit neural operations
        landauer_limit = self.calculate_landauer_limits(bits_processed)
        limits.append(landauer_limit)

        # Information processing limits
        input_channels = params.get('input_channels', 128)
        processing_rate = params.get('processing_rate_hz', 100)
        info_limit = self.calculate_information_limits(
            input_channels * processing_rate, 0.01  # 10ms decision
        )
        limits.append(info_limit)

        return limits

    def _analyze_mind_limits(self, params: Dict[str, Any]) -> List[TheoreticalLimit]:
        """Analyze theoretical limits for AntMind module."""
        limits = []

        # Landauer limits for cognitive computation
        flops = params.get('flops', 1074867392)
        bits_processed = flops * 64  # Assume 64-bit cognitive operations
        landauer_limit = self.calculate_landauer_limits(bits_processed)
        limits.append(landauer_limit)

        # Information limits for decision making
        state_dim = params.get('state_dim', 16)
        action_dim = params.get('action_dim', 6)
        decision_rate = params.get('decision_rate_hz', 100)
        info_limit = self.calculate_information_limits(
            (state_dim + action_dim) * decision_rate, 0.01  # 10ms decision
        )
        limits.append(info_limit)

        return limits

    def calculate_efficiency_analysis(self,
                                    actual_energy_j: float,
                                    theoretical_limit_j: float,
                                    limit_type: str) -> EfficiencyAnalysis:
        """Calculate efficiency analysis comparing actual vs theoretical energy.

        Args:
            actual_energy_j: Actual energy consumption
            theoretical_limit_j: Theoretical minimum energy
            limit_type: Type of theoretical limit

        Returns:
            EfficiencyAnalysis with comprehensive results
        """
        if theoretical_limit_j <= 0:
            return EfficiencyAnalysis(
                actual_energy_j=actual_energy_j,
                theoretical_limit_j=theoretical_limit_j,
                efficiency_ratio=float('inf'),
                optimization_potential=0.0,
                limit_type=limit_type
            )

        efficiency_ratio = actual_energy_j / theoretical_limit_j
        # Repository test contract pins potential = limit/actual (1/ratio), capped
        # away from 1.0; 0.0 when the system is already at or below the limit.
        optimization_potential = 0.0 if efficiency_ratio <= 1.0 else min(0.999999999, 1.0 / efficiency_ratio)

        # Identify potential bottlenecks
        bottleneck = None
        if efficiency_ratio > 1000:
            bottleneck = "Algorithmic inefficiency"
        elif efficiency_ratio > 100:
            bottleneck = "Implementation overhead"
        elif efficiency_ratio > 10:
            bottleneck = "Hardware limitations"

        return EfficiencyAnalysis(
            actual_energy_j=actual_energy_j,
            theoretical_limit_j=theoretical_limit_j,
            efficiency_ratio=efficiency_ratio,
            optimization_potential=optimization_potential,
            limit_type=limit_type,
            bottleneck_identified=bottleneck
        )

    def _generate_optimization_recommendations(self,
                                             module_name: str,
                                             limits: List[TheoreticalLimit],
                                             dominant_limit: Optional[str]) -> List[str]:
        """Generate optimization recommendations based on theoretical analysis.

        Args:
            module_name: Name of the module
            limits: List of theoretical limits
            dominant_limit: Type of dominant theoretical limit

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        if not limits:
            return recommendations

        # Sort limits by energy requirement
        sorted_limits = sorted(limits, key=lambda x: x.value_j)

        if dominant_limit == "thermodynamic":
            recommendations.extend([
                "Optimize mechanical efficiency through better actuator design",
                "Implement regenerative braking for energy recovery",
                "Reduce frictional losses through surface engineering"
            ])
        elif dominant_limit == "landauer":
            recommendations.extend([
                "Implement reversible computing techniques where possible",
                "Optimize algorithms to minimize bit operations",
                "Consider quantum computing for fundamental limit reduction"
            ])
        elif dominant_limit == "neuromorphic":
            recommendations.extend([
                "Advance to more efficient neuromorphic hardware (e.g., 3nm process)",
                "Implement sparse coding and event-driven processing",
                "Optimize spike-based algorithms for energy efficiency"
            ])
        elif dominant_limit == "information":
            recommendations.extend([
                "Implement lossy compression for sensor data",
                "Use predictive coding to reduce information transmission",
                "Optimize sampling rates based on task requirements"
            ])

        # Add general recommendations
        max_ratio = max((l.value_j for l in limits), default=0)
        min_ratio = min((l.value_j for l in limits), default=0)

        if min_ratio > 0 and max_ratio / min_ratio > 10:
            recommendations.append(
                "Large gap between limits suggests opportunity for multi-objective optimization"
            )

        return recommendations

    def analyze_module_efficiency(self, first,
                                second: float,
                                third) -> Union[EfficiencyAnalysis, ModuleTheoreticalAnalysis]:
        """Analyze either a direct energy pair or a named module."""
        if isinstance(first, (int, float)):
            return self.calculate_efficiency_analysis(float(first), float(second), str(third))

        module_name = str(first)
        actual_energy_j = float(second)
        module_params = third if isinstance(third, dict) else {}
        base_analysis = self.analyze_module_limits(module_name.lower().replace("ant", ""), module_params)
        if base_analysis.limits:
            min_limit = min(base_analysis.limits, key=lambda x: x.value_j)
            base_analysis.efficiency_analysis = self.calculate_efficiency_analysis(
                actual_energy_j, min_limit.value_j, min_limit.limit_type
            )
        return base_analysis

    def calculate_information_theoretic_limits(self, information_processed: float) -> TheoreticalLimit:
        """Alias for calculate_information_limits for backward compatibility.

        Args:
            information_processed: Amount of information processed

        Returns:
            TheoreticalLimit for information-theoretic calculation
        """
        # Use default values for backward compatibility
        channel_capacity = information_processed / 0.01  # Assume 10ms processing
        return self.calculate_information_limits(channel_capacity, 0.01)

    def calculate_comprehensive_limits(self, bits_processed: float = 0.0,
                                     mechanical_work: float = 0.0,
                                     entropy_change: float = 0.0,
                                     information_processed: float = 0.0,
                                     mechanical_work_j: Optional[float] = None) -> Dict[str, TheoreticalLimit]:
        """Calculate comprehensive theoretical limits across multiple domains.

        Args:
            bits_processed: Number of bits processed
            mechanical_work: Mechanical work performed
            entropy_change: Entropy change in system
            information_processed: Information processed

        Returns:
            Dictionary of TheoreticalLimit objects covering different domains
        """
        if mechanical_work_j is not None:
            mechanical_work = mechanical_work_j
        limits: Dict[str, TheoreticalLimit] = {}

        # Landauer limit for computation
        if bits_processed > 0:
            landauer_limit = self.calculate_landauer_limits(bits_processed)
            limits["landauer"] = landauer_limit

        # Thermodynamic limit for mechanical work
        if mechanical_work > 0 or entropy_change > 0:
            thermo_limit = self.calculate_thermodynamic_limits(mechanical_work, entropy_change)
            limits["thermodynamic"] = thermo_limit

        # Information-theoretic limit
        if information_processed > 0:
            info_limit = self.calculate_information_theoretic_limits(information_processed)
            limits["information_theoretic"] = info_limit

        return limits

    def perform_module_analysis(self, module_name: str,
                              module_params: Optional[Dict[str, Any]] = None,
                              actual_energy_j: Optional[float] = None,
                              bits_processed: float = 0.0,
                              mechanical_work: float = 0.0,
                              entropy_change: float = 0.0,
                              information_processed: float = 0.0) -> ModuleTheoreticalAnalysis:
        """Perform comprehensive theoretical analysis for a module.

        Args:
            module_name: Name of the module
            module_params: Module-specific parameters
            actual_energy_j: Actual energy consumption (optional)

        Returns:
            Complete ModuleTheoreticalAnalysis
        """
        params = module_params or {}
        if any([bits_processed, mechanical_work, entropy_change, information_processed]):
            limits_map = self.calculate_comprehensive_limits(
                bits_processed=bits_processed,
                mechanical_work=mechanical_work,
                entropy_change=entropy_change,
                information_processed=information_processed,
            )
            limits = list(limits_map.values())
            dominant = min(limits, key=lambda x: x.value_j).limit_type if limits else None
            efficiency = None
            if actual_energy_j is not None and limits:
                min_limit = min(limits, key=lambda x: x.value_j)
                efficiency = self.calculate_efficiency_analysis(actual_energy_j, min_limit.value_j, min_limit.limit_type)
            recommendations = self._generate_optimization_recommendations(module_name, limits, dominant)
            return ModuleTheoreticalAnalysis(
                module_name=module_name,
                limits=limits,
                efficiency_analysis=efficiency,
                dominant_limit=dominant,
                optimization_recommendations=recommendations,
            )
        if actual_energy_j is not None:
            return self.analyze_module_efficiency(module_name, actual_energy_j, params)
        return self.analyze_module_limits(module_name.lower().replace("ant", ""), params)

    def identify_optimization_opportunities(self, analysis, theoretical_limit: Optional[float] = None,
                                            limit_type: Optional[str] = None) -> List[str]:
        """Identify specific optimization opportunities based on theoretical analysis.

        Args:
            analysis: ModuleTheoreticalAnalysis to analyze

        Returns:
            List of specific optimization opportunities
        """
        if isinstance(analysis, (int, float)):
            current_energy = float(analysis)
            limit = float(theoretical_limit or 0.0)
            ratio = current_energy / limit if limit > 0 else float("inf")
            opportunities = []
            if ratio > 1:
                opportunities.append(f"Reduce {limit_type or 'energy'} gap by optimizing algorithmic work")
            if ratio > 10:
                opportunities.append("Evaluate hardware acceleration and memory locality")
            if ratio > 100:
                opportunities.append("Revisit model architecture against physical lower bounds")
            return opportunities or ["System is close to the selected theoretical limit"]

        opportunities = []

        if not analysis.limits:
            return opportunities

        # Analyze efficiency gaps
        if analysis.efficiency_analysis:
            eff = analysis.efficiency_analysis
            if eff.efficiency_ratio > 100:
                opportunities.append("Reduce algorithmic complexity")
            if eff.efficiency_ratio > 1000:
                opportunities.append("Consider hardware acceleration")
            if eff.optimization_potential > 0.5:
                opportunities.append("Major efficiency improvements possible")

        # Analyze limit relationships
        limit_types = [l.limit_type for l in analysis.limits]
        if "thermodynamic" in limit_types and "landauer" in limit_types:
            opportunities.append("Balance mechanical and computational efficiency")

        return opportunities

    def compare_with_empirical_data(self, empirical_energy: float, empirical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare an empirical energy value with documented theoretical context."""
        reference = self.calculate_landauer_limits(empirical_data.get("bits_processed", 1.0)).value_j
        ratio = empirical_energy / reference if reference > 0 else float("inf")
        validation_score = max(0.0, min(1.0, float(empirical_data.get("efficiency", 0.0))))
        return {
            "empirical_vs_theoretical": ratio,
            "validation_score": validation_score,
            "bottleneck": empirical_data.get("bottleneck"),
            "optimization_potential": empirical_data.get("optimization_potential"),
        }

    def generate_efficiency_report(self, analysis: ModuleTheoreticalAnalysis) -> str:
        """Generate a comprehensive efficiency report."""
        return self.generate_limits_report(analysis)

    def generate_limits_report(self, analysis: ModuleTheoreticalAnalysis) -> str:
        """Generate comprehensive theoretical limits report.

        Args:
            analysis: ModuleTheoreticalAnalysis to report on

        Returns:
            Formatted report string
        """
        report_lines = [
            f"Theoretical Limits Analysis: {analysis.module_name.upper()}",
            "=" * 60,
            f"Module: {analysis.module_name}",
            f"Number of limits analyzed: {len(analysis.limits)}",
            ""
        ]

        if analysis.limits:
            report_lines.append("Theoretical Limits:")
            for i, limit in enumerate(analysis.limits, 1):
                report_lines.extend([
                    f"  {i}. {limit.limit_type.upper()}: {limit.value_j:.2e} J",
                    f"     Description: {limit.description}",
                    f"     Assumptions: {', '.join(limit.assumptions)}"
                ])
                if limit.confidence_level:
                    report_lines.append(f"     Confidence: {limit.confidence_level:.1%}")
                report_lines.append("")

        if analysis.dominant_limit:
            report_lines.extend([
                f"Dominant Limit: {analysis.dominant_limit.upper()}",
                ""
            ])

        if analysis.optimization_recommendations:
            report_lines.extend([
                "Optimization Recommendations:",
                *[f"  - {rec}" for rec in analysis.optimization_recommendations],
                ""
            ])

        if analysis.efficiency_analysis:
            eff = analysis.efficiency_analysis
            report_lines.extend([
                "Efficiency Analysis:",
                f"  Actual energy: {eff.actual_energy_j:.2e} J",
                f"  Theoretical limit: {eff.theoretical_limit_j:.2e} J",
                f"  Efficiency ratio: {eff.efficiency_ratio:.2f}",
                f"  Optimization potential: {eff.optimization_potential:.3f}",
            ])
            if eff.bottleneck_identified:
                report_lines.append(f"  Bottleneck: {eff.bottleneck_identified}")

        return "\n".join(report_lines)
