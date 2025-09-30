"""Scientific analysis utilities for quantitative research.

This module provides comprehensive analysis capabilities including:
- Energy estimation and breakdown analysis
- Statistical methods with uncertainty quantification
- Scaling relationship analysis and power law detection
- Bootstrap confidence intervals and significance testing
- Theoretical limit calculations and efficiency metrics
- Key numbers integration for paper sections
- Enhanced scaling analysis with multiple parameters

Design follows .cursorrules principles:
- Real methods only (no mocks)
- Test-driven development
- Professional, functional implementation
- Comprehensive statistical validation
"""

from .energy import (
    EnergyCoefficients,
    ComputeLoad,
    EnergyBreakdown,
    estimate_detailed_energy,
    pj_to_j,
    aj_to_j,
    GRAVITY_M_S2,
    j_to_mj,
    j_to_kj,
    j_to_wh,
    wh_to_j,
    w_to_mw,
    mw_to_w,
    s_to_ms,
    ms_to_s,
    integrate_power_to_energy,
    estimate_compute_energy,
    add_baseline_energy,
    cost_of_transport
)
from .statistics import (
    bootstrap_mean_ci,
    analyze_scaling_relationship,
    calculate_energy_efficiency_metrics,
    estimate_theoretical_limits
)
from .workloads import (
    body_workload_closed_form,
    brain_workload_closed_form,
    mind_workload_closed_form,
    calculate_contact_complexity,
    calculate_sparse_neural_complexity,
    calculate_active_inference_complexity,
    body_workload,
    brain_workload,
    mind_workload,
    estimate_body_energy_mech,
    enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form,
    estimate_body_compute_per_decision,
    estimate_brain_compute_per_decision,
    estimate_mind_compute_per_decision
)
from .key_numbers import KeyNumbersLoader, KeyNumbersManager
from .enhanced_estimators import EnhancedEnergyEstimator
from .scaling_analysis import ScalingAnalyzer
from .theoretical_limits import TheoreticalLimitsAnalyzer
from .power_meters import (
    PowerSample,
    PowerMeter,
    NullPowerMeter,
    RaplPowerMeter,
    NvmlPowerMeter,
    measure_energy,
    create_power_meter
)
from .experiment_config import (
    WorkloadConfig,
    EnergyCoefficientsConfig,
    ScalingConfig,
    MeterConfig,
    ExperimentManifest
)
from .empirical_data import EmpiricalEvidence, CaseStudy
from .statistical_analysis import (
    StatisticalAnalyzer,
    analyze_measurement_quality,
    detect_significance,
    perform_k_fold_cross_validation
)
from .report_generator import ReportGenerator
from .veridical_reporting import VeridicalReporter

__all__ = [
    "EnergyCoefficients",
    "ComputeLoad",
    "EnergyBreakdown",
    "estimate_detailed_energy",
    "pj_to_j",
    "bootstrap_mean_ci",
    "analyze_scaling_relationship",
    "calculate_energy_efficiency_metrics",
    "estimate_theoretical_limits",
    "body_workload_closed_form",
    "brain_workload_closed_form",
    "mind_workload_closed_form",
    "calculate_contact_complexity",
    "calculate_sparse_neural_complexity",
    "calculate_active_inference_complexity",
    "body_workload",
    "brain_workload",
    "mind_workload",
    "estimate_body_energy_mech",
    "KeyNumbersLoader",
    "KeyNumbersManager",
    "EnhancedEnergyEstimator",
    "ScalingAnalyzer",
    "TheoreticalLimitsAnalyzer",
    "PowerSample",
    "PowerMeter",
    "NullPowerMeter",
    "RaplPowerMeter",
    "NvmlPowerMeter",
    "measure_energy",
    "create_power_meter",
    "WorkloadConfig",
    "EnergyCoefficientsConfig",
    "ScalingConfig",
    "MeterConfig",
    "ExperimentManifest",
    "EmpiricalEvidence",
    "CaseStudy",
    "StatisticalAnalyzer",
    "analyze_measurement_quality",
    "detect_significance",
    "perform_k_fold_cross_validation",
    "ReportGenerator",
    "VeridicalReporter"
]
