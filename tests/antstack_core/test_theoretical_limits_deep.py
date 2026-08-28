"""Deepened behavioral tests for antstack_core.analysis.theoretical_limits.

Edge cases, physical invariants, dispatch contracts, and report rendering on
top of the existing unit tests — all real computation, no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from antstack_core.analysis.theoretical_limits import (
    EfficiencyAnalysis,
    ModuleTheoreticalAnalysis,
    TheoreticalLimit,
    TheoreticalLimitsAnalyzer,
)


class TestPhysicalInvariants:
    def test_landauer_scales_linearly_with_bits(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        one = analyzer.calculate_landauer_limits(1.0)
        million = analyzer.calculate_landauer_limits(1_000_000.0)
        assert million.value_j == analyzer.landauer_limit * 1_000_000
        assert million.value_j / one.value_j == 1_000_000

    def test_landauer_contract_is_the_same_order_as_kt_ln2(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        import math
        kT_ln2 = analyzer.kT * math.log(2)
        # The repo documents 1.4e-21 as a conservative Landauer-style contract:
        # same order of magnitude as kT*ln(2), and strictly below it (idealized).
        assert 0.1 * kT_ln2 < analyzer.landauer_limit <= kT_ln2

    def test_thermodynamic_limit_never_below_the_work_itself(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        for efficiency in (1.0, 0.9, 0.45, 0.1, 1e-9):
            limit = analyzer.calculate_thermodynamic_limits(0.001, efficiency=efficiency)
            assert limit.value_j >= 0.001 - 1e-15

    def test_lower_efficiency_raises_required_energy(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        ideal = analyzer.calculate_thermodynamic_limits(0.001, efficiency=1.0)
        lossy = analyzer.calculate_thermodynamic_limits(0.001, efficiency=0.25)
        assert lossy.value_j > ideal.value_j

    def test_entropy_change_adds_kT_term(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        base = analyzer.calculate_thermodynamic_limits(0.001, entropy_change=0.0)
        dissipative = analyzer.calculate_thermodynamic_limits(0.001, entropy_change=1.0)
        assert abs((dissipative.value_j - base.value_j) - analyzer.kT) < 1e-18

    def test_nonpositive_work_yields_zero_limit(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        for work in (0.0, -0.5):
            limit = analyzer.calculate_thermodynamic_limits(work)
            assert limit.value_j == 0.0

    def test_information_limits_scale_with_capacity_and_time(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        a = analyzer.calculate_information_limits(100.0, 0.01)
        b = analyzer.calculate_information_limits(200.0, 0.01)
        c = analyzer.calculate_information_limits(100.0, 0.02)
        assert b.value_j == 2 * a.value_j
        assert c.value_j == 2 * a.value_j

    def test_neuromorphic_energy_uses_ten_millisecond_decision_window(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        limit = analyzer.calculate_neuromorphic_limits(1_000_000.0, spike_energy_aj=0.4)
        expected = 1_000_000 * 0.4 * 1e-18 * 0.01
        assert abs(limit.value_j - expected) < 1e-30


class TestEfficiencyAnalysis:
    def test_perfect_efficiency_has_zero_potential(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.calculate_efficiency_analysis(1e-9, 1e-9, "landauer")
        assert analysis.efficiency_ratio == 1.0
        assert analysis.optimization_potential == 0.0

    def test_sub_limit_actual_energy_clamps_potential_to_zero(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.calculate_efficiency_analysis(1e-12, 1e-9, "landauer")
        assert analysis.efficiency_ratio < 1.0
        assert analysis.optimization_potential == 0.0

    def test_nonpositive_theoretical_limit_gives_infinite_ratio(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.calculate_efficiency_analysis(1e-9, 0.0, "landauer")
        assert analysis.efficiency_ratio == float("inf")
        assert analysis.optimization_potential == 0.0

    def test_bottleneck_ladder(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        limit = 1e-15
        assert analyzer.calculate_efficiency_analysis(limit * 5, limit, "x").bottleneck_identified is None
        assert analyzer.calculate_efficiency_analysis(limit * 50, limit, "x").bottleneck_identified == "Hardware limitations"
        assert analyzer.calculate_efficiency_analysis(limit * 500, limit, "x").bottleneck_identified == "Implementation overhead"
        assert analyzer.calculate_efficiency_analysis(limit * 5000, limit, "x").bottleneck_identified == "Algorithmic inefficiency"

    def test_optimization_potential_is_capped_below_one(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.calculate_efficiency_analysis(1e-3, 1e-21, "landauer")
        assert 0.0 < analysis.optimization_potential < 1.0


class TestModuleDispatch:
    def test_body_brain_mind_each_produce_three_limits(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        for module, expected in (("body", 3), ("brain", 3), ("mind", 2)):
            analysis = analyzer.analyze_module_limits(module, {})
            assert len(analysis.limits) == expected, module
            assert analysis.dominant_limit is not None

    def test_unknown_module_has_no_limits(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.analyze_module_limits("unknown", {})
        assert analysis.limits == []
        assert analysis.dominant_limit is None
        assert analysis.optimization_recommendations == []

    def test_dominant_limit_is_the_minimum_energy_limit(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.analyze_module_limits("body", {})
        minimum = min(analysis.limits, key=lambda limit: limit.value_j)
        assert analysis.dominant_limit == minimum.limit_type

    def test_module_name_strips_ant_prefix_in_perform_module_analysis(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        direct = analyzer.perform_module_analysis("body", actual_energy_j=1e-6)
        prefixed = analyzer.perform_module_analysis("AntBody", actual_energy_j=1e-6)
        assert direct.module_name == prefixed.module_name
        assert len(direct.limits) == len(prefixed.limits)

    def test_analyze_module_efficiency_numeric_form(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        result = analyzer.analyze_module_efficiency(1e-9, 1e-12, "landauer")
        assert isinstance(result, EfficiencyAnalysis)
        import pytest
        assert result.efficiency_ratio == pytest.approx(1000.0)

    def test_analyze_module_efficiency_module_form(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        result = analyzer.analyze_module_efficiency("body", 1e-6, {})
        assert isinstance(result, ModuleTheoreticalAnalysis)
        assert result.efficiency_analysis is not None
        assert result.efficiency_analysis.actual_energy_j == 1e-6

    def test_comprehensive_limits_only_includes_requested_domains(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        empty = analyzer.calculate_comprehensive_limits()
        assert empty == {}
        bits_only = analyzer.calculate_comprehensive_limits(bits_processed=100.0)
        assert set(bits_only) == {"landauer"}
        both = analyzer.calculate_comprehensive_limits(
            bits_processed=100.0, mechanical_work_j=0.001, information_processed=10.0
        )
        assert set(both) == {"landauer", "thermodynamic", "information_theoretic"}

    def test_comprehensive_limits_mechanical_work_j_alias(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        via_alias = analyzer.calculate_comprehensive_limits(mechanical_work_j=0.002)
        via_name = analyzer.calculate_comprehensive_limits(mechanical_work=0.002)
        assert via_alias["thermodynamic"].value_j == via_name["thermodynamic"].value_j


class TestOptimizationRecommendations:
    def test_thermodynamic_dominant_gives_actuator_advice(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        # A vanishingly small mechanical work floor makes the thermodynamic
        # limit the binding minimum (work/efficiency ≈ 2.2e-20 J).
        analysis = analyzer.analyze_module_limits("body", {"mechanical_work_j": 1e-20})
        assert analysis.dominant_limit == "thermodynamic"
        assert any("actuator" in rec for rec in analysis.optimization_recommendations)

    def test_landauer_dominant_gives_reversible_computing_advice(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        # A single flop (32 bits) puts the Landauer bound below the default
        # information limit (32*1.4e-21 ≈ 4.5e-20 J < 1.79e-19 J).
        analysis = analyzer.analyze_module_limits("brain", {"flops": 1})
        assert analysis.dominant_limit == "landauer"
        assert any("reversible" in rec for rec in analysis.optimization_recommendations)

    def test_large_limit_gap_appends_multi_objective_note(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.analyze_module_limits("body", {})
        values = [limit.value_j for limit in analysis.limits]
        if max(values) / min(values) > 10:
            assert any("multi-objective" in rec for rec in analysis.optimization_recommendations)


class TestReports:
    def test_limits_report_contains_key_sections(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.perform_module_analysis("body", actual_energy_j=1e-6)
        report = analyzer.generate_limits_report(analysis)
        assert "Theoretical Limits Analysis: BODY" in report
        assert "Dominant Limit" in report
        assert "Optimization Recommendations" in report
        assert "Efficiency Analysis" in report

    def test_efficiency_report_is_alias_for_limits_report(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.perform_module_analysis("brain", actual_energy_j=1e-9)
        assert analyzer.generate_efficiency_report(analysis) == analyzer.generate_limits_report(analysis)

    def test_report_includes_bottleneck_when_present(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.perform_module_analysis("mind", actual_energy_j=1e-3)
        if analysis.efficiency_analysis and analysis.efficiency_analysis.bottleneck_identified:
            report = analyzer.generate_limits_report(analysis)
            assert "Bottleneck:" in report

    def test_compare_with_empirical_data_contract(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        result = analyzer.compare_with_empirical_data(
            1e-9,
            {"bits_processed": 1e6, "efficiency": 0.42, "bottleneck": "actuation"},
        )
        assert result["validation_score"] == 0.42
        assert result["bottleneck"] == "actuation"
        assert result["empirical_vs_theoretical"] > 0

    def test_identify_opportunities_numeric_path(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        assert analyzer.identify_optimization_opportunities(1.0, 1.0, "energy") == [
            "System is close to the selected theoretical limit"
        ]
        gaps = analyzer.identify_optimization_opportunities(1000.0, 1.0, "energy")
        assert any("hardware" in gap.lower() for gap in gaps)
        assert any("architecture" in gap.lower() for gap in gaps)
