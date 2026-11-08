"""Deepened behavioral tests for antstack_core.analysis.theoretical_limits.

Edge cases, physical invariants, dispatch contracts, and report rendering on
top of the existing unit tests — all real computation, no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import math
import pytest

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
        assert million.value_j / one.value_j == pytest.approx(1_000_000, rel=1e-14)

    def test_landauer_is_kt_ln2_at_stated_temperature(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        assert analyzer.calculate_landauer_limits(1).value_j == pytest.approx(1.380649e-23 * 298.15 * math.log(2), rel=1e-14, abs=0)
        hot = TheoreticalLimitsAnalyzer(596.3)
        assert hot.calculate_landauer_limits(1).value_j == pytest.approx(2 * analyzer.landauer_limit, rel=1e-14, abs=0)

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

    def test_entropy_change_has_joules_per_kelvin_units(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        base = analyzer.calculate_thermodynamic_limits(.001)
        dissipative = analyzer.calculate_thermodynamic_limits(.001, entropy_change=1.0)
        assert dissipative.value_j - base.value_j == pytest.approx(298.15)
        assert analyzer.calculate_thermodynamic_limits(0, entropy_change=1.0).value_j == 298.15

    def test_zero_work_is_valid_but_negative_work_is_rejected(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        assert analyzer.calculate_thermodynamic_limits(0).value_j == 0
        with pytest.raises(ValueError):
            analyzer.calculate_thermodynamic_limits(-.5)

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

    @pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
    def test_invalid_physical_inputs_are_rejected(self, bad) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        calls = (
            lambda: analyzer.calculate_landauer_limits(bad),
            lambda: analyzer.calculate_thermodynamic_limits(bad),
            lambda: analyzer.calculate_thermodynamic_limits(1, entropy_change=bad),
            lambda: analyzer.calculate_information_limits(bad, 1),
            lambda: analyzer.calculate_information_limits(1, bad),
            lambda: analyzer.calculate_neuromorphic_limits(bad),
            lambda: analyzer.calculate_neuromorphic_limits(1, bad),
            lambda: analyzer.calculate_comprehensive_limits(mechanical_work=bad),
            lambda: analyzer.calculate_efficiency_analysis(bad, 1, 'example'),
        )
        for call in calls:
            with pytest.raises(ValueError):
                call()

    @pytest.mark.parametrize('efficiency', [0, -1, 1.01, float('nan'), float('inf')])
    def test_invalid_mechanical_efficiency(self, efficiency) -> None:
        with pytest.raises(ValueError):
            TheoreticalLimitsAnalyzer().calculate_thermodynamic_limits(1, efficiency=efficiency)

    @pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
    def test_invalid_temperature(self, temperature) -> None:
        with pytest.raises(ValueError):
            TheoreticalLimitsAnalyzer(temperature)


class TestEfficiencyAnalysis:
    def test_perfect_efficiency_has_zero_potential(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.calculate_efficiency_analysis(1e-9, 1e-9, "landauer")
        assert analysis.efficiency_ratio == 1.0
        assert analysis.optimization_potential == 0.0

    def test_sub_bound_energy_identifies_incompatible_assumptions(self) -> None:
        analysis = TheoreticalLimitsAnalyzer().calculate_efficiency_analysis(1e-12, 1e-9, 'landauer')
        assert analysis.efficiency_ratio == pytest.approx(.001)
        assert analysis.valid is False
        assert analysis.error

    def test_zero_reference_or_actual_has_undefined_ratios(self) -> None:
        for actual, bound in ((1, 0), (0, 1), (0, 0)):
            analysis = TheoreticalLimitsAnalyzer().calculate_efficiency_analysis(actual, bound, 'landauer')
            assert analysis.efficiency_ratio is None
            assert analysis.efficiency_fraction is None
            assert analysis.optimization_potential is None
            assert not analysis.valid

    def test_bottleneck_ladder(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        limit = 1e-15
        assert analyzer.calculate_efficiency_analysis(limit * 5, limit, "x").bottleneck_identified is None
        assert analyzer.calculate_efficiency_analysis(limit * 50, limit, "x").bottleneck_identified == "Hardware limitations"
        assert analyzer.calculate_efficiency_analysis(limit * 500, limit, "x").bottleneck_identified == "Implementation overhead"
        assert analyzer.calculate_efficiency_analysis(limit * 5000, limit, "x").bottleneck_identified == "Algorithmic inefficiency"

    def test_overhead_fraction_and_reducible_energy_are_distinct(self) -> None:
        analysis = TheoreticalLimitsAnalyzer().calculate_efficiency_analysis(4, 1, 'example')
        assert analysis.efficiency_ratio == 4
        assert analysis.efficiency_fraction == .25
        assert analysis.optimization_potential == .75


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


    def test_dominant_limit_is_the_binding_common_total_floor(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.analyze_module_limits('body', {})
        binding = max(analysis.limits, key=lambda limit: limit.value_j)
        assert analysis.dominant_limit == binding.limit_type
        combined = analyzer.perform_module_analysis('example', actual_energy_j=1, bits_processed=1, mechanical_work=1)
        assert combined.dominant_limit == 'thermodynamic'
        assert combined.efficiency_analysis.theoretical_limit_j == 1
        assert combined.efficiency_analysis.efficiency_ratio == 1
        assert len(combined.limits) == 2
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
        analysis = analyzer.analyze_module_limits("body", {"mechanical_work_j": 1.0})
        assert analysis.dominant_limit == "thermodynamic"
        assert any("actuator" in rec for rec in analysis.optimization_recommendations)

    def test_landauer_dominant_gives_reversible_computing_advice(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        analysis = analyzer.analyze_module_limits("brain", {"flops": 1e12})
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


    def test_identify_opportunities_numeric_path(self) -> None:
        analyzer = TheoreticalLimitsAnalyzer()
        assert analyzer.identify_optimization_opportunities(1.0, 1.0, "energy") == [
            "System is close to the selected theoretical limit"
        ]
        gaps = analyzer.identify_optimization_opportunities(1000.0, 1.0, "energy")
        assert any("hardware" in gap.lower() for gap in gaps)
        assert any("architecture" in gap.lower() for gap in gaps)


def test_supplied_efficiency_does_not_certify_empirical_validation():
    import math
    import pytest
    analyzer = TheoreticalLimitsAnalyzer(temperature_k=310.0)
    for efficiency in (0, 0.42, 1):
        result = analyzer.compare_with_empirical_data(1e-9, {"bits_processed": 1e6, "efficiency": efficiency})
        assert result["empirical_vs_theoretical"] == pytest.approx(1e-9 / (1e6 * 1.380649e-23 * 310.0 * math.log(2)))
        assert result["validation_status"] == "not-assessed"
        assert "validation_score" not in result
    assert analyzer.compare_with_empirical_data(0, {"bits_processed": 0})["empirical_vs_theoretical"] is None
    for energy in (-1, math.inf, math.nan):
        with pytest.raises(ValueError):
            analyzer.compare_with_empirical_data(energy, {})
