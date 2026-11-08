"""Architecture and real-method contract tests."""

from __future__ import annotations

from pathlib import Path

from antstack_core.architecture import (
    build_default_architecture,
    validate_architecture,
)
from antstack_core.analysis.complexity_analysis import (
    ComplexityEntropyAnalyzer,
    NetworkComplexityAnalyzer,
)
from antstack_core.analysis.scaling_analysis import ScalingAnalyzer


ROOT = Path(__file__).resolve().parents[2]


def test_default_architecture_contract_validates() -> None:
    """The executable architecture contract should match the real checkout."""
    issues = validate_architecture(ROOT)
    assert [issue.to_dict() for issue in issues] == []


def test_default_architecture_is_fractal_and_queryable() -> None:
    """Nested architecture contracts should expose every major repo layer."""
    architecture = build_default_architecture()

    assert architecture.find_contract("antstack_core") is not None
    assert architecture.find_contract("outputs") is not None
    assert architecture.find_contract("antstack_core/orchestration") is not None

    layers = architecture.layer_index()
    assert "package kernel" in layers
    assert "workflow kernel" in layers
    assert "generated artifacts" in layers






def test_stateful_analyzers_record_real_results() -> None:
    """Analyzer constructors should initialize useful state, and methods should update it."""
    scaling = ScalingAnalyzer()
    scaling_result = scaling.analyze_single_parameter_scaling(
        [1.0, 2.0, 4.0, 8.0],
        [2.0, 4.0, 8.0, 16.0],
    )
    assert scaling.backend_capabilities.keys() == {"numpy", "scipy"}
    assert scaling.last_result == scaling_result
    assert scaling_result.valid

    network = NetworkComplexityAnalyzer()
    metrics = network.analyze_network_complexity(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert network.last_metrics == metrics
    assert metrics.network_density > 0.0

    entropy = ComplexityEntropyAnalyzer()
    diagram = entropy.create_complexity_entropy_diagram([0.0, 0.5, 1.0, 0.2, 0.8], window_size=3)
    assert entropy.last_diagram == diagram
    assert len(diagram["complexity"]) == 3


