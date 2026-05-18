"""Architecture and real-method contract tests."""

from __future__ import annotations

import ast
from pathlib import Path

from antstack_core.architecture import (
    build_default_architecture,
    render_architecture_mermaid,
    validate_architecture,
)
from antstack_core.analysis.complexity_analysis import (
    ComplexityEntropyAnalyzer,
    NetworkComplexityAnalyzer,
)
from antstack_core.analysis.scaling_analysis import ScalingAnalyzer


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "antstack_core"


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


def test_architecture_mermaid_is_renderable_source() -> None:
    """Architecture diagrams should come from the executable contract."""
    mermaid = render_architecture_mermaid()

    assert mermaid.startswith("flowchart TD")
    assert "run-all" not in mermaid.lower()
    assert "antstack_core" in mermaid
    assert "outputs" in mermaid
    assert "orchestration" in mermaid


def test_production_methods_are_not_placeholder_bodies() -> None:
    """Production methods should do real work, not contain empty pass bodies."""
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _PlaceholderBodyVisitor(path.relative_to(ROOT))
        visitor.visit(tree)
        offenders.extend(visitor.offenders)

    assert offenders == []


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


class _PlaceholderBodyVisitor(ast.NodeVisitor):
    """Find empty method bodies while allowing abstract interfaces."""

    def __init__(self, rel_path: Path):
        self.rel_path = rel_path
        self.class_stack: list[str] = []
        self.offenders: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast API
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if _is_abstract_method(node):
            return
        executable_body = _body_without_docstring(node.body)
        if len(executable_body) == 1 and _is_empty_statement(executable_body[0]):
            qualname = ".".join([*self.class_stack, node.name])
            self.offenders.append(f"{self.rel_path}:{node.lineno}:{qualname}")


def _body_without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def _is_empty_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
            return True
    return False
