"""Behavioral tests for antstack_core.figures.mermaid text-level logic.

Covers syntax validation, ASCII sanitization, image-signature validation,
caption generation, and the no-op preprocessing path. Network render backends
(mmdc/docker/kroki) are environment-dependent and are deliberately not exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path (tests run without the package installed in some envs)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.figures.mermaid import (
    ValidationResult,
    _generate_diagram_caption,
    _is_valid_image,
    preprocess_mermaid_diagrams,
    sanitize_mermaid_for_rendering,
    validate_mermaid_syntax,
)


class TestValidationResult:
    def test_unpacks_as_valid_issues_pair(self) -> None:
        result = ValidationResult({"valid": True, "issues": []})
        valid, issues = result
        assert valid is True
        assert issues == []

    def test_defaults_when_keys_absent(self) -> None:
        valid, issues = ValidationResult({})
        assert valid is False
        assert issues == []


class TestValidateMermaidSyntax:
    def test_clean_flowchart_is_valid(self) -> None:
        result = validate_mermaid_syntax("graph TD\n  A[Start] --> B[End]")
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["line_count"] == 2

    def test_empty_code_is_invalid(self) -> None:
        result = validate_mermaid_syntax("   ")
        assert result["valid"] is False
        assert "Empty Mermaid code" in result["issues"]

    def test_missing_diagram_type_is_flagged(self) -> None:
        result = validate_mermaid_syntax("A --> B")
        assert result["valid"] is False
        assert any("diagram type" in issue for issue in result["issues"])

    def test_fenced_block_is_extracted_before_validation(self) -> None:
        fenced = "```mermaid\nflowchart LR\n  A --> B\n```"
        result = validate_mermaid_syntax(fenced)
        assert result["valid"] is True
        assert result["line_count"] == 2

    def test_unicode_characters_are_flagged(self) -> None:
        result = validate_mermaid_syntax("graph TD\n  A[alpha α] --> B[beta β]")
        assert result["valid"] is False
        assert any("α" in issue for issue in result["issues"])

    def test_latex_math_is_flagged(self) -> None:
        result = validate_mermaid_syntax("graph TD\n  A[$x^2$] --> B")
        assert result["valid"] is False
        assert any("LaTeX" in issue for issue in result["issues"])

    def test_unbalanced_brackets_are_flagged(self) -> None:
        result = validate_mermaid_syntax("graph TD\n  A[Start --> B")
        assert any("Unbalanced square brackets" in issue for issue in result["issues"])

    def test_unbalanced_braces_are_flagged(self) -> None:
        result = validate_mermaid_syntax("graph TD\n  A{Start --> B")
        assert any("Unbalanced braces" in issue for issue in result["issues"])

    def test_mid_line_end_is_flagged(self) -> None:
        code = "graph TD\n  subgraph S\n  A --> B end"
        result = validate_mermaid_syntax(code)
        assert any("'end' keyword not on separate line" in issue for issue in result["issues"])

    def test_all_recognized_diagram_types(self) -> None:
        for decl in ("graph TD", "flowchart LR", "sequenceDiagram", "classDiagram",
                     "stateDiagram-v2", "gantt", "pie", "erDiagram"):
            code = f"{decl}\n  a"
            if decl == "sequenceDiagram":
                code = f"{decl}\n  a->>b: hi"
            result = validate_mermaid_syntax(code)
            assert result["valid"] is True, decl


class TestSanitizeMermaidForRendering:
    def test_unicode_symbols_replaced_with_ascii(self) -> None:
        code = sanitize_mermaid_for_rendering("A[α] -->|→| B[β × γ]")
        assert "α" not in code
        assert "β" not in code
        assert "->" in code
        assert "x" in code

    def test_stylized_big_o_normalized(self) -> None:
        code = sanitize_mermaid_for_rendering("A[𝒪(n)] --> B")
        assert "𝒪" not in code
        assert "O(n)" in code

    def test_mathrm_commands_are_stripped(self) -> None:
        code = sanitize_mermaid_for_rendering("A[" + chr(92) + "," + chr(92) + "mathrm{A1}] --> B")
        assert "\\mathrm" not in code
        assert "A1" in code

    def test_texttt_commands_unwrapped(self) -> None:
        code = sanitize_mermaid_for_rendering(r"A[\\texttt{sensor}] --> B")
        assert "\\texttt" not in code
        assert "sensor" in code

    def test_inline_math_markers_stripped(self) -> None:
        code = sanitize_mermaid_for_rendering("A[$x$] --> B")
        assert "$" not in code
        assert "x" in code

    def test_edge_labels_removed(self) -> None:
        code = sanitize_mermaid_for_rendering("A -->|rate 0.5| B")
        assert "|" not in code

    def test_semicolons_split_into_lines(self) -> None:
        code = sanitize_mermaid_for_rendering("graph TD; A --> B; B --> C")
        assert ";" not in code
        assert "A --> B\n" in code

    def test_idempotent_on_already_clean_code(self) -> None:
        clean = "graph TD\n  A[Start] --> B[End]"
        once = sanitize_mermaid_for_rendering(clean)
        assert sanitize_mermaid_for_rendering(once) == once

    def test_round_trip_with_validation(self) -> None:
        """Sanitized output of flagged code should re-validate cleanly."""
        raw = "graph TD\n  A[α × β] -->|→| B[\\texttt{C}]"
        issues = validate_mermaid_syntax(raw)["issues"]
        assert issues  # raw code is genuinely problematic
        sanitized = sanitize_mermaid_for_rendering(raw)
        result = validate_mermaid_syntax(sanitized)
        assert result["valid"] is True


class TestIsValidImage:
    def test_png_signature_accepted(self, tmp_path: Path) -> None:
        png = tmp_path / "x.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"rest of a real header")
        assert _is_valid_image(str(png)) is True

    def test_svg_accepted(self, tmp_path: Path) -> None:
        svg = tmp_path / "x.svg"
        svg.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>")
        assert _is_valid_image(str(svg)) is True

    def test_xml_prolog_accepted(self, tmp_path: Path) -> None:
        svg = tmp_path / "x.svg"
        svg.write_text('<?xml version="1.0"?><svg></svg>')
        assert _is_valid_image(str(svg)) is True

    def test_garbage_rejected(self, tmp_path: Path) -> None:
        junk = tmp_path / "x.png"
        junk.write_bytes(b"not an image at all........")
        assert _is_valid_image(str(junk)) is False

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        assert _is_valid_image(str(tmp_path / "nope.png")) is False


class TestGenerateDiagramCaption:
    def test_analysis_pipeline_caption(self) -> None:
        caption = _generate_diagram_caption("Analysis[A] --> Methods[M]", "diagram_abcd1234")
        assert "analysis pipeline" in caption.lower()

    def test_module_complexity_caption(self) -> None:
        caption = _generate_diagram_caption("Body[B] --> Brain[Br]", "diagram_abcd1234")
        assert "complexity" in caption.lower()

    def test_energy_flows_caption(self) -> None:
        caption = _generate_diagram_caption("Physical[P] --> Control[C]", "diagram_abcd1234")
        assert "energy" in caption.lower()

    def test_generic_caption_includes_hash_prefix(self) -> None:
        caption = _generate_diagram_caption("A --> B", "diagram_abcd1234")
        assert "abcd" in caption


class TestPreprocessPassthrough:
    def test_no_output_dir_returns_content_unchanged(self) -> None:
        content = "# Title\n\n```mermaid\ngraph TD\n  A --> B\n```\n"
        assert preprocess_mermaid_diagrams(content) == content

    def test_content_without_mermaid_blocks_unchanged(self, tmp_path: Path) -> None:
        content = "# Title\n\nSome plain text.\n"
        assert preprocess_mermaid_diagrams(content, tmp_path) == content

    def test_empty_mermaid_block_left_in_place(self, tmp_path: Path) -> None:
        content = "```mermaid\n\n```"
        assert preprocess_mermaid_diagrams(content, tmp_path, clean_existing=False) == content
