"""Behavioral tests for antstack_core.mathematics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.mathematics import (
    UNICODE_TO_LATEX,
    extract_latex_labels,
    normalize_unicode_math,
)


class TestNormalizeUnicodeMath:
    def test_known_symbols_are_wrapped_in_inline_math(self) -> None:
        assert normalize_unicode_math("a " + chr(8804) + " b") == "a $\\le$ b"
        assert normalize_unicode_math("x " + chr(177) + " y") == "x $\\pm$ y"

    def test_greek_letters_converted(self) -> None:
        out = normalize_unicode_math(chr(945) + " " + chr(946) + " " + chr(947))
        assert out == "$\\alpha$ $\\beta$ $\\gamma$"

    def test_plain_ascii_unchanged(self) -> None:
        text = "plain text with no math symbols 123"
        assert normalize_unicode_math(text) == text

    def test_all_mapping_entries_round_trip(self) -> None:
        """Every mapping symbol is replaced and no Unicode symbol survives."""
        text = "".join(UNICODE_TO_LATEX)
        out = normalize_unicode_math(text)
        for symbol in UNICODE_TO_LATEX:
            assert symbol not in out
        assert out.count("$") == 2 * len(UNICODE_TO_LATEX)

    def test_empty_string(self) -> None:
        assert normalize_unicode_math("") == ""


class TestExtractLatexLabels:
    def test_finds_single_label(self) -> None:
        text = "See equation " + chr(92) + "label{eq:energy} for details."
        assert extract_latex_labels(text) == ["eq:energy"]

    def test_finds_multiple_labels_in_order(self) -> None:
        text = (
            chr(92) + "label{eq:a} and " + chr(92) + "label{fig:b} and "
            + chr(92) + "label{sec:c}"
        )
        assert extract_latex_labels(text) == ["eq:a", "fig:b", "sec:c"]

    def test_no_labels_returns_empty(self) -> None:
        assert extract_latex_labels("no labels here") == []

    def test_label_with_nested_braces_takes_up_to_first_close(self) -> None:
        text = chr(92) + "label{eq:nested{inner}}"
        assert extract_latex_labels(text) == ["eq:nested{inner"]

    def test_realistic_markdown_document(self) -> None:
        doc = (
            "# Energy\n\n"
            "$$" + chr(92) + "label{eq:free_energy} F = E - TS$$\n\n"
            "As shown in " + chr(92) + "eqref{eq:free_energy}."
        )
        assert extract_latex_labels(doc) == ["eq:free_energy"]
