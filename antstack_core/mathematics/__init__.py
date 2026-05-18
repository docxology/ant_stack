"""Mathematics helpers used by Ant Stack publishing workflows.

The module keeps small, dependency-light helpers for normalizing scientific
notation before Markdown is handed to the publishing stack.
"""

from __future__ import annotations

import re

UNICODE_TO_LATEX = {
    "≈": r"\approx",
    "≤": r"\le",
    "≥": r"\ge",
    "±": r"\pm",
    "×": r"\times",
    "→": r"\to",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "Ω": r"\Omega",
}


def normalize_unicode_math(text: str) -> str:
    """Replace common Unicode math symbols with inline LaTeX macros."""
    normalized = text
    for symbol, macro in UNICODE_TO_LATEX.items():
        normalized = normalized.replace(symbol, f"${macro}$")
    return normalized


def extract_latex_labels(text: str) -> list[str]:
    """Return LaTeX labels found in Markdown or LaTeX source."""
    return re.findall(r"\\label\{([^}]+)\}", text)


__all__ = ["UNICODE_TO_LATEX", "extract_latex_labels", "normalize_unicode_math"]
