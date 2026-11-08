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

_CONTEXT_TOKENS = re.compile(
    r"(?P<code>^[ ]{0,3}(?P<fence>(?P<fchar>\x60|~)(?P=fchar){2,})[^\n]*\n.*?(?:^[ ]{0,3}(?P=fence)(?P=fchar)*[ \t]*(?=\n|$)|\Z)"
    r"|(?<!\x60)(?P<ticks>\x60+)(?!\x60).*?(?<!\x60)(?P=ticks)(?!\x60)"
    r"|\\begin\{(?P<verbatim>verbatim\*?|lstlisting|minted)\}.*?\\end\{(?P=verbatim)\}"
    r"|\\verb\*?(?P<verbsep>[^\w\s]).*?(?P=verbsep))"
    r"|(?P<math>(?<!\\)\$\$.*?(?<!\\)\$\$"
    r"|(?<![\\$])\$(?!\$).*?(?<!\\)\$"
    r"|\\\(.*?\\\)|\\\[.*?\\\]"
    r"|\\begin\{(?P<environment>equation\*?|align\*?|gather\*?|multline\*?|displaymath|math|eqnarray\*?)\}"
    r".*?\\end\{(?P=environment)\})"
    r"|(?P<destination>\]\()"
    r"|(?P<latex>\\[A-Za-z]+\*?(?=\s*[\[{]))"
    r"|(?P<identifier>\{#[^}]+\})"
    r"|(?P<citation>(?<![\w\\])@[A-Za-z0-9_][A-Za-z0-9_:.+-]*)"
    r"|(?P<autolink><[A-Za-z][A-Za-z0-9+.-]*:[^>\n]+>)"
    r"|(?P<url>[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\[\]{}]+)"
    r"|(?P<path>(?<![\w@])[\w./~-]+\.(?:py|md|yaml|yml|json|txt|csv)(?![\w@]))",
    re.MULTILINE | re.DOTALL,
)
_SYMBOLS = re.compile("[" + re.escape("".join(UNICODE_TO_LATEX)) + "]")


def _balanced_end(text: str, start: int) -> int:
    """End of one balanced Markdown/TeX argument; ambiguous input stays protected."""
    closing = {"(": ")", "[": "]", "{": "}"}
    stack = [closing[text[start]]]
    cursor = start + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char in closing:
            stack.append(closing[char])
        elif char == stack[-1]:
            stack.pop()
            if not stack:
                return cursor + 1
        cursor += 1
    return len(text)


def _contexts(text: str):
    """Yield protected spans shared by Unicode and prose-only formatting."""
    consumed = 0
    for token in _CONTEXT_TOKENS.finditer(text):
        if token.start() < consumed:
            continue
        end = token.end()
        if token.group("destination"):
            end = _balanced_end(text, end - 1)
        elif token.group("latex"):
            cursor = end
            while cursor < len(text):
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] not in "[{":
                    break
                end = cursor = _balanced_end(text, cursor)
        consumed = end
        yield token, end


def normalize_unicode_math(text: str) -> str:
    r"""Normalize supported math glyphs, preserving code, identifiers and resource URIs.

    Existing $, $$, \(...\), \[...] and math environments receive bare macros;
    ordinary prose receives inline math. Raw non-math TeX arguments are untouched.
    This is notation normalization, not a mathematical or full TeX validator.
    """
    def convert(segment: str, in_math: bool = False) -> str:
        def replace(match):
            macro = UNICODE_TO_LATEX[match.group()]
            if not in_math:
                return "$" + macro + "$"
            following = segment[match.end():match.end() + 1]
            return macro + ("{}" if following and following.isascii() and following.isalpha() else "")
        return _SYMBOLS.sub(replace, segment)
    output = []
    end = 0
    for token, stop in _contexts(text):
        output.append(convert(text[end:token.start()]))
        fragment = text[token.start():stop]
        output.append(convert(fragment, True) if token.group("math") else fragment)
        end = stop
    output.append(convert(text[end:]))
    return "".join(output)


def extract_latex_labels(text: str) -> list[str]:
    """Return LaTeX labels found in Markdown or LaTeX source."""
    return re.findall(r"\\label\{([^}]+)\}", text)


__all__ = ["UNICODE_TO_LATEX", "extract_latex_labels", "normalize_unicode_math"]
