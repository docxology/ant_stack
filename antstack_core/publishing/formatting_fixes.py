"""Conservative prose formatting; never invent citation identities or reference targets."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from ..figures.references import _without_code
from ..mathematics import _contexts
from .provenance import atomic_text, ordinary_path

_VARIABLES = {
    **{f"E_{name}": r"$E_{\text{" + name + r"}}$" for name in
       ("body", "brain", "mind", "decision", "total", "compute", "actuation")},
    "N_KC": r"$N_{\text{KC}}$", "H_p": "$H_p$",
}
_VARIABLE_PATTERN = re.compile(r"(?<![\w@])(?:" + "|".join(map(re.escape, _VARIABLES)) + r")(?![\w@])")


def _map_prose(text, transform, *, transform_context=None):
    output = []
    end = 0
    for token, stop in _contexts(text):
        output.append(transform(text[end:token.start()]))
        fragment = text[token.start():stop]
        output.append(transform_context(token, fragment) if transform_context else fragment)
        end = stop
    output.append(transform(text[end:]))
    return "".join(output)


def fix_naked_urls(text: str) -> str:
    """Autolink bare HTTP(S) URLs without changing existing links or assigning authors."""
    def link(token, fragment):
        if not token.group("url") or not fragment.startswith(("http://", "https://")):
            return fragment
        url, suffix = fragment, ""
        while url and (url[-1] in ".,;:!?" or (url[-1] == ")" and url.count(")") > url.count("("))):
            suffix = url[-1] + suffix
            url = url[:-1]
        return "<" + url + ">" + suffix
    return _map_prose(text, lambda value: value, transform_context=link)


def fix_file_paths(text: str) -> str:
    """Code-format unquoted prose paths; preserve code, URLs, image targets and math."""
    def path(token, fragment):
        return "`" + fragment + "`" if token.group("path") else fragment
    return _map_prose(text, lambda value: value, transform_context=path)


def fix_variables(text: str) -> str:
    """Format unambiguous model identifiers only, not ordinary words such as alpha."""
    return _map_prose(text, lambda value: _VARIABLE_PATTERN.sub(lambda match: _VARIABLES[match.group()], value))


def _format(text: str) -> str:
    visible = _without_code(text)
    if re.search(r"\\(?:ref|cref|eqref)\{[^}]*\?\?[^}]*\}|(?:Figure|Table|Section|Equation)\s+\?\?", visible):
        raise ValueError("Unresolved reference requires an author-supplied target; no placeholder was written")
    return fix_variables(fix_file_paths(fix_naked_urls(text)))


def process_file(file_path: Path) -> None:
    """Atomically format one explicit Markdown file; preserve it on validation failure."""
    path = ordinary_path(file_path)
    if path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("Formatting requires an explicit Markdown file")
    original = path.read_text(encoding="utf-8")
    formatted = _format(original)
    if formatted != original:
        atomic_text(path, formatted)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Format explicit prose files; citation/reference identity repair is manual.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    # Prepare the whole selected batch before any mutation.
    planned = []
    for selected in args.files:
        path = ordinary_path(selected)
        if path.suffix.lower() not in {".md", ".markdown"}:
            parser.error(f"Not a Markdown file: {path}")
        original = path.read_text(encoding="utf-8")
        planned.append((path, original, _format(original)))
    for path, original, formatted in planned:
        if formatted != original:
            atomic_text(path, formatted)
        print(f"Formatted: {path}")
