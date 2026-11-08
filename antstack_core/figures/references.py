"""Document-wide reference validation; explicit mappings never guess evidence targets."""

from __future__ import annotations

import re
from typing import Any

from ..mathematics import _balanced_end, _contexts


class ValidationResult(dict):
    """Mapping also unpackable as ``valid, issues``."""

    def __iter__(self):
        yield self.get("valid", False)
        yield self.get("issues", [])


def _without_code(content: str) -> str:
    """Mask non-content examples/resources, preserving labels, math and line numbers."""
    output, end = [], 0
    for token, stop in _contexts(content):
        output.append(content[end:token.start()])
        fragment = content[token.start():stop]
        if any(token.group(kind) for kind in ("code", "destination", "url", "autolink", "path")):
            fragment = re.sub(r"[^\r\n]", " ", fragment)
        elif token.group("latex") and re.fullmatch(r"\\(?:href|url|nolinkurl|path|includegraphics)\*?", token[0]):
            cursor = token.end() - token.start()
            while cursor < len(fragment):
                while cursor < len(fragment) and fragment[cursor].isspace():
                    cursor += 1
                if cursor < len(fragment) and fragment[cursor] == "[":
                    cursor = _balanced_end(fragment, cursor)
                else:
                    break
            if cursor < len(fragment) and fragment[cursor] == "{":
                argument_end = _balanced_end(fragment, cursor)
                fragment = fragment[:cursor + 1] + re.sub(r"[^\r\n]", " ", fragment[cursor + 1:argument_end - 1]) + fragment[argument_end - 1:]
        output.append(fragment)
        end = stop
    output.append(content[end:])
    return re.sub(r"<!--.*?-->", lambda match: re.sub(r"[^\r\n]", " ", match[0]), "".join(output), flags=re.S)


def _symbols(content: str):
    text = _without_code(content)
    definitions: dict[str, list[dict[str, Any]]] = {}
    references: dict[str, list[dict[str, Any]]] = {}
    for pattern in (r"\{[^{}\n]*#((?:fig|tab|eq|sec):[^}\s]+)[^}]*\}", r"\\label\{((?:fig|tab|eq|sec):[^}]+)\}"):
        for match in re.finditer(pattern, text):
            identifier = match[1].strip()
            definitions.setdefault(identifier, []).append({"full_match": match[0], "line_number": text.count("\n", 0, match.start()) + 1})
    patterns = (r"\\(?:[cC]|eq|auto)?ref\{([^}]+)\}", r"(?<![\w\\])@((?:fig|tab|eq|sec):[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?)")
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            for identifier in match[1].split(","):
                references.setdefault(identifier.strip(), []).append({"full_match": match[0], "line_number": text.count("\n", 0, match.start()) + 1})
    return text, definitions, references


class CrossReferenceValidator:
    """Validate actual definitions/usages across an assembled document, ignoring code examples."""

    def __init__(self):
        self.figure_definitions = {}
        self.figure_references = {}
        self.validation_results = {}

    def validate_document(self, content: str, file_path: str | None = None) -> ValidationResult:
        text, definitions, references = _symbols(content)
        issues = []
        warnings = []
        for identifier, occurrences in definitions.items():
            if len(occurrences) > 1:
                issues.append(f"Duplicate definition: {identifier} on lines {[item['line_number'] for item in occurrences]}")
            if identifier not in references:
                warnings.append(f"Unused definition: {identifier}")
        for identifier, uses in references.items():
            if identifier not in definitions:
                issues.extend(f"Broken reference: {use['full_match']} on line {use['line_number']} - no definition found for '{identifier}'" for use in uses)
        for match in re.finditer(r"!\[([^\]]*)\]\([^)]+\)", text):
            if not match[1].strip():
                issues.append(f"Figure has no caption on line {text.count(chr(10), 0, match.start()) + 1}")
        # Project heading syntax and native Pandoc image/table IDs are both real targets.
        for match in re.finditer(r"^\s*##\s+Figure:\s+([^\n]+)$", text, re.M):
            if not re.search(r"\{#fig:[^}]+\}", match[1]):
                issues.append("Malformed figure definition: missing {#fig:id}")
        self.figure_definitions = self._extract_figure_definitions(content)
        self.figure_references = self._extract_figure_references(content)
        self.validation_results = ValidationResult({
            "valid": not issues, "issues": issues, "warnings": warnings,
            "figure_definitions": len(self.figure_definitions),
            "figure_references": sum(map(len, self.figure_references.values())),
            "definitions": sorted(definitions), "references": sorted(references), "file_path": file_path,
        })
        return self.validation_results

    def _extract_figure_definitions(self, content: str) -> dict[str, dict[str, Any]]:
        text, definitions, _ = _symbols(content)
        result = {}
        for full_id, records in definitions.items():
            if full_id.startswith("fig:"):
                record = dict(records[0])
                line = text.splitlines()[record["line_number"] - 1]
                record["title"] = re.sub(r"\{#fig:[^}]+\}", "", line).strip()
                result[full_id[4:]] = record
        return result

    def _extract_figure_references(self, content: str) -> dict[str, list[dict[str, Any]]]:
        _, _, references = _symbols(content)
        return {identifier[4:]: records for identifier, records in references.items() if identifier.startswith("fig:")}


def validate_cross_references(content: str, file_path: str | None = None) -> ValidationResult:
    """Return coherent undefined/duplicate-reference diagnostics for a whole document."""
    return CrossReferenceValidator().validate_document(content, file_path)


def fix_figure_ids(content: str, id_mapping: dict[str, str] | None = None) -> str:
    """Apply an explicit ID mapping; unresolved references require an author-supplied target."""
    if id_mapping is None:
        result = validate_cross_references(content)
        if not result["valid"]:
            raise ValueError("Reference repair requires an explicit, reviewed id_mapping")
        return content
    for old, new in id_mapping.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", old) or not re.fullmatch(r"[A-Za-z0-9_.-]+", new):
            raise ValueError("Figure mappings use bare ordinary identifiers")
    mask = _without_code(content)
    edits = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])fig:([A-Za-z0-9_.-]+)", mask):
        if match[1] in id_mapping:
            edits.append((match.start(1), match.end(1), id_mapping[match[1]]))
    for start, end, value in reversed(edits):
        content = content[:start] + value + content[end:]
    return content


def generate_cross_reference_report(content: str, file_path: str | None = None) -> str:
    """Render the actual validator result without implying any repairs were applied."""
    result = validate_cross_references(content, file_path)
    lines = ["# Cross-Reference Validation Report", f"Status: {'VALID' if result['valid'] else 'ISSUES FOUND'}",
             f"Figure Definitions: {result['figure_definitions']}", f"Figure References: {result['figure_references']}"]
    if file_path:
        lines.append(f"File: {file_path}")
    for kind in ("issues", "warnings"):
        if result[kind]:
            lines.extend(["", "## " + kind.title(), *["- " + value for value in result[kind]]])
    return "\n".join(lines)
