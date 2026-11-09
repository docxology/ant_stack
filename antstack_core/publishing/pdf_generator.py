"""Document-scoped Pandoc/XeLaTeX publishing with atomic PDF publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import unquote, urlparse

from ..figures.mermaid import preprocess_mermaid_diagrams
from ..mathematics import UNICODE_TO_LATEX, normalize_unicode_math
from .provenance import file_identity, publication_path, publish_file

logger = logging.getLogger(__name__)


@dataclass
class PDFGenerationConfig:
    """Renderer settings; custom templates are explicit, otherwise Pandoc owns its template."""

    pandoc_path: str = "pandoc"
    pandoc_args: list[str] = field(default_factory=list)
    latex_engine: str = "xelatex"
    latex_args: list[str] = field(default_factory=lambda: ["-interaction=nonstopmode", "-synctex=1"])
    enable_mermaid: bool = True
    enable_crossrefs: bool = True
    enable_unicode_conversion: bool = True
    validate_before_build: bool = True
    template_dir: Path | None = None
    temp_dir: Path | None = None

    def __post_init__(self):
        if self.pandoc_args is None:
            self.pandoc_args = []
        if self.latex_args is None:
            self.latex_args = ["-interaction=nonstopmode", "-synctex=1"]


@dataclass
class BuildResult:
    """Observed renderer result; success requires a newly produced PDF."""

    success: bool
    output_path: Path | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    build_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class PDFGenerator:
    """Render ordered sources as one document; never guess or retarget a reference."""

    def __init__(self, config: PDFGenerationConfig | None = None):
        self.config = config or PDFGenerationConfig()

    def generate_pdf(self, input_files: list[Path], output_path: Path,
                     metadata: dict[str, Any] | None = None) -> BuildResult:
        started = time.monotonic()
        try:
            if not input_files:
                raise ValueError("No input files provided")
            sources = [Path(path).absolute() for path in input_files]
            for source in sources:
                if not source.is_file():
                    raise FileNotFoundError(f"Input file not found: {source}")
            output = publication_path(output_path)
            if output in sources:
                raise ValueError("PDF output must not replace an input source")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary_parent = publication_path(self.config.temp_dir if self.config.temp_dir else output.parent)
            temporary_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".antstack-pdf-", dir=temporary_parent) as directory:
                workspace = Path(directory)
                document, hashes, resources = self._prepare_document(sources, workspace)
                code_font = self._default_code_font()
                resources[str(code_font)] = file_identity(code_font)["sha256"]
                staged_input = workspace / "document.json"
                staged_input.write_text(json.dumps(document), encoding="utf-8")
                staged_output = workspace / "document.pdf"
                header = workspace / "references.tex"
                header.write_text(
                    "\\usepackage{hyperref}\n\\usepackage{cleveref}\n"
                    "\\usepackage{fvextra}[2024/05/16]\n"
                    "\\fvset{breaklines=true,breakanywhere=true,breaknonspaceingroup=true,"
                    "breaksymbolleft={},breakanywheresymbolpre={}}\n",
                    encoding="utf-8",
                )
                command = self._build_pandoc_command([staged_input], staged_output, metadata or {})
                command.extend(["--from=json", "--include-in-header", str(header)])
                completed = subprocess.run(command, cwd=workspace, stdin=subprocess.DEVNULL,
                                           capture_output=True, text=True, timeout=300)
                warnings = self._parse_warnings(completed.stderr)
                if completed.returncode:
                    raise RuntimeError(completed.stderr or completed.stdout or "Pandoc failed")
                fatal = re.compile(r"undefined references|(?:reference|citation).*?(?:undefined|not found)|"
                                   r"could not (?:fetch resource|load image)|missing character:", re.I)
                if any(fatal.search(line) for line in completed.stderr.splitlines()):
                    raise RuntimeError("Unresolved renderer diagnostics:\n" + completed.stderr)
                if not staged_output.is_file() or not staged_output.read_bytes().startswith(b"%PDF-"):
                    raise RuntimeError("Renderer returned success without producing a PDF")
                publish_file(staged_output, output)
                return BuildResult(True, output_path=output, warnings=warnings,
                                   build_time_seconds=time.monotonic() - started,
                                   metadata={"source_files": [str(p) for p in sources], "input_sha256": hashes,
                                             "resource_sha256": resources})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            logger.error("PDF generation failed: %s", error)
            return BuildResult(False, error_message=str(error), build_time_seconds=time.monotonic() - started)

    def _parse_markdown(self, content: str) -> dict[str, Any]:
        completed = subprocess.run(
            [self.config.pandoc_path, "--from=markdown+tex_math_dollars+tex_math_single_backslash+raw_tex+implicit_figures", "--to=json"],
            input=content, capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(completed.stdout)

    def _prepare_document(self, sources: list[Path], workspace: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        hashes, resources, markers, chunks = {}, {}, {}, []
        for index, source in enumerate(sources):
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            hashes[str(source)] = digest
            marker = f"<!-- antstack-source-{index}-{digest} -->"
            markers[marker] = source
            chunks.extend((marker, data.decode("utf-8")))
        # One reader invocation resolves document-wide notes and reference definitions.
        # Removable source boundaries retain each image use's local resource context.
        document = self._parse_markdown("\n\n".join(chunks))
        blocks, source = [], sources[0]
        for block in document["blocks"]:
            if block.get("t") == "RawBlock" and block["c"][0] == "html":
                marker = block["c"][1].strip()
                if marker in markers:
                    source = markers[marker]
                    continue
            blocks.append(self._prepare_nodes(block, source, workspace, resources))
        document["blocks"] = self._bind_reference_headers(blocks)
        if self.config.enable_crossrefs and self.config.validate_before_build:
            self._validate_references(document)
        return document, hashes, resources

    def _prepare_nodes(self, value, source: Path, workspace: Path, resources: dict[str, str]):
        if isinstance(value, list):
            transformed = [self._prepare_nodes(item, source, workspace, resources) for item in value]
            # Project reference tokens may have literal braces around a Pandoc Cite.
            for index, item in enumerate(transformed):
                if isinstance(item, dict) and item.get("t") == "RawInline" and item["c"][1].startswith("\\ref{"):
                    if index and transformed[index - 1].get("t") == "Str":
                        transformed[index - 1]["c"] = transformed[index - 1]["c"].removesuffix("{")
                    if index + 1 < len(transformed) and isinstance(transformed[index + 1], dict) and transformed[index + 1].get("t") == "Str":
                        transformed[index + 1]["c"] = transformed[index + 1]["c"].removeprefix("}")
            return transformed
        if not isinstance(value, dict):
            return value
        kind = value.get("t")
        if self.config.enable_unicode_conversion and kind == "Str":
            text = value["c"]
            pieces, start = [], 0
            for index, char in enumerate(text):
                macro = UNICODE_TO_LATEX.get(char)
                if macro is not None:
                    if start < index:
                        pieces.append({"t": "Str", "c": text[start:index]})
                    pieces.append({"t": "Math", "c": [{"t": "InlineMath"}, macro]})
                    start = index + 1
            if pieces:
                if start < len(text):
                    pieces.append({"t": "Str", "c": text[start:]})
                return {"t": "Span", "c": [["", [], []], pieces]}
        if self.config.enable_unicode_conversion and kind == "Math":
            value["c"][1] = self._convert_unicode_symbols("$" + value["c"][1] + "$")[1:-1]
        if kind == "CodeBlock" and "mermaid" in value["c"][0][1] and self.config.enable_mermaid:
            markdown = "```mermaid\n" + value["c"][1] + "\n```"
            rendered = preprocess_mermaid_diagrams(markdown, workspace / "mermaid", clean_existing=False)
            if "```mermaid" in rendered:
                raise RuntimeError("Mermaid rendering failed; a local Mermaid CLI and browser are required")
            replacement = self._parse_markdown(rendered)["blocks"]
            if len(replacement) != 1:
                raise RuntimeError("Mermaid renderer did not produce one figure")
            return self._prepare_nodes(replacement[0], source, workspace, resources)
        if kind in ("CodeBlock", "Code"):
            return value
        if kind == "Image":
            url = value["c"][2][0]
            parsed = urlparse(url)
            if parsed.scheme not in ("", "file") or parsed.netloc:
                raise ValueError(f"Images must be explicit local files, not remote resources: {url}")
            path = Path(unquote(parsed.path))
            candidates = [path] if path.is_absolute() else [source.parent / path]
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"Image not found in {source}: {url}")
            if not resolved.is_relative_to(workspace):
                snapshots = workspace / "resources"
                snapshots.mkdir(exist_ok=True)
                snapshot = snapshots / (hashlib.sha256(str(resolved).encode()).hexdigest() + resolved.suffix.lower())
                if not snapshot.exists():
                    digest = hashlib.sha256()
                    with resolved.open("rb") as incoming, snapshot.open("xb") as outgoing:
                        while chunk := incoming.read(1024 * 1024):
                            digest.update(chunk)
                            outgoing.write(chunk)
                    resources[str(resolved)] = digest.hexdigest()
                resolved = snapshot
            if resolved.suffix.lower() == ".svg":
                converter = shutil.which("rsvg-convert")
                if converter is None:
                    raise RuntimeError("SVG figures require rsvg-convert")
                converted = workspace / (hashlib.sha256(str(resolved).encode()).hexdigest() + ".png")
                if not converted.exists():
                    subprocess.run([converter, "-o", str(converted), str(resolved)], stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, check=True, timeout=30)
                resolved = converted
            value["c"][2][0] = str(resolved)
        if kind == "Cite":
            citations = value["c"][0]
            if len(citations) == 1 and citations[0]["citationId"].startswith(("fig:", "tab:", "eq:", "sec:")):
                return {"t": "RawInline", "c": ["latex", "\\ref{" + citations[0]["citationId"] + "}"]}
        return {key: self._prepare_nodes(item, source, workspace, resources) for key, item in value.items()}

    def _bind_reference_headers(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        pending = None
        for block in blocks:
            kind = block.get("t")
            if kind == "Header" and block["c"][1][0].startswith(("fig:", "tab:")):
                if pending:
                    raise ValueError(f"Reference heading has no following figure/table: {pending[0]}")
                pending = (block["c"][1][0], block["c"][2])
                continue
            if pending:
                expected = "Figure" if pending[0].startswith("fig:") else "Table"
                if kind == "Para" and expected == "Figure" and len(block["c"]) == 1 and block["c"][0].get("t") == "Image":
                    image = block["c"][0]
                    block = {"t": "Figure", "c": [["", [], []], [None, [{"t": "Plain", "c": image["c"][1]}]], [block]]}
                    kind = "Figure"
                if kind == expected:
                    if block["c"][0][0] not in ("", pending[0]):
                        raise ValueError("Conflicting reference heading and figure/table identifier")
                    block["c"][0][0] = pending[0]
                    title = pending[1]
                    if title and title[0].get("t") == "Str" and title[0]["c"] in ("Figure:", "Table:"):
                        title = title[1:]
                        if title and title[0].get("t") == "Space":
                            title = title[1:]
                    caption = block["c"][1][1]
                    if not caption:
                        block["c"][1][1] = [{"t": "Plain", "c": title}]
                    elif caption[0].get("t") in ("Plain", "Para") and caption[0]["c"] != title:
                        caption[0]["c"] = title + [{"t": "Str", "c": ":"}, {"t": "Space"}] + caption[0]["c"]
                    pending = None
                elif kind == "Header":
                    raise ValueError(f"Reference heading has no following {expected.lower()}: {pending[0]}")
            result.append(block)
        if pending:
            raise ValueError(f"Reference heading has no following figure/table: {pending[0]}")
        return result

    def _validate_references(self, document: dict[str, Any]) -> None:
        definitions = {}
        references = set()
        native = {"fig": "Figure", "tab": "Table", "sec": "Header"}
        raw_environments = {"fig": {"figure"}, "tab": {"table"},
                            "eq": {"equation", "align", "gather", "multline", "eqnarray"}}
        def define(identifier, kind):
            prefix = identifier.split(":", 1)[0]
            if prefix not in ("fig", "tab", "eq", "sec"):
                return
            if kind != "raw" and native.get(prefix) != kind:
                raise ValueError(f"{identifier} is not attached to its actual {prefix} counter")
            definitions[identifier] = definitions.get(identifier, 0) + 1
        def inspect(value):
            if isinstance(value, list):
                for item in value:
                    inspect(item)
            elif isinstance(value, dict):
                kind = value.get("t")
                if kind in ("CodeBlock", "Code"):
                    return
                if kind in ("RawInline", "RawBlock") and value["c"][0] not in ("tex", "latex"):
                    return
                if kind in ("Figure", "Table", "Div", "Span", "Image"):
                    define(value["c"][0][0], kind)
                elif kind == "Header":
                    define(value["c"][1][0], kind)
                if kind in ("RawInline", "RawBlock", "Math"):
                    text = value["c"][1]
                    environments = list(re.finditer(r"\\begin\{(figure|table|equation|align|gather|multline|eqnarray)\}(.*?)\\end\{\1\}", text, re.S))
                    for label in re.finditer(r"\\label\{([^}]+)\}", text):
                        identifier = label[1]
                        prefix = identifier.split(":", 1)[0]
                        if prefix in raw_environments:
                            hosts = [host for host in environments if host.start() < label.start() < host.end() and host[1] in raw_environments[prefix]]
                            if kind == "Math" or not hosts:
                                raise ValueError(f"{identifier} requires a numbered {prefix} environment, not an unnumbered anchor")
                            if prefix in ("fig", "tab") and not re.search(r"\\caption\b", text[hosts[0].start():label.start()]):
                                raise ValueError(f"{identifier} must follow its caption counter")
                        elif prefix == "sec" and not re.search(r"\\(?:sub)*section\b", text[:label.start()]):
                            raise ValueError(f"{identifier}: use an identifier on the actual numbered heading")
                        define(identifier, "raw")
                    for group in re.findall(r"\\(?:[cC]|eq|auto)?ref\{([^}]+)\}", text):
                        references.update(identifier.strip() for identifier in group.split(","))
                for item in value.values():
                    inspect(item)
        inspect(document)
        duplicate = sorted(name for name, count in definitions.items() if count > 1)
        missing = sorted(references - definitions.keys())
        if duplicate or missing:
            raise ValueError(f"Cross-reference validation failed: duplicate={duplicate}; undefined={missing}")

    def _convert_unicode_symbols(self, content: str) -> str:
        return normalize_unicode_math(content)

    @staticmethod
    def _default_code_font() -> Path:
        from matplotlib import get_data_path
        return Path(get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"

    def _build_pandoc_command(self, input_files: list[Path], output_path: Path,
                              metadata: dict[str, Any]) -> list[str]:
        command = [self.config.pandoc_path, *map(str, input_files), "-o", str(output_path),
                   "--pdf-engine", self.config.latex_engine, "--number-sections",
                   "--variable", "geometry:margin=1in", "--variable", "fontsize=11pt",
                   "--variable", "colorlinks=true", "--variable", f"monofont:{self._default_code_font()}",
                   "--variable", "monofontoptions:NFSSFamily=AntStackCode"]
        for argument in self.config.latex_args:
            command.extend(["--pdf-engine-opt", argument])
        command.extend(self.config.pandoc_args)
        if self.config.template_dir is not None:
            template = Path(self.config.template_dir) / "default.latex"
            if not template.is_file():
                raise FileNotFoundError(f"Explicit template not found: {template}")
            command.extend(["--template", str(template.absolute())])
        for key, value in metadata.items():
            command.extend(["--metadata", f"{key}:{value}"])
        return command

    def _parse_warnings(self, stderr: str) -> list[str]:
        return [line.strip() for line in stderr.splitlines()
                if re.search(r"warning|(?:over|under)full \\hbox|undefined|missing character", line, re.I)]

    def validate_build_environment(self) -> dict[str, bool]:
        results = {}
        for name, executable in (("pandoc", self.config.pandoc_path), ("latex", self.config.latex_engine)):
            try:
                results[name] = subprocess.run([executable, "--version"], stdin=subprocess.DEVNULL,
                                               capture_output=True, timeout=10).returncode == 0
            except (OSError, subprocess.SubprocessError):
                results[name] = False
        lookup = shutil.which("kpsewhich")
        results["latex_packages"] = bool(lookup) and all(
            subprocess.run([lookup, package], stdin=subprocess.DEVNULL, capture_output=True, timeout=10).returncode == 0
            for package in ("amsmath.sty", "cleveref.sty", "fvextra.sty")
        )
        return results


def generate_publication_pdf(input_files: list[Path], output_path: Path,
                             title: str = "", author: str = "") -> BuildResult:
    """Generate a publication PDF with the default document-scoped renderer."""
    metadata = {key: value for key, value in (("title", title), ("author", author)) if value}
    return PDFGenerator().generate_pdf(input_files, output_path, metadata)


def validate_pdf_environment() -> dict[str, Any]:
    """Observe executable/package availability; this is not an actual render."""
    validation = PDFGenerator().validate_build_environment()
    recommendations = [f"Install or repair {name}" for name, valid in validation.items() if not valid]
    return {"validation": validation, "recommendations": recommendations, "all_valid": all(validation.values())}
