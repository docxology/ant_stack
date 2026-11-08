"""Local Mermaid rendering; remote rendering requires an explicit service URL."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ValidationResult(dict):
    """Mapping also unpackable as ``valid, issues``."""

    def __iter__(self):
        yield self.get("valid", False)
        yield self.get("issues", [])


def validate_mermaid_syntax(mermaid_code: str) -> ValidationResult:
    """Perform lexical checks; only the real renderer establishes renderability."""
    wrapped = re.search(r"```mermaid[^\n]*\n(.*?)\n\s*```", mermaid_code, re.S)
    code = wrapped[1] if wrapped else mermaid_code
    issues = []
    if not code.strip():
        issues.append("Empty Mermaid code")
    elif not re.match(r"\s*(?:graph|flowchart|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|gantt|pie|erDiagram)\b", code):
        issues.append("Missing diagram type declaration")
    if code.count("[") != code.count("]"):
        issues.append("Unbalanced square brackets")
    if code.count("{") != code.count("}"):
        issues.append("Unbalanced braces")
    suggestions = []
    if re.search(r"\\[A-Za-z]+|\$", code):
        suggestions.append("Use readable Mermaid labels rather than LaTeX commands")
    return ValidationResult(valid=not issues, issues=issues, suggestions=suggestions, line_count=len(code.splitlines()))


def sanitize_mermaid_for_rendering(mermaid_code: str) -> str:
    """Normalize label math without deleting edge labels or changing graph structure."""
    code = mermaid_code.strip()
    replacements = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'Δ': 'Delta', 'δ': 'delta',
                    'η': 'eta', 'μ': 'mu', 'ρ': 'rho', '·': '*', '×': 'x', '𝒪': 'O', '𝑂': 'O'}
    for symbol, replacement in replacements.items():
        code = code.replace(symbol, replacement)
    code = re.sub(r"\\(?:texttt|mathrm|text)\{([^}]*)\}", lambda match: match[1], code)
    code = code.replace(r"\,", " ").replace(r"\(", "").replace(r"\)", "")
    return re.sub(r"\$([^$]*)\$", lambda match: match[1], code)


def preprocess_mermaid_diagrams(markdown_content: str, output_dir: Path | str | None = None,
                               img_format: str = "png", kroki_url: str | None = None,
                               clean_existing: bool = False) -> str:
    """Render active fenced diagrams locally; failed producers never reuse an old image.

    Caller owns the dedicated cache namespace. Default cleanup is off. No implicit
    remote service is used; distributed source should retain its original blocks.
    """
    from ..mathematics import _contexts
    from ..publishing.provenance import publication_path, publish_file
    if type(clean_existing) is not bool:
        raise ValueError("clean_existing must be a boolean")
    if img_format not in ("png", "svg"):
        raise ValueError("Mermaid image format must be png or svg")
    if output_dir is None:
        return markdown_content
    blocks = []
    for token, stop in _contexts(markdown_content):
        fence = token.group("fence") if token.group("code") else None
        if fence is None:
            continue
        fragment = markdown_content[token.start():stop]
        opening, _, body = fragment.partition("\n")
        info = opening.lstrip()[len(fence):].strip().split(maxsplit=1)
        if not info or info[0] != "mermaid":
            continue
        closing = re.search(r"(?:^|\n)[ ]{0,3}" + re.escape(fence) + re.escape(fence[0]) + r"*[ \t]*\Z", body)
        code = sanitize_mermaid_for_rendering(body[:closing.start()] if closing else body)
        if not code:
            raise ValueError("Empty Mermaid code")
        blocks.append((token.start(), stop, code))
    if not blocks:
        return markdown_content
    directory = publication_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if clean_existing:
        for path in directory.iterdir():
            if re.fullmatch(r"diagram_[0-9a-f]{8,64}\.(?:mmd|png|svg)", path.name):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Unsafe Mermaid cache entry: {path}")
                path.unlink()
    output, end = [], 0
    for start, stop, code in blocks:
        name = "diagram_" + hashlib.sha256(code.encode()).hexdigest()[:16]
        source, image = directory / (name + ".mmd"), directory / (name + "." + img_format)
        if source.is_symlink() or image.is_symlink():
            raise ValueError("Symlink Mermaid output is not allowed")
        with tempfile.TemporaryDirectory(prefix=".antstack-mermaid-", dir=directory) as temporary:
            staged_source = Path(temporary) / "diagram.mmd"
            staged_image = Path(temporary) / ("diagram." + img_format)
            staged_source.write_text(code, encoding="utf-8")
            if not _render_mermaid_diagram(str(staged_source), str(staged_image), kroki_url) or not _is_valid_image(str(staged_image)):
                raise RuntimeError("Mermaid rendering failed; install local mmdc and configure a browser")
            publish_file(staged_source, source)
            publish_file(staged_image, image)
        output.extend([markdown_content[end:start], f"![{_generate_diagram_caption(code, name)}]({image.as_uri()}){{ width=90% }}"])
        end = stop
    output.append(markdown_content[end:])
    return "".join(output)


def _render_mermaid_diagram(mmd_path: str, output_path: str, kroki_url: str | None = None) -> bool:
    if _render_with_mmdc(mmd_path, output_path):
        return True
    return bool(kroki_url) and _render_with_kroki(mmd_path, output_path, kroki_url)


def _render_with_mmdc(mmd_path: str, output_path: str) -> bool:
    explicit = os.environ.get("ANTSTACK_MMDC")
    local = Path.cwd() / "node_modules/.bin/mmdc"
    executable = explicit or shutil.which("mmdc") or (str(local) if local.is_file() else None)
    if not executable:
        return False
    browser = os.environ.get("PUPPETEER_EXECUTABLE_PATH")
    if not browser:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        browser = str(chrome) if chrome.is_file() else shutil.which("chromium") or shutil.which("chromium-browser")
    configuration = None
    try:
        command = [executable, "-i", mmd_path, "-o", output_path, "-b", "transparent"]
        descriptor, name = tempfile.mkstemp(prefix="browser-", suffix=".json", dir=Path(output_path).parent)
        configuration = Path(name)
        options = {"pipe": True}
        if browser:
            options["executablePath"] = browser
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(options, stream)
        command.extend(["-p", str(configuration)])
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
        if result.returncode:
            logger.warning("Local Mermaid renderer failed: %s", result.stderr.strip())
            return False
        return _is_valid_image(output_path)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Local Mermaid renderer unavailable: %s", error)
        return False
    finally:
        if configuration:
            configuration.unlink(missing_ok=True)


def _render_with_kroki(mmd_path: str, output_path: str, kroki_url: str) -> bool:
    """Use only an explicitly supplied endpoint; no remote fallback is enabled by default."""
    parsed = urlparse(kroki_url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Kroki requires an explicit credential-free HTTP(S) service URL")
    image_format = "svg" if output_path.endswith(".svg") else "png"
    request = urllib.request.Request(kroki_url.rstrip("/") + "/mermaid/" + image_format,
                                     data=Path(mmd_path).read_bytes(), headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(16 * 1024 * 1024 + 1)
        if len(content) > 16 * 1024 * 1024:
            return False
        Path(output_path).write_bytes(content)
        return _is_valid_image(output_path)
    except (OSError, ValueError):
        return False


def _is_valid_image(path: str) -> bool:
    try:
        file = Path(path)
        if file.suffix.lower() == ".svg":
            return ET.parse(file).getroot().tag.split("}")[-1] == "svg"
        from PIL import Image
        with Image.open(file) as image:
            image.verify()
            return image.format == "PNG"
    except (OSError, ValueError, ET.ParseError):
        return False


def _generate_diagram_caption(diagram_content: str, diagram_id: str) -> str:
    return "Mermaid diagram " + diagram_id.removeprefix("diagram_")
