#!/usr/bin/env python3
"""Stage the architecture paper and its source-bound diagrams for Template."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
MERMAID = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
SCHEMATIC = "ant_stack_graphical_abstract.png"
SCHEMATIC_LABEL = "fig:ant_stack_graphical_abstract"


def main() -> None:
    source = ROOT / "docs/manuscript"
    paper = source / "ant_stack"
    config_path = source / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    native = yaml.safe_load((paper / "paper_config.yaml").read_text(encoding="utf-8"))
    sections = config["front_matter"]["files"]
    if sections != [f"ant_stack/{name}" for name in native["content"]["files"]]:
        raise ValueError("Template and native architecture section order differ")
    output = ROOT / "output"
    hydrated = output / "manuscript"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (hydrated / "ant_stack").mkdir(parents=True, exist_ok=True)
    (hydrated / "config.yaml").write_text(
        "# Generated manuscript ordering\n" + config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    shutil.copyfile(source / "preamble.md", hydrated / "preamble.md")

    registry = {
        SCHEMATIC_LABEL: {
            "filename": SCHEMATIC,
            "alt_text": "Proposed Ant Stack interfaces and implemented analytical boundary",
        },
    }
    shutil.copyfile(source / "graphical_abstracts" / SCHEMATIC, figures / SCHEMATIC)

    for name in sections:
        def rasterize(match: re.Match[str]) -> str:
            diagram = match[1].strip()
            digest = hashlib.sha256(diagram.encode()).hexdigest()[:16]
            filename = f"diagram_{digest}.png"
            label = f"fig:architecture-{digest}"
            title = re.search(r"^\s*accTitle:\s*(.+)$", diagram, re.M)
            if title is None:
                raise ValueError(f"{name}: diagram requires an accessible title")
            alt = title[1].strip()
            shutil.copyfile(paper / "assets/mermaid" / filename, figures / filename)
            registry[label] = {"filename": filename, "alt_text": alt}
            return f"![{alt}](../../figures/{filename}){{#{label} width=95%}}"

        text = (source / name).read_text(encoding="utf-8")
        text = MERMAID.sub(rasterize, text)
        if name == "ant_stack/Abstract.md":
            text = text.replace(f" {{#{SCHEMATIC_LABEL}}}", "")
            text = text.replace(
                f"](../graphical_abstracts/{SCHEMATIC}){{width=100%}}",
                f"](../../figures/{SCHEMATIC}){{#{SCHEMATIC_LABEL} width=100%}}",
            )
        (hydrated / name).write_text(text, encoding="utf-8")

    (figures / "figure_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
