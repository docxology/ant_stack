#!/usr/bin/env python3
"""Produce original conceptual graphical abstracts; never plot model results.

Usage (from this repository, with an existing staging parent)::

    uv run python tools/generate_graphical_abstracts.py --output .staging/abstracts

The explicitly selected repository-local destination MUST NOT exist. SVG and
PNG are exported from each same native Matplotlib figure. No browser, network,
model execution, empirical inputs, random sampling, or new dependencies are used.
A deterministic JSON receipt binds captions, semantic sources, producer bytes,
software versions, and generated asset hashes. Reproducibility is byte-level
within the same Python/Matplotlib/font environment, not across library versions.

Figures use the object-oriented Agg API, never enter pyplot's global registry,
and are cleared in finally blocks. Plot settings are confined to rc_context;
no global backend is selected or changed. This is not a Mermaid fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PRODUCER = "tools/generate_graphical_abstracts.py"
SCHEMA = "ant-stack-original-graphical-abstracts/v1"
HASH_SALT = "ant-stack-original-conceptual-abstracts-v1"
RECEIPT = "graphical_abstracts.provenance.json"
DPI = 300

# Dark text on light surfaces; evidence classes also differ by labels and lines.
INK = "#142D3D"
MUTED = "#43596A"
RULE = "#B9C6CE"
BLUE = "#175A86"
BLUE_LIGHT = "#EDF5FA"
TEAL = "#17665E"
TEAL_LIGHT = "#EDF7F4"
PURPLE = "#644A82"
PURPLE_LIGHT = "#F4F0F8"
NEUTRAL = "#F3F5F7"

CAPTIONS = {
    "ant_stack_graphical_abstract": (
        "Ant Stack as a proposed Body–Brain–Mind architecture (dashed cards), "
        "distinct from implemented analytical workload and energy models (solid cards). "
        "Arrows describe intended information flow, not an executed sensorimotor loop. "
        "Counter models are not contact, neural, or active-inference solvers and do not "
        "establish biological fidelity, real-time performance, or energy efficiency."
    ),
    "complexity_energetics_graphical_abstract": (
        "Analytical complexity-to-energy accounting: declared assumptions produce "
        "workload counters, which are multiplied by illustrative energy prices. "
        "Module-isolated rows and a shared-system sum have different boundaries; "
        "a system sum requires a matched interval and one idle baseline. "
        "The schematic contains no numerical results or empirical calibration."
    ),
}

# Section/symbol references remain meaningful if manuscript line numbers move.
SEMANTIC_SOURCES = (
    {
        "path": "papers/ant_stack/Abstract.md",
        "references": ["Abstract"],
        "supports": "Three proposed layers; architecture and analytical estimates are not empirical validation.",
        "evidence_class": "qualified design proposal",
    },
    {
        "path": "papers/ant_stack/AntBody.md",
        "references": ["Proposed Interfaces (I/O Contract)", "Morphology and Biomechanics", "Sensory Apparatus"],
        "supports": "Proposed morphology, actuation, sensing, observation and action interfaces.",
        "evidence_class": "design targets, not an implemented dynamics solver",
    },
    {
        "path": "papers/ant_stack/AntBrain.md",
        "references": ["Scope and Assumptions", "Integration", "Proposed Implementation Choices and Timing Targets"],
        "supports": "Proposed insect-inspired sensory, memory and navigation abstractions; no ant connectome validation.",
        "evidence_class": "functional design assumptions",
    },
    {
        "path": "papers/ant_stack/AntMind.md",
        "references": ["Scope and Assumptions", "Proposed Single-Agent Generative Model", "Proposed Diffusion\u2013Decay Pheromone Field", "Technical Pointers and References"],
        "supports": "Proposed beliefs, policy selection and stigmergy; decision-step horizon; base/enhanced distinction.",
        "evidence_class": "conceptual cognitive design",
    },
    {
        "path": "papers/complexity_energetics/Abstract.md",
        "references": ["Abstract"],
        "supports": "Analytical framework, assumed coefficients, separate base/enhanced families, absent empirical calibration.",
        "evidence_class": "analytical scope statement",
    },
    {
        "path": "papers/complexity_energetics/Methods.md",
        "references": ["Methodological Framework and Evidence Classes", "Selected Base Workloads", "Separate Enhanced Models", "Energy Estimation and Time Accounting", "Descriptive Scaling Fits and Resampling Limits"],
        "supports": "Floor(f*t) updates, traffic units, exact tree nodes versus bounded leaf policies, nonadditive module rows.",
        "evidence_class": "analytical definitions and declared assumptions",
    },
    {
        "path": "papers/complexity_energetics/Energetics.md",
        "references": ["Compute, Memory, and Idle Energy", "AntBody: Mechanics and Sensing", "AntMind: Conceptual Layer and Physical Compute", "Integrated Per-Decision Accounting", "Units and Conversions"],
        "supports": "Joule conversions, matched time windows, physical terms, shared idle once, Mind physical compute retained.",
        "evidence_class": "accounting definitions, not observed energy flows",
    },
    {
        "path": "papers/complexity_energetics/manifest.example.yaml",
        "references": ["coefficients", "workloads", "meter"],
        "supports": "Illustrative input authority selects base closed_form workloads and a null meter; no numeric constants copied into diagrams.",
        "evidence_class": "illustrative configuration, not measurements",
    },
    {
        "path": "antstack_core/analysis/workloads.py",
        "references": ["body_workload_closed_form", "brain_workload_closed_form", "mind_workload_closed_form", "calculate_contact_complexity", "calculate_sparse_neural_complexity", "calculate_active_inference_complexity", "effective_policy_count", "enhanced_mind_workload_closed_form", "estimate_body_energy_mech"],
        "supports": "Implemented count formulas, operation/traffic/spike counters, base tree-node and enhanced leaf-policy conventions, assumed physical power integration.",
        "evidence_class": "implemented analytical models, not solver execution",
    },
    {
        "path": "antstack_core/analysis/energy.py",
        "references": ["ComputeLoad", "EnergyCoefficients", "estimate_detailed_energy", "estimate_compute_energy", "EnergyBreakdown", "pj_to_j", "aj_to_j"],
        "supports": "Counter units, coefficient units, pJ/aJ conversions, idle power times duration, joule-valued energy components.",
        "evidence_class": "implemented analytical energy accounting",
    },
    {
        "path": "antstack_core/orchestration/complexity_energetics.py",
        "references": ["CLOSED_FORM_WORKLOADS", "_load_for_manifest_mode", "run_manifest"],
        "supports": "Base-model dispatch and separate module-isolated versus common-window shared-idle report boundaries.",
        "evidence_class": "implemented analytical orchestration",
    },
)


def _text(ax: Axes, x: float, y: float, text: str, *, size: float = 11,
          color: str = INK, weight: str = "normal", ha: str = "left",
          va: str = "top") -> None:
    ax.text(x, y, text, fontsize=size, color=color, weight=weight,
            ha=ha, va=va, linespacing=1.55, clip_on=False)


def _card(ax: Axes, x: float, y: float, width: float, height: float,
          title: str, body: str, *, accent: str = BLUE,
          fill: str = BLUE_LIGHT, proposed: bool = False,
          size: float = 11) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.5, edgecolor=accent, facecolor=fill,
        linestyle=(0, (5, 3)) if proposed else "solid",
    ))
    _text(ax, x + 0.21, y + height - 0.22, title,
          size=14, weight="bold", color=accent)
    _text(ax, x + 0.21, y + height - 0.67, body, size=size)


def _arrow(ax: Axes, start: tuple[float, float],
           end: tuple[float, float], *, proposed: bool = False) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=MUTED, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 3)) if proposed else "solid",
    ))


def _canvas(title: str, subtitle: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(9, 6), dpi=DPI, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set(xlim=(0, 12), ylim=(0, 8))
    axes.set_axis_off()
    _text(axes, 0.5, 7.69, title, size=23, weight="bold")
    _text(axes, 0.5, 7.18, subtitle, size=11.5, color=MUTED)
    return figure, axes


def _architecture(axes: Axes) -> None:
    _text(axes, 0.5, 6.68, "PROPOSED DESIGN  /  intended information flow",
          size=10.5, weight="bold", color=MUTED)
    cards = (
        (0.5, "AntBody", "Morphology & actuation\nSensors ↔ environment\nObservations → AntBrain", BLUE, BLUE_LIGHT),
        (4.4, "AntBrain", "Sensing & memory\nNavigation abstractions\nState cues → AntMind", TEAL, TEAL_LIGHT),
        (8.3, "AntMind", "Beliefs & policy selection\nActive-inference proposal\nStigmergic coordination", PURPLE, PURPLE_LIGHT),
    )
    for x, title, body, accent, fill in cards:
        _card(axes, x, 4.25, 3.2, 2.1, title, body,
              accent=accent, fill=fill, proposed=True)
    _arrow(axes, (3.82, 5.23), (4.28, 5.23), proposed=True)
    _arrow(axes, (7.72, 5.23), (8.18, 5.23), proposed=True)
    # The return channel is outside all cards, so it cannot look like a counter.
    axes.plot([9.9, 9.9], [4.14, 3.83], color=MUTED, linewidth=1.3,
              linestyle=(0, (4, 3)))
    _arrow(axes, (9.9, 3.83), (2.1, 3.83), proposed=True)
    axes.plot([2.1, 2.1], [3.83, 4.14], color=MUTED, linewidth=1.3,
              linestyle=(0, (4, 3)))
    _text(axes, 6, 3.59, "Proposed feedback: policy context → motor targets → environment",
          size=11, ha="center", color=MUTED)
    axes.plot([0.5, 11.5], [3.12, 3.12], color=RULE, linewidth=1)
    _text(axes, 0.5, 2.87, "IMPLEMENTED  /  analytical models, not an embodied agent",
          size=10.5, weight="bold", color=MUTED)
    _card(axes, 0.5, 0.98, 3.2, 1.52, "Body counters",
          "Contact, sensing & control\nNo physical dynamics solved", size=10.5)
    _card(axes, 4.4, 0.98, 3.2, 1.52, "Brain counters",
          "Sparse work, traffic & spikes\nNo neural dynamics solved",
          accent=TEAL, fill=TEAL_LIGHT, size=10.5)
    _card(axes, 8.3, 0.98, 3.2, 1.52, "Mind counters",
          "Policy-related work & traffic\nNo policy decisions executed",
          accent=PURPLE, fill=PURPLE_LIGHT, size=10.5)
    _text(axes, 6, 0.61, "Counters + assumed energy prices → analytical estimates, not empirical validation.",
          size=11, ha="center", weight="bold")
    _text(axes, 6, 0.07, "Original conceptual schematic • no numerical results • base and enhanced models remain distinct",
          size=9.5, ha="center", va="bottom", color=MUTED)


def _accounting(axes: Axes) -> None:
    _text(axes, 0.5, 6.68, "ANALYTICAL WORKFLOW  /  assumptions → counts → estimated energy",
          size=10.5, weight="bold", color=MUTED)
    _card(axes, 0.5, 4.15, 3.2, 2.2, "1  Declare inputs",
          "Body: joints, contacts, sensors\nBrain: population, activity\nMind: branching, horizon\nSelect model family & interval", size=10.5)
    _card(axes, 4.4, 4.15, 3.2, 2.2, "2  Count workload",
          "FLOPs; SRAM / DRAM bytes\nSpikes; modeled updates\nBytes are traffic, not storage\nHorizon is in decision steps",
          accent=TEAL, fill=TEAL_LIGHT, size=10.5)
    _card(axes, 8.3, 4.15, 3.2, 2.2, "3  Apply prices",
          "FLOP price: pJ / FLOP\nTraffic prices: pJ / byte\nSpike price: aJ / spike\nConvert every term to joules",
          accent=PURPLE, fill=PURPLE_LIGHT, size=10.5)
    _arrow(axes, (3.82, 5.23), (4.28, 5.23))
    _arrow(axes, (7.72, 5.23), (8.18, 5.23))
    _text(axes, 6, 3.87,
          r"$n=\lfloor f t\rfloor$ updates; $f$ [Hz], $t$ [s].  "
          "Prices, activity and memory splits are assumed.",
          size=11, ha="center")
    _text(axes, 6, 3.39,
          r"$E_{\rm compute}=10^{-12}(N_F e_F+V_S e_S+V_D e_D)"
          r"+10^{-18}N_{\rm spk}e_{\rm spk}\quad[\mathrm{J}]$",
          size=15, ha="center")
    _text(axes, 6, 2.92,
          "Base Mind: exact tree nodes. Enhanced Mind: separate leaf-policy model, with optional cap.",
          size=10.5, ha="center", color=MUTED)
    axes.plot([0.5, 11.5], [2.55, 2.55], color=RULE, linewidth=1)
    _card(axes, 0.5, 0.76, 5.3, 1.54, "Module-isolated rows",
          "Compute + module idle; each row's own window\nNot additive shared-controller system energy\nRepeats do not create measurements",
          fill=NEUTRAL, accent=MUTED, size=10.5)
    _card(axes, 6.1, 0.76, 5.4, 1.54, "Shared system: matched window",
          "Body + Brain + Mind compute\nAdd actuation, sensing; count idle once\nMean power = energy / interval [W]",
          fill=TEAL_LIGHT, accent=TEAL, size=10.5)
    _text(axes, 6, 0.62,
          "Model estimates—not device calibration, biological validation or real-time proof.",
          size=10.5, ha="center", weight="bold")
    _text(axes, 6, 0.08, "Original analytical schematic • no numerical results • physical loads and compute use matched intervals",
          size=9, ha="center", va="bottom", color=MUTED)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_records(repository: Path) -> list[dict[str, Any]]:
    records = []
    for source in SEMANTIC_SOURCES:
        data = (repository / source["path"]).read_bytes()
        records.append({**source, "sha256": _sha256(data), "size_bytes": len(data)})
    return records


def _fresh_destination(repository: Path, requested: Path) -> Path:
    """Refuse existing roots and symlink traversal, including symlinked parents.

    The staging parent must already exist; creation is exclusive and never an
    overwrite option. Output ownership must remain exclusive while producing.
    """
    candidate = requested if requested.is_absolute() else repository / requested
    if ".." in candidate.parts:
        raise ValueError("--output must not contain '..' components")
    try:
        relative = candidate.relative_to(repository)
    except ValueError as error:
        raise ValueError("--output must be inside this repository") from error
    if not relative.parts:
        raise ValueError("--output must select a fresh staging subdirectory")
    current = repository
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"--output cannot traverse a symlink: {current}")
    if candidate.exists():
        raise ValueError("--output already exists; choose a fresh staging directory")
    if not candidate.parent.is_dir():
        raise ValueError("--output parent must already exist")
    implementation = repository / "antstack_core"
    for node in (candidate, *candidate.parents):
        if (node.name.casefold() in (".git", "node_modules") or (node / "pyvenv.cfg").is_file() or
                (node.exists() and node.samefile(implementation))):
            raise ValueError("--output cannot use Git, runtime or implementation namespaces")
    candidate.mkdir(exist_ok=False)
    return candidate


def generate(output: Path) -> Path:
    """Write four original assets and one provenance receipt to a fresh root.

    A receipt is written only after every requested export succeeds. Failures
    propagate; partial output is left for inspection, never labeled complete.
    """
    repository = Path(__file__).resolve().parents[1]
    sources = _source_records(repository)
    producer_bytes = (repository / PRODUCER).read_bytes()
    producer_hash = _sha256(producer_bytes)
    destination = _fresh_destination(repository, output)
    settings = dict(matplotlib.rcParamsDefault)
    settings.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "text.usetex": False,
        "svg.hashsalt": HASH_SALT,
        "svg.fonttype": "path",
        "path.simplify": False,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    # Do not ask rc_context to select or resolve any global backend.
    settings.pop("backend", None)
    inventory = []
    figures = (
        ("ant_stack_graphical_abstract", "Ant Stack: design ≠ implementation",
         "Proposed embodied architecture, with a separate implemented analytical model layer.", _architecture),
        ("complexity_energetics_graphical_abstract", "From complexity to an energy account",
         "Transparent assumptions and accounting boundaries; no empirical calibration.", _accounting),
    )
    with rc_context(rc=settings):
        for stem, title, subtitle, draw in figures:
            figure, axes = _canvas(title, subtitle)
            try:
                draw(axes)
                description = (
                    f"{CAPTIONS[stem]} Producer: {PRODUCER}; SHA-256: {producer_hash}. "
                    f"Semantic source hashes and output inventory: {RECEIPT}."
                )
                for extension in ("svg", "png"):
                    name = f"{stem}.{extension}"
                    metadata = (
                        {"Title": title, "Description": description,
                         "Creator": PRODUCER, "Date": None}
                        if extension == "svg" else
                        {"Title": title, "Description": description,
                         "Software": PRODUCER}
                    )
                    # Opening exclusively also protects against accidental filename collisions.
                    with (destination / name).open("xb") as stream:
                        figure.savefig(stream, format=extension, dpi=DPI,
                                       metadata=metadata, facecolor="white", transparent=False)
                    data = (destination / name).read_bytes()
                    inventory.append({
                        "path": name, "sha256": _sha256(data), "size_bytes": len(data),
                        "figure_id": stem, "format": extension,
                        "caption": CAPTIONS[stem],
                        "evidence_class": "original conceptual diagram; not data-driven evidence",
                    })
            finally:
                # No pyplot manager is created; clear artists and release the canvas.
                figure.clear()
                figure.set_canvas(None)
    receipt = {
        "schema": SCHEMA,
        "complete": True,
        "evidence_class": "original conceptual/analytical schematics, not empirical or numerical results",
        "producer": {"path": PRODUCER, "sha256": producer_hash},
        "rendering": {
            "geometry": "native Matplotlib vector patches, lines, arrows and text",
            "backend": "object-oriented FigureCanvasAgg; SVG exporter for the same figure",
            "font": "Matplotlib bundled DejaVu Sans; SVG glyphs embedded as vector paths",
            "figure_inches": [9, 6], "png_dpi": DPI, "png_pixels": [2700, 1800],
            "svg_hashsalt": HASH_SALT, "svg_date": None,
            "randomness": "none", "network_or_browser": False,
            "scope": "byte reproducibility within the same Python/Matplotlib/font environment",
            "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        },
        "sources": sources,
        "captions": CAPTIONS,
        "assets": inventory,
        "output_inventory": [item["path"] for item in inventory] + [RECEIPT],
        "limitations": [
            "No workload or energy function is executed to construct these schematics.",
            "No numeric model results, device calibration, or observed behavior are depicted.",
            "Base and enhanced workload families are separate analytical conventions.",
            "The source hashes bind interpretation to this checkout, not empirical validation.",
            "This producer is an original requested asset path, not a fallback for failed Mermaid rendering.",
        ],
    }
    receipt_path = destination / RECEIPT
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh repository-local output directory (must not exist; parent must exist).")
    args = parser.parse_args(argv)
    try:
        receipt = generate(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Graphical abstract generation failed: {error}\n")
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
