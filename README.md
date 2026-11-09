# Ant Stack

Analytical workload/energy models and scientific-publication tooling, packaged as `antstack_core` (3.0.1). The Body–Brain–Mind architecture is a proposal: the package does **not** implement an ant dynamics solver, validated neural circuit, active-inference decision engine, or calibrated hardware-energy model.

## Start with the native toolchain

Use `uv` for Python and Bun for Node tooling. `.python-version` selects Python 3.15; the package requires Python 3.11 or newer. `.node-version` pins Node. Keep development environments and caches local to the checkout:

```bash
export UV_CACHE_DIR="$PWD/cache/uv"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/cache/xdg"
export PUPPETEER_CACHE_DIR="$PWD/cache/puppeteer"
export BUN_INSTALL_CACHE_DIR="$PWD/cache/bun"
uv sync --extra dev
bun install
```

Full paper rendering additionally needs Pandoc 3.13, XeLaTeX with `cleveref`, and a working local Chromium/Puppeteer installation. SVG inputs need `rsvg-convert`; MP4 output needs FFmpeg. Installing Python dependencies alone does not qualify these paths. Follow inherited admission rules when this checkout is managed by a collection; standalone CI uses only this project's commands.

## Choose the intended output contract

| Command | Result and boundary |
|---|---|
| `uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only` | Validate the analytical manifest without creating a run. |
| `uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out outputs/standalone-ce-example` | Generate one standalone CE artifact family. The selected root must be new or empty; choose a fresh name on subsequent runs. |
| `uv run run-all-antstack --config configs/run_all_antstack.example.yaml` | Generate a fresh canonical `outputs/<run_id>/` and, when enabled, its CE paper projection. |
| `uv run antstack-build --validate-only` | Check configured sources/references; write `build_report.md`, but no PDFs. |
| `uv run antstack-build` | Run configured analysis and actually render the three papers. A failed render returns nonzero and does not replace a previous PDF. |

`bun run ce --out outputs/standalone-ce-example` forwards the explicit output choice. It no longer writes directly over the paper's populated `out/` directory.

## Canonical evidence, projections, and historical artifacts

A completed canonical run has `manifest.json`, relative artifact paths, SHA-256/byte identities, input snapshots, package-source identities, dependencies, and an explicit completion flag. These certify recorded bytes and requested producer coverage—not empirical accuracy, authenticity against a malicious editor, or biological validity. [Reproducibility](docs/reproducibility.md) defines the boundaries.

The CE projection consists of `papers/complexity_energetics/Generated.md`, produced assets, and `out/`, with an ownership inventory. Use run-all or the configured paper builder to update it coherently. A failed replacement preserves the previous completed generation; owned replacements retain their prior tree in an archive. Do not edit generated numbers by hand.

The default run-all config copies the existing PDF as `historical_paper`; `build_pdf: false` does **not** compile a fresh manuscript. `build_pdf: true` creates a generated analytical-results report, not the complete manuscript. Full configured papers are built by `antstack-build`. Retained v2 PDFs, `outputs/example-run/`, `outputs/test-run/`, and top-level `complexity_energetics/` are historical comparison material unless a fresh source-bound receipt explicitly identifies them.

The reviewed 3.0 publication set is identified by [manuscript provenance](papers/manuscripts.provenance.json): [architecture](1_ant_stack.pdf), [complexity–energetics](2_complexity_energetics.pdf) and [documentation](1_ant_stack_documentation.pdf). The [qualified CI capsule](outputs/ci-42548062547-1/manifest.json) and [native replay capsule](outputs/release-v3-0-0-20261109/manifest.json) preserve their own producing sources; neither turns copied historical papers or analytical estimates into empirical validation.

## Source map and evidence classes

| Surface | Responsibility |
|---|---|
| `antstack_core/analysis`, `cohereants` | Workload/energy assumptions, statistics, graph/spectral/behavioral helpers and explicit unavailable measurements. |
| `antstack_core/figures`, `mathematics` | Data-aligned plots, local Mermaid preprocessing and context-preserving math handling. |
| `antstack_core/orchestration` | Installed CE runner; staged canonical publication and current projection inventory. |
| `antstack_core/publishing`, `cli` | Installed configured builds, document-wide counters, validation and provenance. |
| `papers/ant_stack` | Conceptual architecture and proposed evaluation protocols. |
| `papers/complexity_energetics` | Analytical models with assumed coefficients; current numbers come from a completed generated run. |
| `papers/documentation` | Configured-source and rendering guide. |
| `tools`, `scripts` | Repository maintenance, reproducible graphical abstracts, release smoke and thin launchers. |

Base workload models and the separate enhanced models are different model families. Repeated deterministic values and descriptive bootstrap intervals are not independent experiments or evidence of calibrated uncertainty. Hardware measurements, species validity, timing guarantees, and physical cost of transport require their own matched observations.

## Validate, inspect, then release

[Validation and receipts](docs/validation.md) lists the native gates and records what was actually exercised. [Public API contracts](docs/public_api_contracts.md) describes breaking 3.0.0 behavior; [PDF rendering](PDF_RENDERING_GUIDE.md) distinguishes readiness from a readable render. [Contributing](CONTRIBUTING.md) covers source ownership and release discipline. The private Node package has no registry-publishing workflow.

- [Documentation index](docs/README.md)
- [Architecture contract](docs/architecture.md)
- [External ant-data integration](docs/external_ant_data_integration.md)
- [Release history](RELEASE_NOTES.md)
