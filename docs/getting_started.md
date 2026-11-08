# Getting Started: Install, Validate Inputs, and Generate Explicit Outputs

Start with the workflow you need. Source validation does not render a PDF; generating analytical results does not rebuild a manuscript. The authoritative option/input contract is [public API contracts](public_api_contracts.md), with acceptance gates in [validation](validation.md).

## Install the Checkout or Use an Installed Package

The package requires Python **3.11 or newer**. The checkout's `.python-version` selects **3.15**; the reported local runtime for the 3.0 work is Python **3.15.2**, uv **0.13.7**, Bun **1.4.4**, and Node **26.13.0**. These are environment identifiers, not a claim that every supported Python/platform combination has been tested.

From a source checkout:

```bash
uv sync --extra dev
bun install
uv run python -c "import antstack_core; print(antstack_core.__version__)"
```

`uv` manages Python dependencies and installed command registration. Bun installs the pinned JavaScript helpers used for native Mermaid rendering; Python-only analysis does not itself need a browser. Node must be compatible with those helpers. Installation alone does not establish renderer availability.

An installed wheel supplies `antstack-ce`, `run-all-antstack`, and `antstack-build`, backed by package code. It does **not** supply your project's `papers/`, manifests, run configs, bibliographies, assets, scripts, or tests. Supply those inputs explicitly. Use the installed commands directly in that environment; the examples below use `uv run` for checkout-managed execution, not as a requirement to recreate this source tree.

## Validate Inputs Without Claiming a Render

From the checkout, validate the CE manifest and run-all configuration:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

CE validation does not require `--out`, execute timed loops, or acquire hardware observations. Run-all validation creates no run artifacts. It still checks configured paths and prerequisites such as requested historical PDFs or MP4 tooling; success is not scientific validation or proof that generation will succeed.

For a caller-owned paper project, replace the example absolute path with your project location:

```bash
uv run antstack-build --project-root /absolute/path/to/caller-project --validate-only
```

In this checkout, `--project-root .` selects the repository papers. The builder discovers `papers/<name>/paper_config.yaml` and validates the currently configured ordered sources, references, and bibliography. It does not regenerate missing `Generated.md` or render diagrams/PDFs in validation-only mode. Validating all papers writes the selected project's timestamped `build_report.md`; selecting one with `--paper NAME` prints its result without that aggregate report. Read [validation](validation.md) for the full side-effect and gate contract.

## Generate Standalone CE Results into a Fresh Destination

Choose a new, exclusively owned root; the following path must not contain an earlier generation:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out outputs/ce-first-run
```

`--out` is required for generation. All outputs, including `results.csv`, `Generated.md`, `assets/`, input snapshot, summary, and provenance, live inside that root. A new or empty directory is accepted; a nonempty directory is refused. Do not target the retained `papers/complexity_energetics/out` tree or bypass refusal by deleting ownership records. Use another fresh destination for another invocation.

The canonical runner uses the **base** workload family. Closed-form values are analytical estimates; timed loops are synthetic workloads, not robot or neural-controller simulations. Enhanced legacy APIs use different formulas and must be identified separately; see [API reference](api_reference.md).

## Publish a Canonical Run and Inspect Its Inventory

The example config generates a fresh run ID and publishes under `outputs/<run_id>/`:

```bash
uv run run-all-antstack --config configs/run_all_antstack.example.yaml
```

**Before running:** the example also enables `paper_projection`, updating the configured paper's compatibility `Generated.md`, assets, and `out/` files, while retaining the prior paper tree in a sibling archive. To avoid this publication, use a caller-owned config with `complexity_energetics.paper_projection: false`. Paths in run-all configs are resolved from the invoking working directory, not from the YAML file's directory. Installed consumers must provide their own manifests and historical PDF paths or select no historical PDFs.

With the example's enabled tasks, the run contains data, statistics, static plots, illustrative GIF/HTML animations, reports, paper artifacts, logs, and provenance. The root `manifest.json` is the completion record: schema v2 uses relative artifact paths, SHA-256 identities, and byte counts. Follow [reproducibility](reproducibility.md) for input snapshots and portable interpretation. Existing runs cannot be casually overwritten; owned completed replacements require explicit config permission and archive the previous generation.

## Choose the PDF Operation You Actually Need

| Operation | Input and meaning |
| --- | --- |
| `antstack-build ... --validate-only` | Checks current configured paper inputs; no new PDF and no proof of native rendering. |
| `antstack-build --project-root CALLER_PROJECT [--paper NAME]` | Runs configured analysis when enabled, then actually renders ordered manuscript sources through Pandoc/XeLaTeX. Omitting `--validate-only` changes the operation. |
| Run-all with `complexity_energetics.build_pdf: true` | Produces canonical `papers/complexity_energetics.pdf` from the current generated analytical report, not the full manuscript. The example defaults to `false`. |
| Run-all `complexity_energetics.pdf_sources` | Copies explicitly selected historical PDFs. The example retains `2_complexity_energetics.pdf`; this is not a new manuscript build. |

Actual PDF rendering needs Pandoc, XeLaTeX, the required TeX packages (including `amsmath` and `cleveref`), configured fonts, and all local images/bibliography resources. SVG inputs need `rsvg-convert`. Mermaid sources additionally need local `mmdc` and a working sandboxed Chromium/Chrome environment. MP4 animation needs `ffmpeg`; the default GIF/HTML animation does not.

Native Mermaid Chromium sandbox failure has been observed locally. Do not disable the sandbox, suppress the warning, or treat a historical asset as a current render. A prepared remote Linux workflow is not evidence of a launched or green run. See [troubleshooting](troubleshooting.md) before interpreting a failure, and [benchmarking](benchmarking.md) before interpreting timings.
