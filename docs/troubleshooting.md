# Troubleshooting: Input Contracts, Native Renderers, and Refused Publication

Diagnose the failed operation and preserve its stderr, exit status, selected project, input identities, and output destination. Do not turn a failed check into a success claim by suppressing diagnostics, reusing old outputs, or weakening security. [Validation](validation.md) defines acceptance checks; this guide explains common causes.

## Installed Commands Cannot Find Project Inputs

In a checkout, `uv sync --extra dev` installs the Python package and its command entrypoints. Use `uv run python -c "import antstack_core; print(antstack_core.__version__)"` to identify the selected environment rather than assuming a shell's Python is the same one.

Outside the checkout, an installed package does not contain the repository's papers, manifests, helper scripts, or tests. Pass a real manifest to `antstack-ce`, a caller-owned YAML file to `run-all-antstack --config`, and a project root containing `papers/<name>/paper_config.yaml` to `antstack-build --project-root CALLER_PROJECT`. The build default is the current directory, not package resources. An empty discovery result is a failure, not a successful zero-paper build.

Relative run-all paths use the invoking working directory. Check the actual `complexity_energetics.manifest`, `paper_root`, `pdf_sources`, and `run.output_root`, not merely the location of the config file.

## CE Requires an Output Root or Refuses an Existing One

Validation requires no destination:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only
```

Generation requires an explicit new or empty root:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out outputs/ce-next-run
```

The example destination must be fresh for this invocation. Do not retry generation into a nonempty old paper `out/` directory or infer success from old CSV/figures. A partial standalone CE destination is not reusable as a completed run; preserve it for diagnosis and choose a fresh root.

## Paper Validation Succeeds but PDF Rendering Fails

`antstack-build --validate-only` checks current configured sources/references/bibliography. It does not execute the renderer or regenerate missing analytical content. An all-paper validation also writes the project's `build_report.md`; that report's validation-only mode is not a PDF build certificate.

Actual rendering additionally depends on Pandoc, XeLaTeX, TeX packages, configured fonts, and local resources. `validate_pdf_environment()` checks executable/package availability, not a complete render. Inspect the build result's `error`, `warnings`, and source binding, not only whether an output filename exists. Publication occurs only after a new successful staged render; a previous PDF can remain after failure.

Repair missing fonts/glyphs, bibliography keys, unresolved or duplicate references, absent templates, and local image paths at their source. Figure/table identifiers must bind to the actual numbered object, not an unrelated heading or unnumbered anchor. Literal code, mathematics, and URIs should not be rewritten to evade reference checks. Explicit custom templates must exist. Remote image URLs are not implicitly downloaded; supply intentional local resources. SVG resources need `rsvg-convert`.

## Native Mermaid Cannot Start Chromium or Render a Diagram

`bun install` supplies the checkout's pinned Mermaid CLI/Puppeteer dependencies, but does not prove that a usable browser and sandbox are available. The renderer searches `ANTSTACK_MMDC`, then `mmdc` on `PATH`, then the current directory's `node_modules/.bin/mmdc`. Set `ANTSTACK_MMDC` to the intended executable when needed. `PUPPETEER_EXECUTABLE_PATH` selects a browser; otherwise the implementation checks common local Chrome/Chromium locations before relying on the browser tool's own behavior.

Distinguish lexical diagram validation from real rendering. Inspect the logged browser stderr for missing executable, dependency, permissions, or sandbox failures. Native Chromium sandbox failure has been observed locally; do **not** disable the sandbox or add unsafe browser flags. Repair the supported browser/runtime environment or run the actual render in a supported sandboxed environment. A prepared Linux CI job is not proof it ran successfully.

There is no automatic public remote renderer fallback. The Python Mermaid API accepts an explicitly supplied credential-free HTTP(S) service URL; choosing it is a deliberate data-transfer/trust decision, not a CLI troubleshooting default. Failed rendering raises an error and must not be satisfied by a cached old image. Leaving fenced text unprocessed does not count as a rendered diagram.

## Run-All Rejects Paths or an Existing Generation

Path refusal protects inputs, implementation files, Git control namespaces, runtime environments, and output ownership. Do not publish through symlinks, overlap input/output/projection roots, or select a repository/control root as a replaceable tree. Fix the destination rather than removing the guard.

A run ID must be one safe dot-free segment. An existing run is refused unless `clean_run_dir: true` is set and its ownership marker and completed manifest are valid for that run ID. The old generation is archived, not silently erased. Unowned, incomplete, or invalid runs are not eligible for replacement; use a fresh run ID. Paper projection likewise stages the current inventory and archives its prior tree. These protocols require trusted exclusive access during publication; they do not guarantee isolation from hostile concurrent changes.

## Artifacts Are Missing or Historical PDFs Are Misidentified

Use the current invocation's result and `outputs/<run_id>/manifest.json`, not a directory-wide file count. A completed v2 manifest binds relative artifact paths to SHA-256 and byte counts. Check selected tasks and the required artifact types together: a narrow task selection can intentionally fail a config that still requires all default artifact classes. Do not lower requirements merely to conceal a producer failure.

Requested `pdf_sources` must exist even though they are historical copies. A `historical_paper` record is not a new build. The optional canonical `papers/complexity_energetics.pdf` is a render of generated analytical results, not the full manuscript. See [getting started](getting_started.md) for these distinct operations and [reproducibility](reproducibility.md) for inventories/snapshots.

For requested MP4, install a working `ffmpeg`; the default GIF/HTML path does not require it. Animations reveal fixed model values and do not show observed experimental time.

## Hardware Measurements Are Unavailable or Results Differ Across Runs

A null meter explicitly disables measurement; its internal zero is not an empirical zero. RAPL needs a readable supported CPU-package counter; NVML needs its optional backend, compatible driver, and device. Requested unavailable observations are errors. Do not relabel modeled `energy_est_j` as `measured_energy_j` or substitute null results for a hardware request. Closed-form workloads cannot be hardware-measurement runs.

Closed-form calculations and seeded bootstrap intervals can be reproducible while timed synthetic loops vary with CPU scheduling, interpreter, and host load. The bootstrap seed does not make wall-time loops deterministic. Repeated identical analytical values do not demonstrate empirical precision. Base and enhanced legacy models can also differ by design; identify the chosen family before treating a numerical discrepancy as a defect.

## Software Checks Fail Without an Obvious Local Cause

Preserve the actual failure and the selected source/runtime/dependency state. Use the canonical [validation guide](validation.md) to identify the responsible acceptance check and [reproducibility](reproducibility.md) to bind its evidence. Do not claim that a historical pass, skipped test, warning filter, validation-only run, or unlaunched CI workflow establishes present success.
