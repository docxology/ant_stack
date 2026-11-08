# Validation: Native Behavior, Installed Consumers and Publication Evidence

A check establishes only what it exercised on its identified source/runtime. Input validation, artifact integrity, scientific validity, actual rendering and remote publication are separate acceptance claims.

## Required Native Software Gates

```bash
uv sync --extra dev
bun install
uv run pytest --collect-only -q
uv run pytest -q
uv run ruff check .
uv run python tools/ensure_folder_docs.py --check
uv run antstack-build --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

Acceptance requires successful collection, the full selected suite without concealed failures/skips, Ruff success and intentional folder-guide coverage. The folder checker rejects missing/unsafe guides and stale marked source-owned text; curated prose still requires review. No configured type checker is implied by these commands.

`antstack-build --validate-only` validates existing ordered sources, references and citation keys; it does not regenerate missing analytical content, render Mermaid or produce a PDF. All-paper mode regenerates ignored `build_report.md`, a timestamped local diagnostic. A retained SUCCESS report is not fresh evidence. Run-all validation returns before creating its layout but still validates selected paths and required tools.

## Exercise Changed Real Paths and Preserve Failed Outputs

Run the actual configured workflow, not just its validation mode. Preserve all task flags; redirect only caller-owned output/projection paths when staging is necessary. The example enables paper projection, so a root invocation intentionally updates a compatibility paper tree and archives the prior generation. Use fresh roots and safe staging under `cache/` when checks must not mutate versioned publication inputs.

Verify the current completion manifest's relative addresses, SHA-256, byte counts, declared task coverage and input snapshot. Independently verify the projection inventory, not a directory-wide file count. Demonstrate refused occupied/unowned/unsafe destinations leave prior outputs intact, and exercise relocation of a portable run. Numeric repeatability concerns deterministic CSV/statistics, not byte-identical native PDFs or timing loops. See [reproducibility](reproducibility.md).

Changed plots require inspection of actual images: paired samples, actual interval endpoints, meaningful units, visible tiny positive values and explicit axis scales. A real file-size-limit failure must not corrupt a previously reviewed image. Animations illustrate fixed model outputs, not observed experimental time; requested MP4 needs FFmpeg and codec/geometry inspection.

## Build a New Package and Prove Installed Entry Points

```bash
uv build --sdist --out-dir dist
uv build --wheel dist/ant_stack-3.0.0.tar.gz --out-dir dist
uv run python tools/release_smoke.py --wheel dist/ant_stack-3.0.0-py3-none-any.whl --work-dir cache/install-proof --require-pdf
```

The wheel comes from the new sdist, not an old `build/lib`. The work directory must not exist. The gate provisions an independent environment, runs consumers outside the checkout with no editable/source-tree fallback, compares installed Python payload and locked dependency identities, exercises all console commands, checks current artifact bytes and actually renders caller-owned ordered paper inputs. `--offline` requires cached pinned distributions plus isolated build requirements; absence is a failed prerequisite, never permission to use global packages.

`--require-pdf` makes missing native tools a failure. In optional mode the receipt distinguishes unavailable tools from passed render; present-but-broken tools fail. Retain its receipt, exact command logs, wheel/sdist hashes and selected environment identifiers. This Unix-style gate is not proof of every supported Python/platform combination.

## Render and Inspect Every Configured Manuscript

Actually run `uv run antstack-build --project-root CALLER_PROJECT` against fresh staging inputs for all three configured papers. A successful compilation is necessary, not sufficient. Inspect real pages for mathematics, preserved Unicode code glyphs, legible figures, figure/table caption attachment, counter references, citation resolution, informative headings/TOC and complete content. Verify figure/table provenance against the selected result family. Source parser checks do not establish those properties.

Native Mermaid uses a real local browser with its sandbox enabled. The local nested-sandbox failure is a failure, not an absent figure to replace silently. No `--no-sandbox`, implicit public renderer, cached-image fallback or suppressed missing-glyph diagnostic is acceptable. The default code font is the bundled DejaVu mono font with an explicit NFSS family; custom font/template inputs remain caller-owned and trusted. See [PDF rendering](../PDF_RENDERING_GUIDE.md) for syntax and tool requirements.

## CI and Versioned Publication Are Separate Observations

The repository-owned workflow `.github/workflows/validate.yml` targets fresh Linux runners with pinned action/tool versions, tests, lint/docs, installed package proof, all three actual manuscript builds and the configured canonical run. Allowlisted logs/results survive failure. It is not a release/registry publisher. A workflow file does not prove a launched or green run.

Before publication record the exact tested commit, commands, runtime/dependency/native-tool identities, outcomes, artifacts and all failures/non-runs. Require live checks for that source, obey branch protection, then verify remote main and the version tag target. Verify a GitHub Release and its assets independently; a main push/tag/local build is not a published repository release. The private Node package has no registry publication step.

## Intermediate Evidence Is Not a Current Release Receipt

During 3.0 integration, an intermediate full suite observed **877 passed, 232 subtests** before subsequent fixes; template boundary/precision cases and a real configured pipeline were separately exercised. Native counterexamples confirmed caller-RNG mutation in duplicated toy models, corruption of a prior PNG under a real 2048-byte file limit, caller-log capture and plotting-state mutation, and PDF permission broadening. These observations motivate the repairs; they do not qualify later source automatically.

The integration owner must replace intermediate status with current source-bound gate/render/install/remote evidence before claiming a validated release. Scientific calibration, device/animal validation and external-deposit correction remain outside what software/render gates prove.

The subsequent final native suite/package driver and single-process Ruff launch were refused **before child launch** by `Launch policy: waiting_for_maintenance`; they are non-runs, not failed software assertions or passing gates. Do not evade admission through an unqualified launch. A task-branch CI run must qualify the final source on the repository-owned workflow before main/tag publication.
