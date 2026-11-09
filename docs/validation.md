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

The wheel comes from the new sdist, not an old `build/lib`. The work directory must not exist. The gate provisions an independent environment under a fresh repository-private work directory and runs consumers from a caller-owned directory outside the package/source import trees, with no editable/source-tree fallback. It compares installed Python payload and locked dependency identities, exercises all console commands, checks current artifact bytes and actually renders caller-owned ordered paper inputs. `--offline` requires cached pinned distributions plus isolated build requirements; absence is a failed prerequisite, never permission to use global packages.

`--require-pdf` makes missing native tools a failure. In optional mode the receipt distinguishes unavailable tools from passed render; present-but-broken tools fail. Retain its receipt, exact command logs, wheel/sdist hashes and selected environment identifiers. This Unix-style gate is not proof of every supported Python/platform combination.

## Render and Inspect Every Configured Manuscript

Actually run `uv run antstack-build --project-root CALLER_PROJECT` against fresh staging inputs for all three configured papers. A successful compilation is necessary, not sufficient. Inspect real pages for mathematics, preserved Unicode code glyphs, legible figures, figure/table caption attachment, counter references, citation resolution, informative headings/TOC and complete content. Verify figure/table provenance against the selected result family. Source parser checks do not establish those properties.

Native Mermaid uses a real local browser with its sandbox enabled. The local nested-sandbox failure is a failure, not an absent figure to replace silently. No `--no-sandbox`, implicit public renderer, cached-image fallback or suppressed missing-glyph diagnostic is acceptable. The default code font is the bundled DejaVu mono font with an explicit NFSS family; custom font/template inputs remain caller-owned and trusted. See [PDF rendering](../PDF_RENDERING_GUIDE.md) for syntax and tool requirements.

## CI and Versioned Publication Are Separate Observations

The repository-owned workflow `.github/workflows/validate.yml` targets fresh Linux runners with pinned action/tool versions, tests, lint/docs, installed package proof, all three actual manuscript builds and the configured canonical run. Allowlisted logs/results survive failure. It is not a release/registry publisher. A workflow file does not prove a launched or green run.

Before publication record the exact tested commit, commands, runtime/dependency/native-tool identities, outcomes, artifacts and all failures/non-runs. Require live checks for that source, obey branch protection, then verify remote main and the version tag target. Verify a GitHub Release and its assets independently; a main push/tag/local build is not a published repository release. The private Node package has no registry publication step.

## 3.0.0 Source-Bound Qualification and Explicit Limits

The publication build source is **`422c71e5a402d31e7b2a2676423a2651277509ce`**, qualified by [CI run 42548062547](https://github.com/docxology/ant_stack/actions/runs/42548062547). Its [exact workflow commands](https://github.com/docxology/ant_stack/blob/422c71e5a402d31e7b2a2676423a2651277509ce/.github/workflows/validate.yml) are version-bound; the final main/tag commit may additionally contain the resulting artifacts and this receipt. Authored input/resource identities, PDF hashes and build-source identity are preserved in [manuscript provenance](../papers/manuscripts.provenance.json).

| Exercised surface | Observed result |
|---|---|
| Collection and full pytest | 1012 tests and 253 subtests passed; no skipped tests or warning summary |
| Ruff, folder guides and build/CE/run-all validation | All succeeded |
| Fresh sdist → wheel → independent installed environment | All console consumers passed with installed payload/locked dependency identities; no editable/source-tree fallback |
| Actual installed PDF boundary checks | Literal Unicode and long code retained; text stayed within visible page margins; Mermaid title delimiters remained literal text |
| Supplemental producers | All ten actual analysis/figure/table wrappers completed with current-byte inventories |
| Three configured full manuscripts | 23 architecture, 37 complexity–energetics and 5 documentation pages; every page rasterized, all contact sheets and selected full-size graphical/math/citation/code pages inspected |
| Configured canonical pipeline | All seven configured tasks completed; 52 declared artifacts verified in the [CI capsule](../outputs/ci-42548062547-1/manifest.json) |
| Independent analytical replay | 300 native/CI rows compared, with zero observed numeric differences; this is not a byte-identical PDF/plot or empirical replication claim |

Observed CI toolchain: Python 3.15.2, uv 0.13.7, Bun 1.4.4, Node 26.13.0, Pandoc 3.13, XeTeX from TeX Live 2022/Debian, private checksum-pinned fvextra 1.14.0, Mermaid CLI 12.2.0, Puppeteer 25.16.0 and Chrome for Testing 156.0.8078.47. The browser sandbox remained enabled. Native macOS checks used Python 3.15.2 and TeX Live 2026; the full native suite also observed 1012 tests and 253 subtests passing after removal of a misleading global NumPy-reload probe.

The CI bundle contains `logs/`, `metadata/`, `installed-smoke/receipt.json`, package identities and `artifact-sha256.json`; artifact retention is 14 days. Local complete bundles and governed execution receipts are retained under `cache/wfz-release/` and the inherited vault's execution-receipt store. Durable numerical capsules, current manuscript/graphical receipts and the three reviewed PDFs are committed here; a retained diagnostic alone does not qualify later source.

### Failed or Unavailable Attempts Were Not Passing Gates

- [Initial CI 42542443619](https://github.com/docxology/ant_stack/actions/runs/42542443619) failed because a Puppeteer `Promise<string>` was treated as an executable path. The actual pinned API counterexample and awaited correction were exercised; subsequent CI passed.
- [CI 42546733769](https://github.com/docxology/ant_stack/actions/runs/42546733769) observed one failed PDF case and 1011 passed tests because TeX Live 2022's bundled fvextra lacked `breaknonspaceingroup`. The current workflow installs and verifies the real pinned package in its private TeX tree; the complete current run passed.
- Native adversarial rendering exposed clipped highlighted code, discretionary characters in copied code and a Markdown/TeX bracket collision in generated captions. Real long-code geometry/literal-text and actual Pandoc caption regressions now run in the installed-consumer gate; they were not weakened to accept those failures.
- An offline native wheel-install attempt lacked cached PyYAML 6.0.3. It remained a failed prerequisite; the successful CI consumer provisioned the locked distributions online in a fresh environment.
- The qualified local browser environment refused Chromium's nested sandbox, and a separate configured documentation render lacked the requested `Latin Modern Roman` font in its private local font environment. These local attempts were not full-paper successes. The successful Linux run used the authored configurations, real sandboxed browser and available declared fonts; no unsafe browser flag or font fallback closed the gap.
- Earlier `waiting_for_maintenance` refusals occurred before child launch and are non-runs. Later admitted, source-bound checks succeeded; no caller checkpoint or exception bypassed admission.

Known-advisory queries of the selected locked Python distributions and Bun dependency graph reported no findings at the recorded observation time. They are not a security certification or an audit of every native tool. No configured type checker, all-Python/platform matrix, physical RAPL/NVML calibration, independent device/animal validation or external Zenodo-deposit correction is claimed. Live main/tag/CI and GitHub Release state still require independent verification.
