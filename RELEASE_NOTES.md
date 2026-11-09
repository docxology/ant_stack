# Ant Stack Release Notes

## 3.0.0 — Scientific Contracts and Failure-Honest Publication

Source version: **3.0.0**. Source-bound acceptance evidence belongs in [validation](docs/validation.md). Publication state must be verified against the remote main/tag and GitHub Release; it is not inferred from this file.

### Breaking Contracts and Removed Dead Paths

- Python minimum is 3.11; the checkout targets Python 3.15. Python/private Node metadata and both locks are synchronized. The Node package remains private with no registry publisher.
- Installed CE and configured-paper commands use package-owned implementations, not unshipped scripts. CE generation requires a fresh explicit `--out`; the builder uses caller-owned `--project-root` inputs and fails on empty/missing/failed paper sets.
- Canonical manifests are portable schema v2, with input snapshots and current-byte identities. Prior owned generations/projections are archived; unsafe, overlapping, unowned, Git/control/runtime destinations refuse publication.
- Removed obsolete unified-build implementations/default template/filter paths and the orphaned paper-local configuration loader. Custom templates are explicit, strict and caller-owned; no compatibility shim hides a missing resource or malformed render.
- Meter configuration exposes only implemented backend controls. Unimplemented calibration, baseline subtraction, sampling/channel/logging fields and planning helpers are removed; unsupported controls are errors. Provenance metadata cannot silently disable mandatory source/version recording.
- The toy agent model has no meaningless seed argument, global RNG mutation or duplicate enhanced implementation. Its state-pair fraction is named `state_pair_similarity`, not graph clustering; population variance is backend-consistent and its accumulated costs are not physical joules.
- All ten supplemental analysis/figure/table wrappers require explicit fresh output families instead of overwriting retained assets. Complete families publish atomically with checksummed source/runtime provenance; manifest inputs are snapshotted and advertised vector/PDF formats are real. Distinct coefficient/memory boundaries stay disclosed.
- Contact/neural/planning maxima bound every corresponding study, including secondary grids, with explicit unavailable fits at small sizes. Label-only solver/pattern flags and unsupported inverse-input efficiency/tractability claims are removed. Uncomputed diagnostic quantities are null, not fabricated zeros; verbatim YAML/JSON input snapshots are never product-normalized.

### Correctness, Scientific Meaning and Output Custody

- Corrected Student/Welch inference, confidence bounds, constant bootstrap behavior, graph/information/spectral quantities, paired plot filtering, finite/domain checks, physical units and conditional erasure references.
- Undefined constant-response R² no longer crashes paired-bootstrap consumers; unavailable goodness-of-fit stays unavailable. One-ULP cadence rounding cannot erase a reciprocal-period update. Supplied empirical-efficiency notes are not validation scores.
- Comprehensive exports preserve unavailable fits instead of zero, reject incompatible aggregation windows and report isolated sweep means explicitly. Physical cost of transport has no fabricated mass/distance default. The key-number CLI visibly fails for missing/invalid inputs instead of reporting a successful empty validation.
- Base and enhanced workloads stay distinct. Selected canonical models, matched common windows, nonzero Mind compute and shared idle accounting replace inconsistent report substitutions. Null/unavailable hardware is not measured zero energy; missing scientific validation is not success.
- Caller logging/plot state and call-time working directory are preserved. Metadata does not copy ambient process arguments. Atomic PDF/image/text publication preserves prior outputs on failure and retains restrictive permissions; a real file-size-limit probe demonstrates interruption behavior.
- Private text temporaries start mode0600 before payload writes. Cross-filesystem atomic publication uses a destination-side private temporary; ordered-source notes/links resolve document-wide and image snapshots bind actual rendered bytes.
- Interval/dot visualizations expose tiny positive components across wide ranges, state units/scales and retain actual supplied bounds. Conditional references use visible log-scale marks; multipanel titles no longer overlap and zero/tiny components use a declared symlog scale. Figure captions attach to numbered objects; inverse-parameter illustrations are not mislabeled Pareto frontiers.
- Real optional MP4 and generated-results PDF producers are distinguished from copied historical manuscripts. Artifact inventories and compatibility projections are checked against current produced bytes rather than stale directory contents.
- CLI destination/task overrides apply before effective preflight. Configured paper and build-target output collisions fail before overwriting another manuscript. Linked implementation members cannot disappear silently from generating-source identities; supplementary captions render actual TeX commands rather than corrupted escape text.

### Manuscripts and Modular Documentation

- Revised scientific statements separate conceptual design, analytical/synthetic results, assumptions, primary literature, proposed protocols and absent empirical evidence. Incorrect citation identities and unsupported calibration/criticality/biological-performance claims are removed or qualified.
- Informative section titles preserve intentional Pandoc targets. Original conceptual graphical abstracts have reproducible vector/raster exports and a source/output receipt; they contain no measured results.
- Documentation has task-owned setup, API, theory, validation, provenance, data-integration and rendering guides. Source-owned folder guidance has an explicit managed marker/refresh path that preserves curated documents.
- New repository-owned CI covers native tests/lint/docs, packaging/isolated installed consumers, all configured manuscripts and the configured canonical workflow. A workflow definition is not proof of a completed green run; release state must be observed independently.
- The pinned Puppeteer executable-path API is awaited before checking the installed browser. Executability, project-cache containment and the native browser sandbox remain required; dependency installation alone is not render proof.

### Evidence Limits

Software/render checks do not establish device calibration, empirical ant/robot validation, biological superiority, universal real-time feasibility or external-deposit correction. The historical Zenodo DOI identifies the earlier v1 record, not these revised sources. Native browser sandbox failures remain failures; no security bypass or implicit public renderer is introduced.

---

## 2.0.0 — Historical Release (2026-05-18)

The following records the earlier release's statements and checks, not fresh validation of 3.0.0.


Release date: 2026-05-18

Ant Stack v2.0.0 is a major repository restructure and validation release. It turns the project into a modern `uv` Python package with `bun`-managed Mermaid tooling, package-owned orchestration, canonical generated outputs, and executable architecture contracts.

### Highlights

- Rebuilt the repository around installable `antstack_core` package modules for analysis, cohereants, figures, mathematics, orchestration, publishing, and CLI entrypoints.
- Added `run-all-antstack`, a package-owned orchestrator that validates one YAML config and generates canonical data, statistics, visualizations, animations, reports, paper projections, logs, manifests, and provenance under `outputs/<run_id>/`.
- Added current console commands: `antstack-build`, `antstack-ce`, and `run-all-antstack`.
- Added executable architecture contracts through `antstack_core.architecture`, including module/folder validation, public export checks, Mermaid rendering, and tests against placeholder method bodies.
- Added real Cohereants helper APIs for physical conversions, spectroscopy, behavioral analysis, statistics, power analysis, and plotting hooks.
- Added folder-level `README.md` and `AGENTS.md` coverage for every intentional directory and a checker at `tools/ensure_folder_docs.py`.
- Added Mermaid diagrams and corrected command contracts across the curated documentation set.
- Preserved compatibility outputs for the complexity energetics paper while establishing canonical outputs under `outputs/<run_id>/`.
- Removed tracked caches, Python bytecode, OS metadata, and the old `package-lock.json`; added `bun.lock`, `uv.lock`, and project-level ignore policy.

### Validation

The release was validated with:

```bash
uv sync --extra dev
bun install
uv run pytest --collect-only -q
uv run pytest -q
uv run ruff check .
uv run python tools/ensure_folder_docs.py --check
uv run antstack-build --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
bun run validate
bun run test:collect
bun run lint
```

Verified suite result:

```text
676 passed, 10 subtests passed
```

### Compatibility Notes

- New canonical generated artifacts should go under `outputs/<run_id>/`.
- Complexity energetics compatibility artifacts still project to `papers/complexity_energetics/out`, `papers/complexity_energetics/assets`, and `papers/complexity_energetics/Generated.md` unless disabled in config.
- Scripts and CLIs should remain thin wrappers over package APIs.
- Public package contracts are enforced through `__all__` exports and pytest contract tests.
