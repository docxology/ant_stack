# Reproducibility: Inputs, Model Boundaries, and Byte Identity

## Know what was generated

The example manifest selects **base closed-form analytical workloads**. Body contact-cost names are not executed contact solvers; Brain counts are not a validated connectome; Mind tree counts are not executed policy selection. Enhanced models are separate APIs/artifacts and must not be presented as canonical base-run results.

Operation prices are pJ/FLOP, memory-traffic prices pJ/byte, spike prices aJ/spike, powers W, durations s, and output energies J. Traffic is not peak storage. Updates use the configured frequency and interval; the planning horizon is in decision steps, not seconds. Convert every term before comparing it.

Module-isolated rows include that module's idle over its own workload window and are not an additive shared-system total. Common-window accounting recomputes all selected modules for the same interval, includes Mind compute, and counts shared idle once. Sensing/actuation assumptions are separate from digital compute. Cost of transport is unavailable without energy, distance, mass and locomotion interval that actually describe the same motion.

## Canonical completion is a bounded claim

`run-all-antstack` stages a new generation, closes its log, verifies current artifact bytes, prepares any requested compatibility projection, and publishes the canonical completed tree last. `clean_run_dir: true` permits archived replacement only of an owned, completed run; it does not authorize arbitrary directory deletion. A failed producer/replacement does not turn partial output into a completed run.

The top-level `manifest.json` uses schema 2 and relative artifact paths. `artifact_path_base` states how a manifest addresses them. Each declared artifact has a producing task, source configuration, SHA-256 and byte count. Recompute those identities from the current files; an old stored checksum alone proves nothing. The manifest does not hash itself or certify external datasets, biological agreement, hardware accuracy, or all possible package features.

The paper projection has `.antstack-ce-projection.json`, a current generated-path inventory, and byte identities. Existing unrelated paper files are preserved. Compare its CSV/assets with the corresponding completed canonical CE family; generated Markdown intentionally rebases its raw-CSV link to `out/`.

## Provenance distinguishes the caller from the implementation

A run records input snapshots and their identities, the working project's Git state, and a separate generating-package identity with version and Python-source hashes. An installed wheel must not be attributed to an enclosing source checkout merely because it is installed beneath that checkout. Failed/unavailable Git inspection is unknown, not a clean worktree. Staged and untracked source changes participate in dirty-source evidence.

Default provenance does not copy a parent application's command line. An explicit `command=` is opt-in; do not place credentials in it. Inputs/configurations are intentionally retained verbatim, so never put secrets in a manifest or assume generated evidence is sanitized for public release. Review it before publication.

Checksums detect byte changes, not malicious rewriting of both data and checksums. Stable symlinks/special output nodes are rejected where guarded publication requires ordinary paths; this is not hostile same-UID race isolation. The caller must exclusively own output namespaces throughout generation. Do not target Git stores, source roots, dependency environments, or unrelated directories.

## Seeds, repeated values, and statistical interpretation

`run.seed` controls run-all bootstrap intervals, including its CE producer; the manifest seed identifies the workload input. Standalone CE uses its manifest seed. Keep these scopes explicit. A deterministic model recalculated 100 times still supplies one conditional formula value, not 100 physical observations. Constant bootstrap samples have exactly zero-width intervals; resampling does not estimate unmodeled coefficient/calibration uncertainty.

Scaling fits describe a finite supplied grid. A large fitted power exponent is not proof of an exponential law, a phase transition, or measured tractability. Inverse-parameter illustrations are not observed performance objectives or demonstrated Pareto frontiers. Report parameter range, units, time window, model family and assumptions with every figure/table.

## Regenerate without overwriting a retained run

```bash
uv sync --extra dev
bun install
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml
```

The example's null run ID selects a fresh ID. Standalone generation requires an explicit fresh root:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out outputs/standalone-ce-example
```

Keep the original config, manifest, `uv.lock`, `bun.lock`, source/package identity, actual runtime/tool versions and exact invocation with an intentional result family. Native PDF/animation generation also depends on external fonts, TeX, browser and codec versions; Python dependency versions alone do not make those outputs byte-reproducible. Graphical abstracts have their own producer/source/output receipt and contain no numerical data.

All five supplemental figure/table generators require `--output NEW_DIRECTORY`. They stage complete families and publish only after relative product inventories and SHA-256/byte identities are written and rechecked; populated roots are not overwritten. Publication captions are native attached image captions, and multipanel LaTeX uses family-relative paths. Enhanced table data has `schema_version: 1` and `model_family: enhanced_analytical`; loading coherent inputs does not certify their freshness or empirical validity.

```bash
uv run python scripts/complexity_energetics/generate_tables_and_numbers.py --output cache/enhanced-001
uv run python scripts/complexity_energetics/generate_publication_figures.py --output cache/publication-001 --format svg
uv run python scripts/complexity_energetics/generate_multipanel_figures.py --output cache/multipanel-001
```

[Validation](validation.md) separates exercised proof from unavailable or unrun surfaces. [Rendering guidance](../PDF_RENDERING_GUIDE.md) explains full-paper versus generated-results PDFs.
