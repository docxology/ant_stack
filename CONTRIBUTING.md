# Contributing: Package Behavior, Scientific Claims and Release Evidence

Use `uv` for Python and Bun for the pinned JavaScript tooling. Reusable behavior belongs in `antstack_core`; console scripts and research wrappers delegate to it. Installed consumers supply their own project inputs, not this checkout's scripts or tests.

## Preserve the Scientific and Filesystem Boundaries

- Name the selected base or enhanced model, units, window, coefficients and evidence class. Analytical counters, synthetic timing loops and measurements are different products.
- Reproduce a defect when feasible; add deterministic behavioral coverage for meaningful boundaries. Do not pin source wording, incidental defaults, mock echoes or pre-created-file existence.
- Migrate affected callers when contracts change; remove obsolete implementations instead of maintaining compatibility shims.
- Preserve unrelated branches, files and prior outputs. New CE invocations require a fresh explicit destination; canonical replacements require recorded ownership/completion and retain their prior generation.
- Use source-backed or explicitly prospective manuscript prose. Concepts and proposed experiments do not demonstrate a robot, biological validation, calibration or publication readiness.

## Install the Pinned Development Environment

```bash
uv sync --extra dev
bun install
```

Keep environments and caches project-local; do not change host configuration or dependency checkouts. Native rendering also needs the tools described in [PDF rendering](PDF_RENDERING_GUIDE.md). Repository-specific ancestor admission requirements still apply in a governed checkout.

## Verify Behavior, Not Merely Source Readiness

The authoritative checklist and its side effects are in [validation](docs/validation.md). Run focused regressions, then the full suite, Ruff and folder-documentation gate. Exercise changed real CLI/API paths, build a new sdist/wheel, and use the isolated installed-consumer gate when packaging or entrypoints change. Render and inspect affected documents; validation-only is not a render.

For analytical generation choose a fresh root, for example:

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out outputs/ce-review-run
```

Do not direct standalone generation into the occupied retained paper `out/` directory. Use the explicit projection contract in [reproducibility](docs/reproducibility.md), or the configured canonical workflow when intentionally updating a paper projection. The example run-all configuration enables that publication; inspect it before running.

## Keep Documentation Modular and Artifact Provenance Coherent

Use [the documentation index](docs/README.md) to place changes in the owning guide rather than duplicating recipes. Update API contracts, examples, source-owned folder guidance and release notes with changed behavior. Source headings must explain their actual content and retain intentional reference targets.

Keep a numerical result family together: input snapshot, CSV, summary, figures, generated Markdown and provenance from one completed generation. Conceptual graphical abstracts use `tools/generate_graphical_abstracts.py`; inspect their actual exports and retain their source/output receipt. Never edit generated concatenations as manuscript source.

Remove only task-owned temporary scaffolding after proof. Do not delete someone else's ignored environment/cache or treat historical artifacts as current results. Timestamped `build_report.md` is a regenerated local diagnostic, not versioned release evidence.

## Publish Only Validated Source

Synchronize `pyproject.toml`, `antstack_core.__version__`, private `package.json`, locks and release notes. The private Node package is tooling, not a registry publisher. A commit, passing CI, merge, tag and repository release are distinct states: verify each claimed state against the remote. Follow actual branch protection, configured identity, admission and exact-confirmation requirements; no force pushes, tag replacement or gate bypasses.
