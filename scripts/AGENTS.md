# AGENTS: `scripts/` — Thin Orchestrator Scripts

Technical specification for the Ant Stack operational scripts.

## Scope
This file applies to `scripts`.

## Directory contract
Thin operational wrappers around antstack_core and paper-runner APIs.

## Script inventory

| Script | Delegates to |
| --- | --- |
| `run_all_antstack.py` | `antstack_core.orchestration.run_all:main` |
| `demonstrate_improvements.py` | `antstack_core.orchestration.demo:main` |
| `validate_rendering_system.py` | `antstack_core.publishing.rendering_validator:run` |
| `validate_unified_config.py` | `antstack_core.analysis.config_validation:main` |
| `common_pipeline/unified_build.py` | `antstack_core.orchestration.unified_build:run` |
| `common_pipeline/update_paper_key_numbers.py` | `antstack_core.analysis.key_numbers_updater:run` |
| `common_pipeline/comprehensive_formatting_fix.py` | `antstack_core.publishing.formatting_fixes:main` |
| `common_pipeline/build_core.py` | `antstack_core` publishing APIs via `ModularPaperBuilder` |
| `common_pipeline/run_validation_suite.py` | pytest and repo validators via subprocess |
| `common_pipeline/test_ce_runner.py` | `antstack_core.analysis` primitives |
| `common_pipeline/diagnose_crossref_issue.py` | `tools/render_pdf.sh` via subprocess |
| `common_pipeline/trace_pipeline.py` | `tools/render_pdf.sh` via subprocess |
| `common_pipeline/build_papers.sh` | `tools/render_pdf.sh` |
| `complexity_energetics/analyze_active_inference.py` | `antstack_core.analysis.ce_active_inference:run` |
| `complexity_energetics/analyze_contact_dynamics.py` | `antstack_core.analysis.ce_contact_dynamics:run` |
| `complexity_energetics/analyze_neural_networks.py` | `antstack_core.analysis.ce_neural_networks:run` |
| `complexity_energetics/comprehensive_analysis.py` | `antstack_core.analysis.ce_comprehensive_analysis:run` |
| `complexity_energetics/generate_comprehensive_analysis.py` | `antstack_core.analysis.ce_module_scaling_analysis:run` |
| `complexity_energetics/generate_biomechanical_benchmarks.py` | `antstack_core.analysis.ce_biomechanical_benchmarks:run` |
| `complexity_energetics/generate_manuscript_figures.py` | `antstack_core.analysis.ce_manuscript_figures:run` |
| `complexity_energetics/generate_publication_figures.py` | `antstack_core.analysis.ce_publication_figures:run` |
| `complexity_energetics/generate_results_figures.py` | `antstack_core.analysis.ce_results_figures:run` |
| `complexity_energetics/generate_multipanel_figures.py` | `antstack_core.analysis.ce_multipanel_figures:main` |
| `complexity_energetics/generate_tables_and_numbers.py` | `antstack_core.analysis.ce_tables_and_numbers:main` |
| `complexity_energetics/run_ce.sh` | `antstack_core.cli.ce` console script |

See `README.md` for per-script purpose and commands.

## Design contract

- Scripts are orchestration only: path bootstrap, argparse, logging, and a single delegated call into an `antstack_core` entrypoint.
- Business, data, plotting, and analysis logic lives in `antstack_core/` (importable and tested); scripts must not define classes, data tables, or workflow logic inline.
- Delegate modules live under `antstack_core/orchestration/`, `antstack_core/publishing/`, and `antstack_core/analysis/` (see inventory).
- Keep behavior identical when moving code between scripts and the package; update both this inventory and `README.md` whenever scripts or delegate modules change.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run antstack-build --validate-only; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only; uv run pytest -q
uv run python scripts/validate_rendering_system.py --help; uv run python scripts/common_pipeline/unified_build.py --help
```
