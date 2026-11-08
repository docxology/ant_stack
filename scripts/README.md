# Scripts

Thin operational wrappers for Ant Stack paper builds, validation, tracing, key-number updates, run-all orchestration, and complexity energetics analyses. Build and analysis logic lives in `antstack_core`; scripts only set paths, parse arguments, and invoke package entrypoints.

## Top level

| Script | Purpose | Delegates to | Command |
| --- | --- | --- | --- |
| `run_all_antstack.py` | Run-all orchestration entrypoint | `antstack_core.orchestration.run_all:main` | `uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only` |
| `demonstrate_improvements.py` | Package capability demonstration tour | `antstack_core.orchestration.demo:main` | `uv run python scripts/demonstrate_improvements.py` |
| `validate_rendering_system.py` | PDF rendering-system validation (configs, figures, math, hyperlinks) | `antstack_core.publishing.rendering_validator:run` | `uv run python scripts/validate_rendering_system.py [--paper NAME] [--verbose]` |
| `validate_unified_config.py` | Unified configuration consistency checks | `antstack_core.analysis.config_validation:main` | `uv run python scripts/validate_unified_config.py` |

## `common_pipeline/`

| Script | Purpose | Delegates to | Command |
| --- | --- | --- | --- |
| `update_paper_key_numbers.py` | Replace hardcoded manuscript numbers with key-number placeholders | `antstack_core.analysis.key_numbers_updater:run` | `uv run python scripts/common_pipeline/update_paper_key_numbers.py [--paper NAME] [--section NAME] [--validate]` |
| `comprehensive_formatting_fix.py` | Conservative prose URL/path/variable formatting; no identity guessing | `antstack_core.publishing.formatting_fixes:main` | `uv run python scripts/common_pipeline/comprehensive_formatting_fix.py FILE.md ...` |
| `run_validation_suite.py` | Comprehensive validation suite runner | pytest and repo validators via subprocess | `uv run python scripts/common_pipeline/run_validation_suite.py` |
| `test_ce_runner.py` | Complexity-energetics runner sanity assertions | `antstack_core.analysis` primitives | `uv run python scripts/common_pipeline/test_ce_runner.py` |
| `diagnose_crossref_issue.py` | Cross-reference diagnostics | `tools/render_pdf.sh` via subprocess | `uv run python scripts/common_pipeline/diagnose_crossref_issue.py` |
| `trace_pipeline.py` | Pipeline tracing tests | `tools/render_pdf.sh` via subprocess | `uv run python scripts/common_pipeline/trace_pipeline.py` |
| `build_papers.sh` | Legacy paper build loop | `tools/render_pdf.sh` | `bash scripts/common_pipeline/build_papers.sh` |

## `complexity_energetics/`

| Script | Purpose | Delegates to | Command |
| --- | --- | --- | --- |
| `analyze_active_inference.py` | Active-inference complexity and bounded-rationality analysis | `antstack_core.analysis.ce_active_inference:run` | `uv run python scripts/complexity_energetics/analyze_active_inference.py [--output DIR]` |
| `analyze_contact_dynamics.py` | Contact solver comparison and terrain effects | `antstack_core.analysis.ce_contact_dynamics:run` | `uv run python scripts/complexity_energetics/analyze_contact_dynamics.py [--output DIR]` |
| `analyze_neural_networks.py` | Neural connectivity, sparsity, and brain scaling analysis | `antstack_core.analysis.ce_neural_networks:run` | `uv run python scripts/complexity_energetics/analyze_neural_networks.py [--output DIR]` |
| `comprehensive_analysis.py` | Advanced complexity analysis with veridical reporting | `antstack_core.analysis.ce_comprehensive_analysis:run` | `uv run python scripts/complexity_energetics/comprehensive_analysis.py [--manifest PATH]` |
| `generate_comprehensive_analysis.py` | Module scaling and theoretical-limits report | `antstack_core.analysis.ce_module_scaling_analysis:run` | `uv run python scripts/complexity_energetics/generate_comprehensive_analysis.py [--manifest PATH]` |
| `generate_manuscript_figures.py` | Isolated enhanced manuscript figure family | `antstack_core.analysis.ce_manuscript_figures:run` | `uv run python scripts/complexity_energetics/generate_manuscript_figures.py --output NEW_DIRECTORY [--format png\|svg\|pdf]` |
| `generate_publication_figures.py` | Source-bound figures with attached native captions | `antstack_core.analysis.ce_publication_figures:run` | `uv run python scripts/complexity_energetics/generate_publication_figures.py --output NEW_DIRECTORY [--format png\|svg\|pdf]` |
| `generate_results_figures.py` | Distinct enhanced results figure family | `antstack_core.analysis.ce_results_figures:run` | `uv run python scripts/complexity_energetics/generate_results_figures.py --output NEW_DIRECTORY` |
| `generate_multipanel_figures.py` | Isolated multipanel family and actual LaTeX source | `antstack_core.analysis.ce_multipanel_figures:main` | `uv run python scripts/complexity_energetics/generate_multipanel_figures.py --output NEW_DIRECTORY` |
| `generate_tables_and_numbers.py` | Complete enhanced-model Markdown/JSON family and checksummed provenance | `antstack_core.analysis.ce_tables_and_numbers:main` | `uv run python scripts/complexity_energetics/generate_tables_and_numbers.py --output NEW_DIRECTORY` |
| `run_ce.sh` | Standalone canonical CE with an explicit fresh destination | `antstack_core.cli.ce` console script | `bash scripts/complexity_energetics/run_ce.sh NEW_OUTPUT_DIRECTORY [CE options]` |

Use `uv run python <script>` only when there is no console entrypoint for the workflow. Prefer adding a package-backed CLI in `antstack_core/cli` over adding business logic directly in this directory.

## Generated Artifacts

Scripts may write intentional figures, tables, and key-number files only to documented output roots. Canonical run artifacts belong under `outputs/<run_id>/`; paper compatibility artifacts belong under documented paper-local roots. Scripts must not create or retain `__pycache__`, local caches, or OS metadata.
