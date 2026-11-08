# API Reference: Scientific Models, Workflow Results, and Render Boundaries

This is a module-level map, not a duplicate signature catalogue. [Public API contracts](public_api_contracts.md) and each module's `__all__` define the supported surface; inspect the implementing function for exact arguments and return types. The package version is `antstack_core.__version__`.

## Energy and Workload APIs Return Model Quantities

```python
from antstack_core.analysis import (
    ComputeLoad,
    EnergyCoefficients,
    ExperimentManifest,
    body_workload_closed_form,
    estimate_compute_energy,
)

load = body_workload_closed_form(0.01, {"J": 18, "C": 12, "S": 256})
compute_j = estimate_compute_energy(load, EnergyCoefficients())
```

This calculates a base-model compute estimate, not observed robot energy. `ComputeLoad` carries FLOPs, SRAM/DRAM byte counts, spikes, and optional timing/metadata. Coefficients use pJ/FLOP, pJ/byte, aJ/spike, and watts for power terms; energy functions return joules. Physical quantities must be finite and satisfy their domain restrictions.

`estimate_compute_energy` excludes idle, actuation, and sensing. Add these only under explicit, matched accounting assumptions using the relevant energy helpers. Module-isolated workload rows each include their own idle interval and must not simply be summed as a system total. Cost of transport requires locomotion energy matched to mass, gravity, and distance; canonical compute-only workload output does not provide it.

### Select Base or Enhanced Models Explicitly

| Family | APIs and interpretation |
| --- | --- |
| Base canonical | `body_workload_closed_form`, `brain_workload_closed_form`, `mind_workload_closed_form`; synthetic loop counterparts `body_workload`, `brain_workload`, `mind_workload`. The canonical manifest runner uses this family. |
| Enhanced legacy analytical | `EnhancedEnergyEstimator` and explicitly named enhanced workloads in `antstack_core.analysis.workloads`. They use different formulas/assumptions and are not drop-in numerical substitutes for the base family. |

The `estimate_*_compute_per_decision` helpers in `analysis.workloads` call the enhanced family for a fixed 0.01 s window. Do not silently mix their results with the canonical runner's common-window accounting.

## Statistical APIs Quantify the Supplied Values, Not Their Evidentiary Status

`bootstrap_mean_ci` and `analyze_scaling_relationship` support interval and descriptive scaling calculations. `StatisticalAnalyzer` provides further inferential/reporting methods. Confidence levels, sample counts, domain checks, and experimental design still matter. Repeating deterministic model values cannot supply independent empirical replication, and a fitted exponent is not a discovered biological law. Preserve paired x/y masks when filtering plots. See [scientific validation](scientific_validation.md) for interpretation.

## Power Meters Preserve Observation Status

`NullPowerMeter`, `RaplPowerMeter`, `NvmlPowerMeter`, `measure_energy`, and `create_power_meter` are available through `antstack_core.analysis`. Null explicitly disables observation; its zero-valued internals are not measured zero energy. RAPL observes a CPU-package cumulative counter on supported Linux hardware; NVML observes supported NVIDIA GPU power. Neither automatically measures the whole system.

Unavailable requested hardware raises `PowerMeasurementUnavailable` from `antstack_core.analysis.power_meters` or yields an explicitly unavailable sample where specified; it is not silently replaced with empirical zeros. The canonical runner keeps `energy_est_j`, `measured_energy_j`, and `measurement_status` separate and rejects hardware observation for closed-form workloads.

## Orchestration APIs Bind Inputs to Current Outputs

```python
from antstack_core.orchestration import RunAllConfig, run_all

config = RunAllConfig.from_file("configs/run_all_antstack.example.yaml")
preview = run_all("configs/run_all_antstack.example.yaml", validate_only=True)
```

This example assumes the checkout's caller-owned inputs are available from the current directory. Relative run-all input/output paths use the current working directory, not the config file's parent. Validation returns a `RunAllResult` with `validate_only=True`, no manifest/provenance paths, and no artifacts; generation returns the published run location and artifact records.

For standalone CE use the package runner `antstack_core.orchestration.complexity_energetics.run_manifest(manifest_path, out_dir)`. Its `RunManifestResult.output_paths` is a current-invocation inventory. `project_result` publishes that inventory into a separate paper tree; it is not a way to designate arbitrary old files as newly generated. Use fresh, disjoint, caller-owned roots. Full CLI and ownership rules are in [public API contracts](public_api_contracts.md) and [reproducibility](reproducibility.md).

## Figure APIs Require Paired Data and Real Renderers

`antstack_core.figures` exports `PlotConfig`, `bar_plot`, `line_plot`, `scatter_plot`, `interval_plot`, publication plotting helpers, asset helpers, and cross-reference validation. `interval_plot` preserves supplied categorical bounds and uses explicitly labelled scales/precision for wide positive ranges. `validate_mermaid_syntax` performs lexical checks only. `preprocess_mermaid_diagrams` renders active Mermaid fences with a local CLI/browser when an output directory is supplied; omitting the directory leaves text unchanged, not rendered. An explicit remote service URL is optional at the Python API and is not an automatic fallback.

A renderer failure is an error, not permission to reuse an old diagram. See [troubleshooting](troubleshooting.md) for supported environment variables and browser security boundaries.

## Publishing APIs Return Observed Build Results

`antstack_core.publishing` exports `PDFGenerator`, `PDFGenerationConfig`, `BuildResult`, template, quality, reference, build-orchestration, and provenance helpers. `PDFGenerator.generate_pdf` consumes ordered Markdown paths, resolves images relative to their containing source, preserves literal code/math/URI contexts, and returns a result whose `success` requires a newly produced PDF. Inspect `error_message`, `warnings`, and source/resource hashes in result metadata.

`validate_pdf_environment` reports executable/package availability; it does not render a document or validate every font/resource. Explicit missing templates, missing images, undefined references/citations, and missing glyph diagnostics must be repaired rather than suppressed. The installed configured-paper CLI delegates to `antstack_core.publishing.paper_builder.ModularPaperBuilder`, with caller-selected project inputs.

## Empirical Helpers and Structural Contracts Stay Scoped

- `antstack_core.cohereants` exports physical conversions, `SpectralData`, `CHCAnalyzer`, `BehavioralData`, `BehavioralAnalyzer`, and related analyses/plots. Helper availability is not evidence that a dataset was acquired or a biological conclusion validated; follow [external data integration](external_ant_data_integration.md).
- `antstack_core.architecture` models and checks repository structure and declared exports, not installed-package access to manuscript trees.
- `antstack_core.mathematics` handles publication normalization and labels; it is not a symbolic or physical simulation engine.

Use [validation](validation.md) for acceptance checks rather than treating examples on this page as executed verification.
