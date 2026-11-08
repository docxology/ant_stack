# Best Practices: Explicit Models, Safe Artifacts, and Evidence-Bound Claims

Keep implementation, documentation, and artifacts aligned with the supported [public API contracts](public_api_contracts.md). Use [validation](validation.md) for acceptance gates and [reproducibility](reproducibility.md) for provenance requirements rather than maintaining competing command checklists.

## Put Reusable Behavior in Package Modules

- Prefer `antstack_core` implementations with thin CLI/script adapters and explicit failure propagation.
- Read local directory contracts and reuse existing APIs before introducing another workflow convention.
- Validate finite values, units, physical domains, and configuration semantics before executing producers.
- Keep installed behavior independent of this repository's scripts, tests, papers, and current source-tree location. Caller projects provide their own inputs.
- Update affected callers, tests, and guide links when a public interface changes; a wrapper must not conceal failure or invent a successful result.

## Name the Model and Match the Accounting Interval

Use the base canonical and enhanced legacy families as distinct analytical models. Record the actual function/family, revision, coefficients, workload parameters, and duration. Do not compare or substitute results merely because both describe Body, Brain, or Mind.

Label estimates, synthetic counters, and hardware observations separately. Keep `compute_energy_j`, isolated idle, and measured energy distinguishable. Never sum isolated workload rows with repeatedly counted baseline as if they were one system budget. Match common-window components and count shared idle once; report the interval and which modules actually make a decision at that frequency. Cost of transport is unavailable without matched locomotion energy/distance, not a reason to reuse compute energy as locomotion energy.

### Enhanced key-number input migration

`KeyNumbersLoader(path)` accepts the complete enhanced `ce_tables_and_numbers` output: machine `schema_version: 1`, `model_family: enhanced_analytical`, nonempty provenance notes, all assumed coefficients and four numerical sections travel together. Historical subsets, unitless energy aliases, mixed families and contradictory accounting are rejected. Human prose is not a schema/version authority; changing its punctuation does not change the model identity. This producer's default-coefficient contract still requires those actual defaults; changed assumptions need a correspondingly revised producer contract.

With no path, only the caller's `papers/complexity_energetics/generated_content/key_numbers.json` is considered. There are no loose-file or canonical-run fallbacks, no package-tree discovery and no process-global project cache. Installed callers provide their own enhanced inputs. `PaperKeyNumbersUpdater(project_root=..., json_path=...)` binds to that project or explicit input instead of an unrelated working-directory/global loader.

Missing paths raise `KeyError`; unavailable fit estimates remain `None` and cannot be numerically rendered. Missing module metrics never become zero. Summaries expose `total_energy_mj` and `body_scaling_regime`; scaling analysis contains only the supplied `scaling_exponents`, not fabricated Brain/Mind/system fits. Manager edits must supply every affected energy total and power together; incoherent edits leave the file unchanged. To update generated products, regenerate the complete enhanced family rather than independently hand-editing related outputs.

`validate_data` and `validate_key_numbers` check complete input/schema/accounting coherence only. The latter explicitly reports `validation_scope="enhanced_model_input_coherence"`, the supplied assumptions, and false empirical/canonical/source-freshness flags. Even coherent values and matching provenance text cannot establish a current source-bound generation, empirical validity or device calibration; those require separate producer/evidence custody. Keep this enhanced input separate from canonical base-run projections.

## Publish Only into Trusted, Explicit Destinations

Standalone CE generation always names a fresh `--out` root; a nonempty earlier result is immutable to that command. Canonical run-all artifacts belong under the selected `outputs/<run_id>/` generation. Use paper projections only when a consumer needs them and acknowledge that projection updates a paper tree as a separate publication operation.

Respect symlink, overlap, Git/runtime, ownership, and completion checks. Existing owned generations are archived on authorized replacement; do not delete markers or bypass refusal to reuse arbitrary trees. Keep sources and destination namespaces disjoint and exclusively controlled for the whole invocation. Retain failure evidence without promoting partial/staging output to a completed generation.

Use the producer's current-invocation inventory, snapshot, and content identities rather than assuming every file in an output directory came from the latest run. Historical PDFs and retained sample artifacts remain historical until a current source-bound producer result demonstrates otherwise.

## Keep Documents Literal and Rendering Failures Visible

Use informative headings and modular links. Preserve literal code, mathematics, and URIs; fix references and bibliography keys at their definitions, not by deleting checks. Figures need actual paired data and meaningful units; masks must preserve x/y pairing, and zero-valued samples must not silently become missing data.

Store concise source-controlled Mermaid diagrams where useful. Lexical checks are not render proof. Native rendering requires a real local CLI/browser and working sandbox; missing dependencies and sandbox failures must remain visible. Do not disable browser security, suppress renderer warnings, reuse stale images as successful renders, or infer success from a prepared remote workflow.

Distinguish configured manuscript rendering, canonical generated-results PDF production, and copying retained historical PDFs. Publish a new PDF only from a successful current render, preserving source/resource binding and meaningful diagnostics.

## Make Scientific and Benchmark Claims Proportionate to Evidence

Current workload producers are analytical/synthetic. A narrow bootstrap interval over repeated deterministic values is not empirical precision; a fitted scaling curve is descriptive model output, not an established biological law; an inverse-parameter illustration is not a Pareto frontier. Hardware unavailability must remain typed/unavailable, never a fabricated zero observation.

Do not describe proposed MuJoCo/FlyBody integration, whole active-inference control, or a colony loop as implemented. Label conceptual diagrams and proposed experiments accordingly. For empirical claims, cite primary evidence and preserve dataset provenance under [external data integration](external_ant_data_integration.md) and [scientific validation](scientific_validation.md).

Benchmark only identified methods/workflows under recorded source/runtime conditions; historical samples cannot establish current speedups. See [benchmarking](benchmarking.md). Report only checks actually exercised, including their failures and limitations; a source version bump does not establish a published release.
