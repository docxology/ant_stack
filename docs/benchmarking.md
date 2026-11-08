# Benchmarking: Supported Workloads and Source-Bound Measurements

A benchmark is an observed measurement of an identified operation under recorded conditions. Ant Stack's model outputs and retained performance samples are not by themselves evidence of current execution speed, whole-system energy, or biological performance. No speedup is asserted here.

## Choose an Implemented Operation and State the Measured Boundary

| Target | What can be measured | What it does not establish |
| --- | --- | --- |
| Base closed-form Body/Brain/Mind functions | Time/allocation cost of evaluating the chosen accounting formulas for fixed inputs | Real robot execution, measured FLOP throughput, or empirical energy |
| Base synthetic `loop` workloads | Runtime-bound Python loop behavior and accumulated model counters | Actual contact solving, neural simulation, active-inference control, or hardware instruction counts |
| Enhanced legacy estimator/workloads | Evaluation cost of that separately identified analytical family | Equivalence to the canonical base family or a fair substitution baseline |
| Statistical/plotting functions | Time/memory of the chosen function for a fixed input size and method settings | Scientific validation of the supplied data or fitted relationship |
| `antstack-ce` generation | End-to-end manifest loading, workload calculation/execution, figures, CSV, and provenance | A kernel-only timing or full robot-system performance |
| `run-all-antstack` | The selected generation tasks, filesystem work, visualizations, optional rendering/copying, and publication | A pure compute benchmark or a source-independent comparison |
| `antstack-build --project-root ... --validate-only` | Current source/config/reference validation and, for all-paper mode, report writing | Native Mermaid/Pandoc/XeLaTeX render speed or readiness proof for all resources |
| Actual configured paper render | Current ordered source processing, dependencies, assets, and PDF production | Equivalence to validation-only or historical PDF copying |

The manifest runner supports only `body`, `brain`, and `mind`, each in `closed_form` or `loop` mode. Contact-solver names in analytical parameterization do not instantiate a physics engine. There is no implemented MuJoCo/FlyBody robot, whole active-inference agent, or closed colony-loop benchmark to report.

## Separate Wall Time, Model Energy, and Hardware Observations

External timing must wrap exactly the operation being compared; record whether import, warm-up, compilation/cache, rendering, disk I/O, and publication are included. A requested loop duration is not a fixed amount of scientific work: loops accumulate counters until a wall-time deadline, so different host loads/interpreters can produce different counts. Those counters are not hardware performance-counter observations.

For closed-form runs, `duration_s` is the modeled workload window, not the execution time needed to evaluate the formula. `energy_est_j` comes from workload counts and coefficients, not from the timer. Keep compute, isolated idle, and any separately modeled actuation/sensing terms labeled with their units and interval.

RAPL and NVML observations require available supported hardware/backends and an appropriate loop workload. RAPL is CPU-package scoped and NVML is GPU scoped; neither is automatically whole-system energy. Record the measured domain, sampling/counter method, baseline policy, and interval. Null disables measurement; unavailable hardware is not zero energy. Do not present `measured_energy_j` and modeled estimates as interchangeable. See [API reference](api_reference.md).

Memory measurement requires an explicit measurement method and scope: peak process resident memory, Python allocations, and allocator-retained memory are different quantities. The run-all artifact inventory is not a memory profiler, and generated energy statistics are not a performance benchmark report.

## Record a Comparison Contract Before Interpreting Results

For each measurement retain:

- Exact command or function call, selected tasks, model family, parameters, input identities, units, and output destination.
- Source revision and dirty state; package, Python, numerical-library, JavaScript, browser, renderer, and native-tool versions relevant to that operation.
- Hardware/OS, process/thread configuration, power mode, competing load, and whether caches/imports/warm-up were included.
- Repetition design, raw observations, timer/memory/meter boundary, exclusions/failures, and the aggregation/uncertainty method.
- Current producer inventory and provenance, with any historical copies clearly distinguished from newly rendered/generated material.

Use matched workloads, dimensions, durations, model semantics, and environment where possible. A difference in model formula, interpreter, backend, configured tasks, or cached state is a confound, not automatically an optimization. State uncontrolled differences rather than attributing them to code changes.

## Repetition and Scaling Need Honest Interpretation

Seed bootstrap/statistical operations where supported, but do not claim that a seed makes timed loops or host scheduling deterministic. Preserve the distinction between repeated software timings and repeated deterministic model outputs. Zero-width intervals over identical model values say nothing about empirical precision.

Report fitted exponents as descriptive properties of the specified calculations/range, including validity diagnostics. Do not infer a universal biological law from a fit or describe an inverse-parameter proxy as a nondominated Pareto front. Illustrative animations reveal fixed modeled values; animation duration is not experiment duration.

## Use Canonical Artifacts Without Inventing Benchmark Evidence

[Getting started](getting_started.md) gives explicit generation commands and fresh-root rules. [Reproducibility](reproducibility.md) defines snapshots, source state, relative artifact identities, and inventories; [validation](validation.md) defines software acceptance gates. These workflows can supply identifiable inputs/outputs for a measurement, but their successful execution alone does not establish a speedup or scientific result.

The development dependencies include `pytest-benchmark`, but its presence is not evidence that every public API has a current benchmark or that a retained report was regenerated. Repository tests are correctness checks unless a specific test explicitly measures and reports performance. Publish only measurements actually collected for the cited source/runtime state; label older retained samples historical and do not extrapolate speedups from them.
