# Methods for Analytical Workload and Energy Estimation {#methods}

## Methodological Framework and Evidence Classes

This study uses deterministic analytical workloads, not a calibrated energy-measurement campaign. Retained quantities have the following provenance:

- **Assumptions:** workload dimensions, update rates, solver/pattern choices, memory splits, mechanical and sensor power coefficients, and digital energy prices in the manifest.
- **Analytical model outputs:** counters, coefficient-based energy estimates, descriptive fits, and generated plots from that manifest.
- **Literature context:** explicitly cited results from other systems; these do not calibrate this implementation.
- **Actual measured results:** none identified for the native example manifest, whose meter is null.
- **Proposed protocols:** the calibration, sensitivity, and external-validation procedures below; these have not been performed by the example workflow.

Software tests can establish implementation behavior against specified mathematical expectations. They do not establish biological realism, cross-platform timing, measurement accuracy, or publication readiness.

## Computational Workload Modeling

### Selected Base Workloads

The package orchestration module `antstack_core.orchestration.complexity_energetics` selects the base closed-form Body, Brain, and Mind functions. A window of duration $t$ at update frequency $f$ contains $n=\lfloor ft\rfloor$ modeled updates. Per-update counters are multiplied by $n$; byte counters represent modeled traffic over updates, not automatically peak resident memory.

**AntBody:** `calculate_contact_complexity` accepts only `pgs`, `si`, and `ldlt`. They are operation-count models, not solver accuracy or convergence benchmarks. For `pgs`, the iteration convention is $I=\min(50,\max(10,C))$ and contact FLOPs are $IC(6J+12)$. For `si`, FLOPs are $20CJ+J^3/3$. For `ldlt`, FLOPs are $(J+3C)^3/3$. Base Body adds $10S$ sensor FLOPs and $20J^2$ control FLOPs per update. No special transition at 20 contacts is inferred from these formulas.

**AntBrain:** the supported patterns are `random` and `structured`. For $N_a=\lfloor\rho N_{KC}\rfloor$, their neural FLOP models are $N_a^2$ and $N_a\log N_{KC}$ respectively. Base Brain adds $100K$ channel FLOPs and $4KH$ channel bytes per update. These labels do not implement small-world, scale-free, or species-specific biological topologies. The configured $\rho=0.02$ is an assumed activity fraction, not an inferred optimum.

**AntMind:** base planning uses canonical parameters $B$ and $H_p$, with the exact node count
\begin{equation}
N_{\mathrm{tree}}=\sum_{h=0}^{H_p}B^h.
\end{equation}
Belief, policy, and evidence-bound counters scale with this node count and the configured state/action dimensions. $H_p$ is measured in decision steps, not seconds. The model counts work; it does not execute or assess policy decisions.

### Separate Enhanced Models

Enhanced Body adds a more detailed sensor/control decomposition; enhanced Brain separately represents AL, MB, CX, and plasticity counters. These must not be used to interpret a base-model sweep without identifying the change of model. Enhanced Mind uses $P=\min(B^{H_p},M)$ policies for a finite `max_policies=M` (default 1000), uniformly across horizons; `max_policies=None` uses $P=B^{H_p}$. This leaf-policy convention differs from the base model's sum of tree nodes. Bounding policy counts does not demonstrate preserved decision quality. Hierarchical overhead is a model assumption, not an observed acceleration.

## Energy Estimation and Time Accounting

The manifest is the numerical input authority for illustrative coefficients. Compute energy over a window is
\begin{equation}
E_{\mathrm{compute}}=10^{-12}(N_F e_F+V_S e_S+V_D e_D)+10^{-18}N_{\mathrm{spk}}e_{\mathrm{spk}},
\end{equation}
where FLOP/byte prices are in pJ and the spike price is in aJ. Baseline energy is $P_0t$. Mechanical and sensing power estimates must be integrated over the same window before any sum or cost-of-transport calculation. A component must appear only once within the declared system boundary; separate module estimates with their own baseline are not automatically a shared-controller total.

Each result row reports module-isolated compute plus its own idle contribution over that workload window. These rows are not additive shared-controller system energy. The separate per-decision system table uses the same selected base compute models, assumed physical actuation/sensing, and one shared idle baseline at the configured update rate. Cost of transport is unavailable because the supplied distance has no matched modeled locomotion interval.

The generated report must identify whether a statistic is a workload-window energy, a total over repeats, a mean over repeats, or energy per decision. For $n>0$, window energy divided by $n$ yields energy per modeled update; power is window energy divided by $t$. A 0.5 s workload at 100 Hz contains 50 updates, not one decision. At 0.50 W, baseline is 0.25 J over that window and 0.005 J over 10 ms. Do not compare these as equal intervals.

## Descriptive Scaling Fits and Resampling Limits {#scaling-and-uncertainty}

Log–log regression fits a descriptive finite-range power law $y=ax^b$ to positive finite paired values. A large exponent $b$ is not itself an exponential law $y=a\exp(cx)$, and a goodness-of-fit score is not physical validation. Report the parameter range, response boundary (compute-only versus baseline-inclusive), model family, and fit limitations with each result. For a constant response, the descriptive exponent is zero but the variance-based goodness-of-fit $R^2$ is undefined (reported as `None`/unavailable), not evidence of a perfect fit or superiority.

The example requests 100 closed-form repeats and 1000 bootstrap resamples at confidence level 0.95. The former repeat identical formulas for fixed inputs; the latter resample the supplied values. Neither count is an independent empirical sample size, and neither guarantees power greater than 0.8. Identical values have a degenerate resampling interval; no artificial width should be added. Parameter uncertainty and device error require separate justified input distributions or measurements.

Sobol indices, Morris screening, Bayesian posterior calibration, held-out hardware prediction error, and cross-laboratory replication have not been established by this manifest. A future sensitivity analysis must specify parameter distributions, dependence assumptions, sampling design, and evaluated responses. A future power analysis must specify an estimand, independent sampling unit, effect size, variance, test, and error rates.

## Proposed Measurement and Calibration Protocol

Before describing coefficients as calibrated, identify a device and measured rails, run operation/memory/spike microbenchmarks with documented instrumentation, record temperature and utilization, and fit coefficients with uncertainty. Validate on independent held-out workloads. Avoid transferring processor-generation trends [@koomey2011energy] into a device-specific calibration claim.

RAPL/NVML telemetry and external meters are possible instruments, not evidence that a measurement was taken. Record counter range/wrap behavior, monotonic timestamps, sampling bandwidth, synchronization error, firmware, and meter calibration records. Specify whether sensing, actuation, idle power, and host computation are included. No WT310 accuracy, 100 Hz meter synchronization, dynamometer result, 7 nm circuit calibration, or attojoule-scale measurement resolution is claimed here.

Independent ant validation additionally needs species/caste, colony and individual identity, mass definition, temperature, behavioral context, raw records, licenses, and held-out comparisons. General robot modeling literature [@jaramillo2020energy] and compute-aware planning [@sudhakar2020balancing] motivate the approach but do not validate these ant or hardware models.

## Reproducibility and Publication Boundary

Retain the exact manifest, package/environment versions, invocation, model selection, result identity, and artifact provenance. `Generated.md` is the current numerical-result section only when produced by the same completed governed run as the selected figures and tables. Retained older PDFs, concatenations, and plots are historical artifacts, not proof of a fresh successful run. Rendering, cross-reference checks, and numerical tests are separate acceptance steps; none substitutes for empirical calibration.
