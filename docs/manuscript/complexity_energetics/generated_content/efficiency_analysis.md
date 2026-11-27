### Table: Conditional Erasure Reference Comparison {#tab:efficiency_analysis}

| Module | Modeled Energy (J) | Conditional Erasure Reference (J) | Model / Reference |
|--------|--------------------|----------------------------------|-------------------|
| AntBody | 5.000141e-03 | 2.570761e-16 | 1.945e+13 |
| AntBrain | 5.002873e-03 | 2.886011e-15 | 1.73349e+12 |
| AntMind | 5.003586e-03 | 8.469388e-15 | 5.90785e+11 |

Assumes one erased bit per FLOP at 300 K; FLOPs do not inherently determine bit erasures. No mechanical work is supplied. Ratios are not physical efficiencies or achievable improvement factors.

Enhanced analytical workload model, not canonical base-run evidence. Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, sensing power, device calibration or empirical validation. Mind max_policies=1000. Fits describe the displayed deterministic sweep, not independent samples.
