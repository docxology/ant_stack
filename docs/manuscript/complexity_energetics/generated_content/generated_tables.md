# Enhanced Analytical Model Tables

Enhanced analytical workload model, not canonical base-run evidence. Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, sensing power, device calibration or empirical validation. Mind max_policies=1000. Fits describe the displayed deterministic sweep, not independent samples.

### Table: Enhanced Workload Counters per 10 ms {#tab:module_complexities}

| Module | FLOPs/Decision | Memory Access (bytes) | Spikes |
|--------|----------------|-----------------------|--------|
| AntBody | 89543 | 10072 | 0 |
| AntBrain | 1.00524e+06 | 605280 | 100 |
| AntMind | 2.95e+06 | 156000 | 0 |

Enhanced analytical workload model, not canonical base-run evidence. Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, sensing power, device calibration or empirical validation. Mind max_policies=1000. Fits describe the displayed deterministic sweep, not independent samples.

### Table: Assumed Energy Coefficients (Not Device-Calibrated) {#tab:energy_coefficients}

| Component | Value | Units | Basis |
|-----------|-------|-------|-------|
| FLOPs | 1 | pJ/FLOP | Default model assumption, not measured calibration |
| SRAM | 0.1 | pJ/byte | Default model assumption, not measured calibration |
| DRAM | 20 | pJ/byte | Default model assumption, not measured calibration |
| Spikes | 1 | aJ/spike | Default model assumption, not measured calibration |
| Idle baseline per isolated module | 0.5 | W | Default model assumption, not measured calibration |

### Table: Descriptive Enhanced Model Fits {#tab:scaling_laws}

| Module | Parameter Values | Energy Exponent | Log-log R² | FLOPs Exponent | Log-log R² |
|--------|------------------|-----------------|------------|----------------|------------|
| AntBody | J: [6, 12, 18, 24, 36] | 1.7448e-05 | 0.806364 | 0.350245 | 0.9071 |
| AntBrain | K: [64, 128, 256, 512, 1024] | 2.63638e-06 | 0.870968 | 0.00493219 | 0.872138 |
| AntMind | H_p: [5, 7, 9, 11, 13] | 0.000313915 | 0.625076 | 1.32566 | 0.625076 |

Enhanced analytical workload model, not canonical base-run evidence. Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, sensing power, device calibration or empirical validation. Mind max_policies=1000. Fits describe the displayed deterministic sweep, not independent samples.

### Table: Conditional Erasure Reference Comparison {#tab:efficiency_analysis}

| Module | Modeled Energy (J) | Conditional Erasure Reference (J) | Model / Reference |
|--------|--------------------|----------------------------------|-------------------|
| AntBody | 5.000141e-03 | 2.570761e-16 | 1.945e+13 |
| AntBrain | 5.002873e-03 | 2.886011e-15 | 1.73349e+12 |
| AntMind | 5.003586e-03 | 8.469388e-15 | 5.90785e+11 |

Assumes one erased bit per FLOP at 300 K; FLOPs do not inherently determine bit erasures. No mechanical work is supplied. Ratios are not physical efficiencies or achievable improvement factors.

Enhanced analytical workload model, not canonical base-run evidence. Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, sensing power, device calibration or empirical validation. Mind max_policies=1000. Fits describe the displayed deterministic sweep, not independent samples.

### Table: Supported Contact Solver Model Counters {#tab:contact_solvers}

| Solver | Joints | Contacts | FLOPs/Solve | Memory (bytes) | Iterations |
|--------|--------|----------|-------------|----------------|------------|
| PGS | 18 | 5 | 6000 | 2832 | 10 |
| PGS | 18 | 10 | 12000 | 3072 | 10 |
| PGS | 18 | 15 | 27000 | 3312 | 15 |
| PGS | 18 | 20 | 48000 | 3552 | 20 |
| PGS | 18 | 25 | 75000 | 3792 | 25 |
| SI | 18 | 5 | 3744 | 3072 | 1 |
| SI | 18 | 10 | 5544 | 3552 | 1 |
| SI | 18 | 15 | 7344 | 4032 | 1 |
| SI | 18 | 20 | 9144 | 4512 | 1 |
| SI | 18 | 25 | 10944 | 4992 | 1 |
| LDLT | 18 | 5 | 11979 | 8712 | 1 |
| LDLT | 18 | 10 | 36864 | 18432 | 1 |
| LDLT | 18 | 15 | 83349 | 31752 | 1 |
| LDLT | 18 | 20 | 158184 | 48672 | 1 |
| LDLT | 18 | 25 | 268119 | 69192 | 1 |

Analytical workload estimates, not measured execution times or solver accuracy benchmarks.

### Table: Enhanced Model Sweep Configuration {#tab:param_ranges}

| Module | Varied Parameter | Evaluated Values | Fixed Parameters |
|--------|------------------|------------------|------------------|
| AntBody | J | [6, 12, 18, 24, 36] | {'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'} |
| AntBrain | K | [64, 128, 256, 512, 1024] | {'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100} |
| AntMind | H_p | [5, 7, 9, 11, 13] | {'B': 3, 'hz': 100, 'state_dim': 12, 'action_dim': 4, 'max_policies': 1000} |

These are selected model inputs, not measured biological ranges.
