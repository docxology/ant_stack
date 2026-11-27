# Appendices

## A. Energy Input Assumptions

The manifest is the numerical source of truth. The following table records its illustrative default input prices, not calibrated devices or literature measurements. A changed manifest requires fresh generated results, not manual repinning of prose results.

## Table: Illustrative Energy Inputs {#tab:energy_inputs}

| Input | Manifest value | Units | Evidence class |
|---|---:|---|---|
| FLOP price | 1.0 | pJ/FLOP | Assumption |
| SRAM traffic price | 0.10 | pJ/byte | Assumption |
| DRAM traffic price | 20.0 | pJ/byte | Assumption |
| Spike-event price | 1.0 | aJ/spike | Assumption, not circuit measurement |
| Idle power | 0.50 | W | Assumption |
| Electrical actuator power | 2.0 | W/joint | Assumption |
| Sensor power | 0.005 | W/channel | Assumption |

Landauer's $k_BT\ln 2$ [@landauer1961irreversibility] is a separate physical reference per erased bit at specified temperature, not one of these device prices. At 295 K it is approximately $2.82\times10^{-21}$ J/bit. The transport convention uses $g=9.81\,\mathrm{m/s^2}$ as a nominal gravitational acceleration, not an observed platform property. A mass of 0.001 kg is 1 g, not 1 mg.

## B. Proposed Measurement Protocol

The following are requirements for a future calibration campaign, not procedures performed by the native null-meter example:

- Identify devices, measurement rails, instruments, firmware, sampling resolution, counter ranges/wrap rules, synchronization error, and calibration certificates.
- Specify environmental targets before acquisition. For example, 20 ± 2 °C and 45 ± 5% relative humidity would be protocol choices, not recorded conditions in this release.
- Preserve timestamped raw traces and benchmark inputs; report software and hardware versions.
- Distinguish independent devices/sessions/animals from repeated deterministic calculations or repeated readings on one unit.
- Fit coefficient uncertainty and evaluate held-out workloads; report measurement failure/unavailable status rather than fabricated zeros.
- For locomotion, align energy, duration, distance, mass, and physical boundary. Do not infer CoT from an unrelated workload window.

## C. Parameterization and Model Selection

The example's morphology, population, activity, and horizon ranges are illustrative engineering dimensions. They are not a species library, universal operating limits, or inferred physiological optima. A hexapod with 18 modeled joint degrees of freedom is a six-by-three design assumption.

Selected CE workloads are **base** Body/Brain/Mind. Supported solvers are `pgs`, `si`, `ldlt`; supported neural patterns are `random`, `structured`. Base Mind uses canonical $B,H_p$ and sums exact tree nodes. Enhanced Mind is separate: finite `max_policies` (default 1000) bounds leaf policies uniformly; `None` requests exact leaf enumeration. $H_p$ is in steps; seconds require an explicit step duration.

## D. Reproducibility and Acceptance Requirements

- Preserve manifest, invocation, model family, environment, package revision, result identity, and generated-artifact provenance.
- Require a completed coherent run before projection into paper-local compatibility paths; do not reuse concatenation snapshots or retained PDFs as fresh proof.
- Reconcile per-window rows with per-decision tables using the configured update count and explicit baseline allocation.
- Keep analytical results, assumptions, literature data, actual measurements, and proposed protocols distinct in captions/tables.
- Bootstrap effort (1000 resamples) and deterministic repetition (100 repeats) do not establish independent sample size or guaranteed power.
- Rendering and mathematical regression checks are implementation acceptance, not external scientific validation.

## E. Notation

| Symbol | Description |
|---|---|
| $J,C,S$ | Joint DOF, contacts, sensor channels |
| $K,N_{KC},\rho,H$ | Input channels, population, activity fraction, state dimension |
| $B,H_p$ | Branching factor, policy horizon in steps |
| $G,E$ | Proposed field cells and deposits per update |
| $e_F,e_S,e_D$ | FLOP and SRAM/DRAM traffic prices in pJ |
| $E_{\mathrm{spk}}$ | Spike-event price in aJ/spike |
| $P_0$ | Idle power in W |
| $t,f$ | Window duration in s, update rate in Hz |
