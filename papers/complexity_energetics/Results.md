# Analytical Results and Evidence Boundaries

## Provenance and Conditional Status of Generated Results {#current-numerical-results}

The numerical result section is `Generated.md`, produced from the native manifest by the package CE analysis. Its tables and figures must belong to one completed governed run. This prose intentionally does not duplicate fitted exponents, confidence intervals, energies, component percentages, or cost-of-transport values that would become stale when the manifest or implementation changes. Checked-in generated material and retained PDFs are historical until the release owner generates and inspects a coherent fresh result set.

The example manifest uses deterministic closed-form workloads and a null meter. Thus its reported energy is an **analytical model output conditional on assumed coefficients**, not an actual measured result. Repeating those formulas 100 times and using 1000 bootstrap resamples does not create independent physical observations, calibrate coefficients, or demonstrate statistical significance.

## Reading the Generated Outputs

- **Body:** identify `pgs`, `si`, or `ldlt` and the base-model dispatch. A joint sweep includes contact and quadratic control terms; no fixed near-zero exponent or 20-contact timing boundary is asserted.
- **Brain:** identify the supported `random` or `structured` pattern, activity fraction, population size, and channel range. The output is not a comparison of biological, small-world, or scale-free networks.
- **Mind:** the base model uses the exact tree-node sum with $B$ and $H_p$; the separate enhanced model's finite policy budget must not be attributed to a base sweep. A finite power-law approximation to tree growth is descriptive, not proof of a critical exponent or a 15-step feasibility limit.
- **Uncertainty:** say what was resampled and which statistic the interval bounds. A total-over-repeats bar needs an interval for the total, not the mean. Zero-width intervals for identical deterministic inputs are legitimate and do not describe device uncertainty.
- **Energy–proxy comparisons:** identify objective directions and the proxy definition; neither an inverse dimension nor a sorted curve demonstrates measured task performance or a nondominated optimum.

## Unit and Accounting Interpretation

Window energy, energy per decision, and mean power are different quantities. For duration $t$ and frequency $f$, normalize by the actual modeled decision count $\lfloor ft\rfloor$ when nonzero. Match the integration window for compute, sensing, mechanics, and baseline before adding them. Rows are module-isolated compute-plus-idle estimates and must not be summed as a system total. The per-decision system table derives the same base compute models, adds assumed physical actuation and sensing, and allocates one shared idle baseline.

The example's 0.50 W baseline contributes 0.005 J (5 mJ) in a 10 ms interval. This is a dimensional illustration of an **assumed input**, not a generated or measured module result. Body compute must not disappear from a body breakdown, and a compute-only component sum must not be renamed actuation energy. A conceptual Mind layer is not physically energy-free: its implemented policy-evaluation counters incur modeled compute energy wherever they run.

Cost of transport is $E/(mgd)$, with all energy in the numerator integrated over the same traversal as distance $d$. Declare whether $E$ includes actuation, sensing, compute, and idle load. The native run reports CoT as unavailable: its supplied distance has no matched modeled locomotion interval. A workload-window compute estimate alone is not whole-body locomotion CoT. At fixed speed, changing duration also changes distance; at fixed distance, a longer duration can legitimately change CoT because idle energy accumulates. A mass of 0.001 kg equals **1 g**, not 1 mg; 1 mg is $10^{-6}$ kg. No species identity follows from a nominal platform mass.

## Physical References Are Not Efficiency Validation

Landauer's $k_BT\ln 2$ is a reference per logically irreversible erased bit under its thermodynamic assumptions [@landauer1961irreversibility]. At approximately room temperature it is of order $10^{-21}$ J/bit. It is not a calibrated per-FLOP energy and cannot be divided directly into total workload energy without an explicit compatible erasure count and boundary. Separate mechanical work and bit-erasure references are not interchangeable minima.

A value below a stated lower reference is an accounting/applicability question, not evidence of near-optimality. The manuscript makes no retained $4.2\times10^8$ or $2.7\times10^6$ optimization-factor claim, no below-bound efficiency conclusion, and no universal actuation-dominance percentage.

## Biological Comparison: Data Required

No independent raw ant energetics or neural-energy dataset has been identified for this release's native workflow. The earlier *Formica rufa* metabolic rates, ant CoT range, mushroom-body power range, universal mass/neuron exponents, muscle/motor comparisons, and derived biological-to-robotic improvement ratios are not retained as established results. The named literature and configured robot assumptions did not supply traceable matched evidence for those claims.

A biological comparison requires a primary dataset with species/caste, wet/dry mass, temperature, locomotor speed and distance, metabolic/measurement boundary, sample independence, and uncertainty. Robot power-model studies [@jaramillo2020energy] are relevant methodological context, not ant validation. Dataset ingestion and a held-out comparison are prerequisites, as described in the external-data guide; passing source or render checks does not satisfy them.

## Proposed Follow-On Measurements

Measure workload latency and deadline misses on a named device before claiming 100 Hz feasibility. Calibrate operation/memory prices and verify complete system accounting on held-out workloads before claiming prediction accuracy. Test sparse representations and policy budgets against task outcomes before claiming an accuracy–energy advantage. Sobol/Morris analyses, power calculations, and critical-behavior experiments remain proposed studies rather than completed results of this run.
