# Abstract

We present an analytical framework for studying computational complexity and energy accounting in the Ant Stack, a proposed Body–Brain–Mind architecture for embodied artificial intelligence. Closed-form workload counters connect contact-resolution, sparse neural-processing, and policy-evaluation models to explicitly assumed energy coefficients. A nominal 100 Hz update rate defines a 10 ms accounting interval; it is not a measured scheduling guarantee.

The selected manifest uses the base workload family: contact models named `pgs`, `si`, and `ldlt`; neural patterns named `random` and `structured`; and an exact policy-tree model parameterized by branching factor $B$ and horizon $H_p$ in decision steps. The separately available enhanced Mind model bounds the policy count by `max_policies` (default 1000), with `None` selecting exact enumeration. These are different analytical models, not interchangeable evidence of bounded-policy performance.

Energy is estimated from operation counts, memory traffic, spikes, and baseline power, with mechanical and sensing terms reported on an explicitly matched time basis. The coefficients are illustrative assumptions, not measurements calibrated to a named device. Current numerical tables and fitted relationships belong in the run-generated `Generated.md`, rather than duplicated constants in this prose. Bootstrap resampling describes the supplied model values; repeated deterministic calculations do not establish physical uncertainty or statistical power.

The contribution is a transparent model and reproducible analysis interface, not empirical validation of ant physiology, hardware prediction accuracy, real-time feasibility, critical exponents, or species-specific behavior. Independent measurements and a declared validation protocol are required before making those claims. The model provides a basis for asking how solver choice, neural activity, and policy budgets affect an assumed computational energy budget, while keeping implementation checks distinct from biological evidence.

## Figure: From Workload Assumptions to Matched Energy Accounting {#fig:complexity_energetics_graphical_abstract}

![Original analytical schematic: assumed dimensions and energy prices produce model estimates, not empirical calibration. Module-isolated rows and common-window system components have different accounting boundaries. Source and output identities accompany the graphical assets.](../graphical_abstracts/complexity_energetics_graphical_abstract.png){width=100%}
