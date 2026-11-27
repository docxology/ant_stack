# Abstract

We propose the Ant Stack: a compact, modular architecture for studying embodied agents through three conceptual layers—`AntBody` (morphology, actuation, sensors), `AntBrain` (functional abstractions inspired by insect circuits), and `AntMind` (active-inference and stigmergic decision-making). Explicit interfaces and sparse-processing assumptions organize the design; they do not establish a complete ant emulation, measured low-power implementation, or validated species preset library.

The proposed methods connect physical and sensory representations to neural-processing and policy-selection models. Navigation, trail following, task allocation, and adversarial-robustness tasks are evaluation protocols, not completed empirical results in this manuscript. Timing, neuron counts, and horizon ranges in the sections below are declared engineering targets or assumptions, not biological measurements or demonstrated real-time capabilities.

The contribution is an architecture and a set of testable interface/model proposals. The companion complexity–energetics workflow supplies analytical counter-based estimates under illustrative coefficients; it does not establish calibration, species validity, or biological efficiency. Independent data, implemented end-to-end task evaluation, and device measurements are required before claiming replication, accuracy, resilience, transfer, or energy superiority.

For myrmecology, this separation offers a way to formulate hypotheses about sensorimotor and collective organization while keeping literature observations distinct from model assumptions. For synthetic systems, swarm robotics and cognitive security are proposed applications, not demonstrated deployment or alignment guarantees.

## Keywords

Myrmecology; ant colony; stigmergy; active inference; Antennal Lobes; Mushroom Bodies; Central Complex; sparse processing; embodied AI; swarm robotics; cognitive security.

## Figure: Proposed Architecture and Implemented Analytical Boundary {#fig:ant_stack_graphical_abstract}

![Original conceptual schematic distinguishing proposed information flow from the implemented workload-count models. It contains no measured data or numerical results; the source and output identities are recorded in the graphical abstract provenance receipt.](../graphical_abstracts/ant_stack_graphical_abstract.png){width=100%}
