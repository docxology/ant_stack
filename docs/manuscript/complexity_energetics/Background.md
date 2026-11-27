# Background and Research Context

## Research Motivation and Scope

Embodied systems connect computation, sensing, and physical action. Their energy budgets depend on algorithmic work, memory traffic, hardware operating conditions, and mechanical tasks. The Ant Stack provides a conceptual Body–Brain–Mind decomposition for asking how these contributions interact. This companion study analyzes closed-form workload models; it does not report a calibrated robot or validated ant emulation.

The central questions are which configured dimensions drive modeled work, how assumed operation and power prices affect the energy budget, and which additional measurements would discriminate competing design choices. A nominal 100 Hz update target is not an observed control capability. Neither contact count 20 nor planning horizon 15 is a hardware-independent feasibility boundary.

## Energy-Aware Robotics and Computational Co-Design

Platform-specific robot models must specify motion, payload, and hardware. Jaramillo-Morales and colleagues model differential-drive mobile robots and validate on their identified platform [@jaramillo2020energy]; this supports the importance of platform-specific validation, not an accuracy claim for this framework.

Compute Energy Included Motion Planning (CEIMP) explicitly weighs further planning energy against potential actuation savings [@sudhakar2020balancing]. It motivates treating compute and actuation together. The present CE sweep counts work and does not itself implement or evaluate CEIMP's stopping decisions.

Historical improvements in computing's electrical efficiency [@koomey2011energy] provide context for energy-aware design. They do not calibrate our FLOP, memory, or spike coefficients. Device-level or sparse-processing advantages remain workload-, technology-, and boundary-dependent; a generic attojoule spike price is an assumption here.

## Architectural Context

**AntBody** represents morphology, contacts, sensing, and actuation. The analysis counts hypothetical contact/control/sensor work for configured dimensions, not a full biomechanical simulation or empirical locomotion trajectory.

**AntBrain** abstracts sensory and sparse neural processing inspired by insect AL, MB, and CX organization. Supported counter patterns are `random` and `structured`, not validated species topologies. Population size and activity fraction are illustrative input choices, not inferred biological measurements.

**AntMind** represents policy evaluation and active-inference concepts [@friston2015epistemic]. The base counter model enumerates tree nodes; a distinct enhanced model bounds leaf policies. An expected-free-energy score is a decision-theoretic quantity, not automatically thermodynamic energy or a demonstrated biological mechanism.

Stigmergic coordination through proposed diffusion–decay fields is architectural context. The native CE manifest's module sweeps do not establish colony-size scaling, collective-task performance, or nonlinear energy coupling.

## Theoretical Foundations and Applicability

Shannon's information theory [@shannon1948mathematical] motivates explicit channel assumptions. For an ideal bandwidth-limited additive Gaussian-noise channel,
\begin{equation}
C=W\log_2(1+\mathrm{SNR})\quad\text{bits/s}.\label{eq:shannon_capacity}
\end{equation}
Here $W$ is bandwidth, not the planning branching factor $B$. Sensor-channel count alone supplies neither bandwidth, signal-to-noise ratio, nor achieved information rate. No percentage of channel capacity is inferred for AntBrain.

Landauer's principle concerns logically irreversible bit erasure under thermodynamic assumptions [@landauer1961irreversibility]:
\begin{equation}
E_{\mathrm{erase}}\geq k_BT\ln 2\quad\text{per erased bit}.\label{eq:landauer_limit}
\end{equation}
At approximately room temperature this reference is of order $10^{-21}$ J/bit. It is not an exact per-FLOP price, nor a complete workload minimum without a justified erasure count. Mechanical-work and information references require compatible boundaries before comparisons.

The heat-engine Carnot expression
\begin{equation}
\eta_{\mathrm{Carnot}}=1-T_c/T_h\label{eq:carnot_efficiency}
\end{equation}
applies between specified thermal reservoirs. It is not a generic electromechanical actuator-efficiency bound. Consequently this study does not claim a universal thermodynamic optimization factor from its assumed electrical prices.

## Levels of Analysis and Evidence Gaps

At the **problem level**, proposed outcomes include navigation, foraging, task allocation, and robustness. At the **algorithm level**, representations and policy/solver choices determine hypothetical work. At the **physical implementation level**, devices, sensing rails, actuation, latency, and environmental conditions determine measurable resource use. Keeping these levels separate prevents operation-count consistency from being mistaken for successful behavior or calibrated physical predictions.

The workflow supplies an explicit input/result interface and model-based comparisons. Independent device microbenchmarks, held-out energy traces, identified biological datasets, and task outcomes are still required for accuracy, real-time, species, or biological-superiority claims. No current source-based result establishes Sobol/Morris sensitivity, critical phenomena, or external replication.
