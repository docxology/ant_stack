# Appendices

## A. Reproducibility Checklist

- Fixed random seeds (report all)
- Explicit I/O rates and units (see `AntBody`)
- Engine versions: physics, spiking/NN libraries, OS
- Environment configs: terrain, pheromone $D$, $\lambda$, deposit rules
- Agent configs: neuron counts, learning rules, energy budgets
- Evaluation protocols and metrics; trials; confidence intervals
- Data/code availability; exact commit SHA
- Registered experiment manifests (YAML/JSON) to enable one-click reruns
- Model cards: describe objectives, assumptions, limits, and validation datasets
- Unit tests: per-module interface and invariants; scenario tests for end-to-end loops
 - Unit registry and versions (e.g., Pint) recorded; message schema versions (e.g., ROS 2) pinned
 - Continuous integration manifest for experiment reruns and document/PDF builds

## B. Species Parameterization Quickstart

- `AntBody`: leg DOF, segment masses, antenna channel count K, sensor noise $\sigma$, gait presets
- Pheromone: diffusion $D$, decay $\lambda$, baseline deposit, reward/urgency modulation
 - `AntBrain`: AL glomeruli count $\leftrightarrow$ K, MB sparsity (Kenyon cell ratio), CX ring size
- `AntMind`: preference priors, policy horizon, update rates; colony size, sharing frequency
- Colony ecology: resource density, nest locations, predator risk profiles (for security experiments)
- Provenance: cite sources for parameter choices and note uncertainties
 - Energy/compute envelope; actuator latency distributions; friction coefficients per terrain/material

## C. Evaluation Protocol Templates

- Navigation: maze/slope/rough terrain; success, path ratio, energy per reward
- Trail following: deposit reward-linked trail; formation time, stability, traffic
- Task allocation: mixed stochastic arrivals; utilization, latency, resilience
- Adversarial robustness: deceptive pheromones; deviation from norms, recovery time
- Security drills: spoofed gradients, sensor jamming; detection latency and false-positive rate
- Reporting: include seed ranges, CIs, ablations, and failure cases with traces
 - Specify communication-overhead budgets and failure-injection schedules
