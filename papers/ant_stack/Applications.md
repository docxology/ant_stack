# Proposed Applications and Evaluation Protocols {#applications}

The following are proposed applications and evaluation protocols, not demonstrated deployments, calibration standards, or evidence of safety/alignment. Ant-inspired efficiency and robustness remain task-specific hypotheses to evaluate.

Design intent: a single, compact agent architecture that ports from high-fidelity simulation to embedded platforms and multi-agent swarms with minimal change.

## Proposed Cross-Domain Evaluation Metrics {#cross-cutting-evaluation-metrics}

- **Efficiency**: energy per reward; compute per action
- **Robustness**: success under sensor/actuator noise and partial failures
- **Scalability**: performance vs number of agents; communication overhead
- **Alignment**: deviation from intended norms under perturbations/adversaries
- **Reproducibility**: standardized configs, seeds, and dataset/version pinning

## Swarm Robotics

Proposed ant-inspired control for physical-platform evaluation; deployment readiness has not been established.

- **Disaster response:** Navigate debris, locate survivors, perform cleanup in hazardous settings
- **Logistics and construction:** Stigmergic coordination for warehousing and build tasks
- **Environmental monitoring:** Distributed mapping/sampling with trail-guided coverage
- **Fieldable constraints:** low compute/energy budget, lossy comms, intermittent GPS/vision

### Evaluation (swarm robotics)

- Time-to-food, path efficiency, energy per reward, success under sensor noise
- Robustness to actuator loss and terrain changes; recovery time
- Communication budget (bandwidth/duty cycle); MTTF/MTTR under staged failures

## Networks and Optimization

- **Routing/load balancing:** ACO grounded in realistic diffusion/decay for routing, caching, congestion
- **Distributed scheduling:** Colony-inspired task allocation under constraints; throughput, fairness, resilience
- **Resilient overlays:** Decoy trails and evaporation parameters for adaptive rerouting under attack
- **Cache placement:** Use pheromone analogs for content popularity and decay-based invalidation

## Cognitive Security

Secure systems at the cognitive layer: resilience to deception, spoofing, and manipulation.

- **Secure swarms:** Test whether decentralized control improves resilience under a declared compromise model; decentralization alone does not guarantee security
- **Defending complex systems:** Colony-style distributed detection/response for critical infrastructure
- **Cognitive honeypots:** Deploy deceptive pheromone fields to study adversary behavior and improve defenses
- **Incident triage:** Prioritize actions via expected free energy; quantify trade-offs under uncertainty

### Evaluation (cognitive security)

- Adversary success probability vs defense; detection AUC on deceptive signals; recovery time
- False-positive/negative balance; cost-of-delay under competing alerts

## Biosurveillance & Biodefense

- **Cognitive anomaly detection:** Learn baselines from sensors; flag deviations
- **Disease spread modeling:** Reuse pheromone diffusion/decay for pathogen propagation
- **Sentinel networks:** Agent subsets specialized for detection/triage to reduce false alarms
- **Adaptive sampling:** Allocate sensing effort via stigmergy when anomalies persist

### Evaluation (biosurveillance)

- Anomaly PR-AUC on simulated outbreaks; detection latency vs false alarms
- Spatial coverage vs resource budget; escalation pathways

## Foundational AI Research

- **Emergent intelligence:** Study complex goal-directed behavior from simple local rules
- **Benchmark proposal:** Evaluate energy-efficient AI designs after device calibration and task-specific validation
- **Embodiment ablations:** Vary body/brain/mind couplings to test contributions to generalization
- **Neural efficiency:** Map performance--energy trade-offs; quantify sparse spiking benefits

### Evaluation (foundational AI)

- Navigation success on mazes/rough terrain; memory capacity via odor--reward associations
- Policy entropy, sample efficiency, and transfer across terrains/species presets

## AI Alignment and Safety

- **Alignment hypothesis:** Test when local incentives support or conflict with collective welfare; natural organization does not guarantee alignment
- **Emergent goals:** Test designs that avoid undesirable emergent behaviors without central control
- **Norm compliance:** Measure deviation under resource stress and adversarial signals
- **Value handshakes:** Encode minimal priors that align individual incentives with group welfare

### Evaluation (alignment)

- Colony social welfare under scarcity; deviation from norms under adversarial pheromones
- Collateral risk vs objective attainment; robustness to deceptive gradients

### Proposed Testbeds and Requirements for Fair Comparisons

- The application domains are proposed testbeds, not demonstrated implementations or performance results.
- Comparisons require implemented baselines, matched measurement boundaries and declared evaluation protocols.

### Terminology Note

- We follow community guidance on language and update in consultation with myrmecologists
- Domain experts are invited to propose task variants and metrics; small PRs welcome
