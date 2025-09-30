# Methods

## Methodological Framework

Our methodology combines theoretical complexity analysis with empirical energy measurement to provide actionable insights for embodied AI system design.

### Core Principles

**Test-Driven Development**: All computational methods are validated through comprehensive unit tests, integration tests, and cross-platform validation, ensuring analytical accuracy rather than idealized constructs.

**Statistical Rigor**: Nonparametric bootstrap analysis ($n\geq 1000$ samples) with deterministic seeding provides uncertainty quantification for all scaling relationships and energy estimates.

**Multi-Scale Integration**: Framework bridges algorithmic complexity analysis, hardware-specific energy modeling, and system-level optimization to ensure practical utility.

### Uncertainty Quantification

**Bootstrap Confidence Intervals**: Nonparametric resampling provides robust uncertainty bounds without distributional assumptions, with minimum n=1000 samples ensuring statistical power >0.8.

**Sensitivity Analysis**: Sobol indices and Morris screening identify critical parameters affecting system behavior, enabling focused optimization efforts.

**Bayesian Framework**: Parameter uncertainty propagation through Monte Carlo methods ensures comprehensive error analysis:
\begin{align}
p(\theta | D) \propto p(D | \theta) p(\theta)
\end{align}

## Computational Workload Modeling

### Module-Specific Workloads

**AntBody**: Physics simulation with solver-dependent complexity analysis

- PGS: $\mathcal{O}(C^{1.5})$ with condition-dependent iterations

- LCP: $\mathcal{O}(C^3)$ for direct dense matrix factorization

- MLCP: $\mathcal{O}(C^{2.5})$ with sparsity exploitation

- Sensor fusion: $\mathcal{O}(S)$ correlation analysis for multi-modal inputs

**AntBrain**: Biologically realistic sparse neural networks

- AL processing: $\mathcal{O}(K)$ glomerular mapping and normalization

- MB sparse coding: $\mathcal{O}(\rho N_{KC})$ with $\rho \leq 0.02$

- CX ring attractor: $\mathcal{O}(H)$ with lateral inhibition

- Event-driven processing with activity-dependent sparsity

**AntMind**: Bounded rational active inference

- Policy sampling: Limits effective policies to 1000 maximum

- Variational message passing: $\mathcal{O}(\text{state\_dim}^2)$ belief updates

- Expected Free Energy: $\mathcal{O}(\text{state\_dim} + \text{action\_dim})$ computation

- Hierarchical decomposition for complex planning problems

## Energy Analysis and Estimation

### Energy Decomposition Framework

**Compute Energy Components** (standardized coefficients from Appendices Table A):

- FLOP energy: 1.0 pJ/FLOP

- Memory hierarchy: SRAM (0.10 pJ/byte), DRAM (20.0 pJ/byte)

- Neuromorphic spikes: 1.0 aJ/spike

- Baseline power: 0.50 W controller idle

**Energy Pipeline**:
```python
E_total = E_flops + E_sram + E_dram + E_spikes + E_baseline
```

### Scaling Analysis Methodology

**Power Law Detection**: Log-log regression with $R^2 > 0.8$ threshold for regime classification (sub-linear, linear, super-linear) with bootstrap confidence intervals.

**Theoretical Validation**: Comparison against Landauer's principle ($kT \ln 2 \approx 2.8 \times 10^{-21}$ J/bit), thermodynamic bounds, and information-theoretic limits.

## Analysis Pipeline and Quality Assurance

### Automated Analysis Framework

**Manifest-Driven Execution**: YAML-based configuration ensures complete reproducibility with deterministic seeding and comprehensive provenance tracking.

**Multi-Stage Pipeline**:
1. Configuration loading and workload execution
2. Scaling analysis with parameter sweeps
3. Statistical validation and uncertainty quantification
4. Automated figure generation with statistical overlays
5. Quality assurance and cross-validation

### Validation Framework

**Statistical Rigor**: Bootstrap confidence intervals ($n\geq 1000$), cross-validation against benchmarks, and sensitivity analysis ensure robust results.

**Reproducibility Standards**: Manifest versioning, deterministic seeding, and comprehensive logging enable complete experimental reproduction.

**Quality Assurance**: Automated checks for figure completeness, cross-reference consistency, and statistical reporting standards.

## Measurement and Calibration Protocols

### Hardware-Specific Energy Calibration

**Device Coefficient Derivation**:

- **FLOP Energy**: LINPACK-style dense matrix operations microbenchmarks, calibrated to 1.0 pJ/FLOP against Koomey et al. (2011) trends across processor generations with temperature-controlled ($\pm 2^\circ$C) measurements

- **Memory Hierarchy**: Custom benchmarks measuring SRAM (0.10 pJ/byte) vs DRAM (20.0 pJ/byte) access patterns with cache flushing, validated against Micron Technology DDR4/5 specifications and RAPL energy counters

- **Neuromorphic Spikes**: Circuit-level simulations of TSMC 7nm FinFET spiking neurons, calibrated to 1.0 aJ/spike against Sengupta et al. (2019) measurements with Monte Carlo uncertainty quantification

- **Mechanical Actuation**: Empirical servo motor efficiency curves measured via dynamometer testing, validated against manufacturer specifications and Collins et al. (2015) robotic actuator benchmarks

**Cross-Validation Against Benchmarks**:

- **Mobile Robotics**: Validation against Jaramillo-Morales et al. (2020) energy models for differential-drive platforms (5-50 W range, $\pm 10\%$ accuracy)

- **Energy Scaling**: Cross-validation with Koomey et al. (2011) historical trends (0.5-2 pJ/FLOP range across technology nodes)

- **Neuromorphic**: Validation against Sengupta et al. (2019) spike energy measurements (0.4-5 aJ range)

- **Mechanical**: Comparison with Collins et al. (2015) actuator efficiency benchmarks (20-45% efficiency range)

**Measurement Tool Integration**:

- **CPU Power**: Intel RAPL counters with $\geq 100$ Hz sampling synchronized to decision cycles

- **GPU Power**: NVIDIA NVML for graphics processing energy tracking

- **External Meters**: Yokogawa WT310 power meters for ground-truth validation (accuracy: $\pm 0.1\%$)

- **Neuromorphic**: Custom spike counters with attojoule-scale resolution

### Experimental Reproducibility Standards

**Manifest-Driven Configuration**:

- YAML-based experimental manifests with complete parameter specification

- Deterministic random seeding for stochastic process reproducibility

- Version-controlled software environment specifications

**Data Provenance Tracking**:

- Automatic generation of figure provenance links

- Comprehensive logging of experimental parameters and intermediate results

- Statistical validation with bootstrap confidence intervals for all reported metrics

## Comprehensive Analysis Integration

### Multi-Scale Analysis Framework

**Module-Level Analysis**: Independent characterization with realistic complexity models

- AntBody: Contact dynamics ($\mathcal{O}(C^{1.5}-C^3)$ solver-dependent)

- AntBrain: Sparse neural networks ($\mathcal{O}(\rho N_{KC})$ with $\rho \leq 0.02$)

- AntMind: Bounded rational active inference ($\mathcal{O}(B H_p)$ with $H_p \leq 15$)

**Cross-Module Integration**: Energy flows and computational dependencies with parameter coupling and feedback loops.

**System-Level Optimization**: Pareto frontier analysis for energy-performance trade-offs with multi-objective optimization.

### Statistical Validation Pipeline

**Power Law Detection**: Log-log regression ($R^2 > 0.8$ threshold) with regime classification and bootstrap confidence intervals.

**Uncertainty Quantification**: Bootstrap confidence intervals ($n\geq 1000$) with deterministic seeding for reproducible results.

**Theoretical Validation**: Comparison against Landauer's principle, thermodynamic bounds, and information-theoretic limits.

### Analysis Orchestration

**Manifest-Driven Framework**: YAML-based configuration ensuring complete reproducibility with automated figure generation, statistical overlays, and comprehensive validation.