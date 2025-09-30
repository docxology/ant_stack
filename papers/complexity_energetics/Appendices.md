# Appendices

## A. Energy Coefficients

**Table A: Device-Specific Energy Coefficients**

All energy calculations throughout this analysis use these standardized coefficients, calibrated for modern computing platforms and validated against published benchmarks.

| Parameter | Symbol | Value | Units | Reference/Validation |
|---|---|---|---|---|
| **Computational Energy** |
| FLOP energy | $e_\text{FLOP}$ | 1.0 | pJ/FLOP | Modern processors, 7-45nm nodes |
| SRAM access | $e_\text{SRAM}$ | 0.10 | pJ/byte | On-die cache, validated against |
| DRAM access | $e_\text{DRAM}$ | 20.0 | pJ/byte | External memory, validated against |
| **Neuromorphic Energy** |
| Spike generation | $E_\text{spk}$ | 1.0 | aJ/spike | Advanced 7nm circuits |
| **System Power** |
| Baseline power | $P_\text{idle}$ | 0.50 | W | Controller + housekeeping (non-zero) |
| Sensor power | $P_\text{sens}$ | 5.0 | mW/channel | Multi-modal sensing average |
| **Physical Constants** |
| Landauer limit | $kT \ln 2$ | 2.8 $\times$ 10^{-21} | J/bit | Thermodynamic minimum (295K) |
| Gravitational acceleration | $g$ | 9.81 | m/s² | Standard gravity for CoT calculations |

**Usage Note**: These coefficients are applied consistently across all energy calculations, scaling analyses, and efficiency comparisons. Any deviation from these values is explicitly noted.

## B. Detailed Measurement Protocols

**Power Meter Calibration and Environmental Controls**:
- Calibrate all power measurement instruments against NIST-traceable standards before each experimental session
- Record and maintain stable ambient conditions: temperature (20$\pm$2$^\circ$C), humidity (45$\pm$5%), and minimize electromagnetic interference
- Pin all software versions including operating system, drivers, libraries, and analysis frameworks
- Document hardware specifications including CPU model, memory configuration, and thermal management settings

**Experimental Reproducibility Standards**:
- Report deterministic seeds for all pseudorandom number generation across statistical bootstrap sampling
- Version and archive message schema definitions and unit registry entries for long-term reproducibility
- Maintain synchronized timestamps across all measurement systems with sub-millisecond accuracy
- Implement automated validation checks for measurement consistency and outlier detection

## C. Comprehensive Parameter Tables and System Configurations

**Module Scaling Parameters by Platform Type**:
- **Hexapod Configuration**: J=18 (6 legs $\times$ 3 joints), C=12 (typical stance contacts), S=256 (multi-modal sensing)
- **Quadruped Configuration**: J=12 (4 legs $\times$ 3 joints), C=8 (reduced contact complexity), S=128 (streamlined sensing)
- **Biped Configuration**: J=12 (2 legs $\times$ 6 joints), C=4 (minimal contacts), S=192 (enhanced proprioception)

**Neural System Parameter Ranges**:
- AL input channels (K): 64-512 depending on sensory modality emphasis and computational budget
- Mushroom Body populations (N_KC): 10^4-10^5 following biological scaling relationships
- Sparsity levels ($\rho$): 0.01-0.05 with 0.02 representing optimal biological balance
- Central Complex resolution (H): 32-128 bins balancing angular precision and computational cost

## D. Comprehensive Reproducibility Checklist

**Computational Environment Documentation**:
- Complete profiling harness configuration including compiler flags, optimization levels, and runtime parameters
- Detailed device specifications: CPU architecture, memory hierarchy, interconnect topology, and thermal design
- Systematic seed range validation across statistical analysis pipelines to ensure robust confidence intervals
- Continuous integration manifest versioning for automated reproducibility verification

**Data Provenance and Validation**:
- Automated generation of figure provenance links connecting results to specific experimental runs
- Comprehensive logging of all experimental parameters and intermediate computational results  
- Statistical validation frameworks including bootstrap confidence intervals and power law detection
- Cross-validation against theoretical limits including Landauer's principle and thermodynamic bounds

## E. Notation and Symbols (Unified)

| Symbol | Description |
|---|---|
| J | Total joint DOF (Body) |
| C | Active contacts per tick (Body) |
| S | Sensor channels (Body) |
| K | AL input channels (Brain) |
| N_KC | Number of Kenyon cells (MB) |
| $\rho$ | Active fraction in MB |
| H | CX heading bins |
| H_p | Policy horizon (steps) |
| B | Branching factor |
| G | Pheromone grid cells |
| E | Deposit events per tick |
| $e_\text{FLOP}$ | Energy per FLOP (pJ/FLOP) |
| $e_\text{SRAM}$ | Energy per SRAM byte (pJ/byte) |
| $e_\text{DRAM}$ | Energy per DRAM byte (pJ/byte) |
| $E_\text{spk}$ | Energy per spike (aJ/spike) |
| $P_\text{idle}$ | Idle power (W) |
| $P_\text{sens}$ | Sensor power (W) |
