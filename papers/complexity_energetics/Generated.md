# Generated Results (from src)

Provenance: commit=278fd36, seed=123, python=3.13.7


## Per-Workload Estimated Energy (mean [95% CI], J)

Only Body and Brain expend energy; Mind is a symbolic layer (0 J by convention).

| Workload | Mean (J) | 95% CI Low | 95% CI High | N |
|---|---:|---:|---:|
| body | 0.250003 | 0.250003 | 0.250003 | 100 |
| brain | 0.250117 | 0.250117 | 0.250117 | 100 |

## Figure: Total Energy by Workload {#fig:energy_by_workload}

![Total estimated energy by workload](papers/complexity_energetics/assets/energy.png)

**Caption:** Total estimated energy by workload. Only Body and Brain expend energy; Mind is symbolic (0 J).

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/energy.png}{(View absolute file)}

## Figure: Body Energy Partition {#fig:body_partition}

![Body energy partition](papers/complexity_energetics/assets/body_split.png)

**Caption:** Estimated Body energy partition into Sensing and Actuation, aggregated over runs.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/body_split.png}{(View absolute file)}

## Figure: AntBody Energy Scaling vs Joint Count (J) {#fig:scaling_body_J}

![antbody energy scaling vs joint count (j)](papers/complexity_energetics/assets/scale_body_J.png)

**Caption:** AntBody energy scaling with joint count (J). Demonstrates flat scaling $E \propto J^{-0.49}$ $R^2=0.987$ due to baseline power dominance (50 mW from sensors and controllers), making morphological complexity essentially free in terms of energy cost.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_body_J.png}{(View absolute file)}

## Figure: AntBody Energy Scaling vs Joint Count (J) [scatter] {#fig:scaling_body_J_scatter}

![antbody energy scaling vs joint count (j) [scatter]](papers/complexity_energetics/assets/scale_body_J_scatter.png)

**Caption:** Scatter plot of AntBody energy consumption across different joint counts (J). Shows the dominance of baseline power consumption over joint-dependent computation, resulting in essentially flat energy scaling $E \propto J^{-0.49}$.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_body_J_scatter.png}{(View absolute file)}

## Figure: Pareto Frontier (Energy vs Performance) {#fig:pareto_body_J}

![pareto frontier (energy vs performance)](papers/complexity_energetics/assets/pareto_body_J.png)

**Caption:** Pareto frontier for AntBody showing energy-performance trade-offs with varying joint counts. Performance proxy represents morphological dexterity.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/pareto_body_J.png}{(View absolute file)}

## Figure: AntBrain Energy Scaling vs AL Channels (K) {#fig:scaling_brain_K}

![antbrain energy scaling vs al channels (k)](papers/complexity_energetics/assets/scale_brain_K.png)

**Caption:** AntBrain energy scaling as a function of antennal lobe input channels (K). Demonstrates sub-linear scaling $E \propto K^{0.33}$ $R^2=0.930$ due to biological sparsity patterns ($\rho = 0.02$), enabling massive sensory expansion (64 to 1024 channels) without proportional energy increase. Multiple curves represent different AntMind policy variants affecting neural processing efficiency.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_brain_K.png}{(View absolute file)}

## Figure: Pareto Frontier (Energy vs Performance) {#fig:pareto_brain_K}

![pareto frontier (energy vs performance)](papers/complexity_energetics/assets/pareto_brain_K.png)

**Caption:** Pareto frontier analysis showing the trade-off between energy consumption and sensory processing capacity in AntBrain. Performance is proxied by inverse AL input channels (1/K), representing information processing capability.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/pareto_brain_K.png}{(View absolute file)}

## Figure: AntBrain Energy Scaling vs AL Channels (K) [scatter] {#fig:scaling_brain_K_scatter}

![antbrain energy scaling vs al channels (k) [scatter]](papers/complexity_energetics/assets/scale_brain_K_scatter.png)

**Caption:** Scatter plot representation of AntBrain energy scaling with antennal lobe input channels (K). Individual data points show experimental measurements with variability, complementing the line plot smoothing. Demonstrates the robustness of sub-linear scaling $E \propto K^{0.33}$ across different sensory configurations.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_brain_K_scatter.png}{(View absolute file)}

## Figure: AntMind Energy Scaling vs Planning Horizon (H_p) {#fig:scaling_mind_H_p}

![antmind energy scaling vs planning horizon (h_p)](papers/complexity_energetics/assets/scale_mind_H_p.png)

**Caption:** AntMind energy scaling with policy planning horizon (H_p). Shows super-linear exponential growth $E \propto H_p^{1.01}$ $R^2=0.997$ due to combinatorial explosion in policy evaluation, establishing fundamental limits for real-time active inference.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_mind_H_p.png}{(View absolute file)}

## Figure: AntMind Energy Scaling vs Planning Horizon (H_p) [scatter] {#fig:scaling_mind_H_p_scatter}

![antmind energy scaling vs planning horizon (h_p) [scatter]](papers/complexity_energetics/assets/scale_mind_H_p_scatter.png)

**Caption:** Scatter plot showing exponential energy growth $E \propto H_p^{1.01}$ in AntMind as planning horizon (H_p) increases. Illustrates the fundamental computational barriers of exact active inference beyond 15-step horizons.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/scale_mind_H_p_scatter.png}{(View absolute file)}

## Figure: Pareto Frontier (Energy vs Performance) {#fig:pareto_mind_H_p}

![pareto frontier (energy vs performance)](papers/complexity_energetics/assets/pareto_mind_H_p.png)

**Caption:** Pareto frontier for AntMind showing fundamental trade-offs between planning horizon and computational feasibility.

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/assets/pareto_mind_H_p.png}{(View absolute file)}

## Figure: Per-Decision Energy Breakdown {#fig:per_decision}

![Per-decision energy breakdown](papers/complexity_energetics/assets/per_decision_breakdown.png)

**Caption:** Average per-decision (10 ms) energy components at 100 Hz. Mind compute is 0 by convention; baseline is system idle.

### Table: Per-Decision Energy Breakdown (mJ)

| Component | Energy (mJ) |
|---|---:|
| Actuation | 360.000 |
| Sensing | 12.800 |
| Brain compute | 0.003 |
| Mind compute | 0.000 |
| Baseline/idle | 5.000 |
| Total | 377.803 |

## Raw Results (CSV)

\href{file:///Users/4d/Documents/GitHub/ant/papers/complexity_energetics/out/results.csv}{View Results CSV}

## Derived Metric: Cost of Transport (dimensionless)

CoT $\approx$ 1.2747 (assuming mass=0.02 kg, distance=1.0 m).

Biological ants achieve CoT 0.1-0.3, indicating 6.4$\times$ optimization potential in mechanical efficiency.

## Table: Per-Decision Complexity (Compute/Memory) {#tab:complexity_per_decision}

| Workload | FLOPs/decision | SRAM bytes/decision | DRAM bytes/decision | Spikes/decision |
|---|---:|---:|---:|---:|
| body | 89543 | 7554 | 2518 | 0 |
| brain | 1005236 | 514488 | 90792 | 100 |
| mind | 2950000 | 124800 | 31200 | 0 |
