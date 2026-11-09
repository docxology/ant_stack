# Analytical Complexity and Energy Results

These are analytical/synthetic workloads, not independent biological or hardware observations.

## Per-Workload Estimated Energy

Each row includes that module's compute and isolated idle over its workload duration. These isolated rows must not be summed as a system total; locomotion is excluded.

| Workload | Mean (J/window) | 95% CI Low | 95% CI High | N |
|---|---:|---:|---:|---:|
| body | 0.25000291 | 0.25000291 | 0.25000291 | 100 |
| brain | 0.25004641 | 0.25004641 | 0.25004641 | 100 |
| mind | 271.14218 | 271.14218 | 271.14218 | 100 |

## Figure: Module-Isolated Energy {#fig:energy_by_workload}

![Mean module-isolated compute plus idle, with 95% bootstrap intervals for the mean. Repeating deterministic values does not establish empirical precision; positive wide-range values use a labelled logarithmic axis.](../papers/complexity_energetics/out/assets/energy.png)


## Figure: Body configured compute curves {#fig:scaling_body_J}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_body_J.png)


## Figure: Body lowest configured compute points {#fig:scaling_body_J_scatter}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_body_J_scatter.png)


## Figure: Body inverse-parameter illustration {#fig:inverse_parameter_body_J}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. The inverse-parameter curve is a proxy, not measured performance or a nondominated frontier.](../papers/complexity_energetics/out/assets/inverse_parameter_body_J.png)


## Figure: Brain configured compute curves {#fig:scaling_brain_K}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_brain_K.png)


## Figure: Brain lowest configured compute points {#fig:scaling_brain_K_scatter}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_brain_K_scatter.png)


## Figure: Brain inverse-parameter illustration {#fig:inverse_parameter_brain_K}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. The inverse-parameter curve is a proxy, not measured performance or a nondominated frontier.](../papers/complexity_energetics/out/assets/inverse_parameter_brain_K.png)


## Figure: Mind configured compute curves {#fig:scaling_mind_H_p}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_mind_H_p.png)


## Figure: Mind lowest configured compute points {#fig:scaling_mind_H_p_scatter}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. No fixed biological scaling law is inferred.](../papers/complexity_energetics/out/assets/scale_mind_H_p_scatter.png)


## Figure: Mind inverse-parameter illustration {#fig:inverse_parameter_mind_H_p}

![Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. The inverse-parameter curve is a proxy, not measured performance or a nondominated frontier.](../papers/complexity_energetics/out/assets/inverse_parameter_mind_H_p.png)


## Figure: Body Energy Partition {#fig:body_partition}

![Separate sensing and actuation assumptions integrated over the common 10 ms interval, not the compute-only workload row.](../papers/complexity_energetics/out/assets/body_split.png)


## Figure: Common-Window Compute Accounting {#fig:per_decision}

![Selected workload counters over a common 10 ms window, inverse maximum configured update rate, with shared idle counted once. Integer update flooring can omit slower modules; this is not their time-averaged load or necessarily one decision each. Sensing and actuation are separate assumptions, not observed locomotion.](../papers/complexity_energetics/out/assets/per_decision_breakdown.png)

| Component | Energy (mJ / 10 ms) |
|---|---:|
| body | 5.817664e-05 |
| brain | 0.0009281049 |
| mind | 5417.8436 |
| Body actuation | 360 |
| Body sensing | 12.8 |
| Shared idle | 5 |
| Modeled system total | 5795.6445 |

## Table: Common-Window Complexity {#tab:complexity_per_decision}

| Workload | FLOPs / 10 ms | SRAM bytes / 10 ms | DRAM bytes / 10 ms | Spikes / 10 ms |
|---|---:|---:|---:|---:|
| body | 26320 | 6246.4 | 1561.6 | 0 |
| brain | 23619.778 | 389491.2 | 43276.8 | 100 |
| mind | 3.8883771e+12 | 1.7637999e+11 | 7.5591425e+10 | 0 |

## Cost of Transport

Unavailable: compute-window energy is not locomotion energy, and no matching modeled locomotion interval/distance is supplied by this producer.

## Raw Results

[Results CSV](../papers/complexity_energetics/out/results.csv)
