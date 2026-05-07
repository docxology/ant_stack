# External Ant Data Integration Guide

This guide describes how to integrate empirical ant trajectory/contact outputs
with Ant Stack reporting workflows.

## Why this matters

Ant Stack models biological and computational structure of collective behavior.
Empirical movement data (tracked ants in video) provides a way to:

- compare simulated behavior to observed behavior;
- validate contact-driven interaction hypotheses; and
- include reproducible benchmark snapshots in reports and papers.

## Expected input schema

Use CSV files with the following minimal columns.

### Trajectories (`ant_trajectories.csv`)

- `track_id`
- `frame`
- `x`
- `y`

Recommended kinematics:

- `vx_px_s`
- `vy_px_s`
- `speed_px_s`

### Contact events (`collisions.csv`)

- `collision_type` (`ant_ant` or `ant_object`)
- `start_time_s`
- `end_time_s`
- `duration_s`

### Contact samples (`collision_samples.csv`, optional)

Frame-level candidates used to aggregate contact events.

## Practical integration pattern

1. Produce empirical outputs from an external tracking pipeline.
2. Store benchmark artifacts in a reproducible results directory.
3. Attach summary plots as static assets in docs or paper sections.
4. Track provenance of source videos/datasets alongside results.

A common benchmark snapshot includes:

- mean speed by split or condition;
- detections by scene/domain;
- ant-ant versus ant-object contact counts; and
- similarity metrics between real and simulated trajectories.

## Reporting template

Use a compact dashboard image for fast status communication:

```markdown
### Empirical Benchmark Snapshot

![Empirical ant benchmark dashboard](assets/analysis_dashboard.png)
```

Pair this with a short metrics table (CSV-derived) in the same section.

## Reproducibility checklist

- Keep raw source URLs and DOI references with every processed dataset.
- Record exact script versions/commit SHA for generated metrics.
- Separate event-level and sample-level contact outputs.
- Preserve unit semantics (pixels, frames, seconds) in column names.

