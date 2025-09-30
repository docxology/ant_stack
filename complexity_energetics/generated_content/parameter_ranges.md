### Table: Parameter Ranges and Default Values {#tab:param_ranges}

| Symbol | Meaning | Typical Range | Default | Physical Basis |
|--------|---------|---------------|---------|----------------|
| J | Joint DOF | 6–36 | 18 | Hexapod: 3 DOF × 6 legs |
| C | Active contacts | 4–30 | 12 | 4 legs × 3 contact points |
| S | Sensor channels | 64–1024 | 256 | IMU + vision + chemosensors |
| K | AL input channels | 32–1024 | 128 | Olfactory glomeruli |
| N_KC | Kenyon cells | 10⁴–10⁵ | 50,000 | Insect mushroom body |
| ρ | Active sparsity | 0.005–0.1 | 0.02 | Biological cortical activity |
| H | Heading bins | 16–256 | 64 | Compass resolution |
| H_p | Policy horizon | 5–20 | 15 | Bounded rationality limit |
| B | Branching factor | 2–8 | 4 | Action space complexity |