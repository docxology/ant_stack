### Table: Empirical Scaling Laws {#tab:scaling_laws}

| Module | Parameter | Energy Scaling | R² | FLOPs Scaling | R² | Regime |
|--------|-----------|----------------|----|--------------|----|--------|
| AntBody | Joint Count (J) | $E \propto J^{1.03e-02}$ | 0.927 | $\text{FLOPs} \propto J^{0.032}$ | 0.930 | sub-linear |
| AntBrain | AL Channels (K) | $E \propto K^{1.34e-03}$ | 0.871 | $\text{FLOPs} \propto K^{0.002}$ | 0.871 | sub-linear |
| AntMind | Policy Horizon (H_p) | $E \propto H_p^{3.8}$ | 0.940 | $\text{FLOPs} \propto H_p^{3.8}$ | 0.940 | super-linear |