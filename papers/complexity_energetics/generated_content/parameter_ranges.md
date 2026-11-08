### Table: Enhanced Model Sweep Configuration {#tab:param_ranges}

| Module | Varied Parameter | Evaluated Values | Fixed Parameters |
|--------|------------------|------------------|------------------|
| AntBody | J | [6, 12, 18, 24, 36] | {'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'} |
| AntBrain | K | [64, 128, 256, 512, 1024] | {'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100} |
| AntMind | H_p | [5, 7, 9, 11, 13] | {'B': 3, 'hz': 100, 'state_dim': 12, 'action_dim': 4, 'max_policies': 1000} |

These are selected model inputs, not measured biological ranges.
