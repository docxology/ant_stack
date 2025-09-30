### Table: Module Complexities (Per 10ms Tick) {#tab:module_complexities}

| Module | Time Complexity | Space Complexity | FLOPs/Decision | Memory (bytes) | Notes |
|--------|-----------------|------------------|----------------|----------------|-------|
| Physics | $\mathcal{O}(J + C^{\alpha})$ | $\mathcal{O}(J + C)$ | 89543 | 10072 | $\alpha \approx 1.5$ (PGS solver) |
| Sensors | $\mathcal{O}(S)$ | $\mathcal{O}(S)$ | 1280 | 4096 | includes packing/timestamps |
| AL | $\mathcal{O}(K)$ | $\mathcal{O}(K)$ | 1920 | 1024 | sparse linear ops |
| MB | $\mathcal{O}(\rho N_{KC})$ | $\mathcal{O}(N_{KC})$ | 1000000 | 600000 | sparse coding \& local plasticity |
| CX | $\mathcal{O}(H)$ | $\mathcal{O}(H)$ | 2816 | 256 | ring update + soft WTA |
| Policies | $\mathcal{O}(B H_p)$ | $\mathcal{O}(B H_p)$ | 3888377057740 | 251971416832 | bounded rationality sampling |
| Pheromone | $\mathcal{O}(G + E)$ | $\mathcal{O}(G)$ | 1000 | 40000 | explicit diffusion scheme |