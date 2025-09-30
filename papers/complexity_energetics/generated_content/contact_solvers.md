### Table: Contact Solver Complexity Analysis {#tab:contact_solvers}

| Solver | Theoretical Complexity | Memory Scaling | Typical Range | Best Use Case |
|--------|------------------------|----------------|---------------|---------------|
| PGS | $\mathcal{O}(C^{1.5})$ | $\mathcal{O}(C)$ | C ≤ 20 | Real-time applications |
| LCP | $\mathcal{O}(C^3)$ | $\mathcal{O}(C^2)$ | C < 10 | High-accuracy simulation |
| MLCP | $\mathcal{O}(C^{2.5})$ | $\mathcal{O}(C^{1.5})$ | 10 ≤ C ≤ 30 | Balanced performance |