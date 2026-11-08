### Table: Supported Contact Solver Model Counters {#tab:contact_solvers}

| Solver | Joints | Contacts | FLOPs/Solve | Memory (bytes) | Iterations |
|--------|--------|----------|-------------|----------------|------------|
| PGS | 18 | 5 | 6000 | 2832 | 10 |
| PGS | 18 | 10 | 12000 | 3072 | 10 |
| PGS | 18 | 15 | 27000 | 3312 | 15 |
| PGS | 18 | 20 | 48000 | 3552 | 20 |
| PGS | 18 | 25 | 75000 | 3792 | 25 |
| SI | 18 | 5 | 3744 | 3072 | 1 |
| SI | 18 | 10 | 5544 | 3552 | 1 |
| SI | 18 | 15 | 7344 | 4032 | 1 |
| SI | 18 | 20 | 9144 | 4512 | 1 |
| SI | 18 | 25 | 10944 | 4992 | 1 |
| LDLT | 18 | 5 | 11979 | 8712 | 1 |
| LDLT | 18 | 10 | 36864 | 18432 | 1 |
| LDLT | 18 | 15 | 83349 | 31752 | 1 |
| LDLT | 18 | 20 | 158184 | 48672 | 1 |
| LDLT | 18 | 25 | 268119 | 69192 | 1 |

Analytical workload estimates, not measured execution times or solver accuracy benchmarks.
