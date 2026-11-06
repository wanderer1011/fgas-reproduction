# M3 — smoke run and backend benchmark

Runs: `20261106-140753_smoke_sweep`, `20261106-140817_bench` (runs/ is git-ignored; numbers copied here).

## Iris / Wine, full sigma grid (cpu, float64, tol 1e-3)

| Dataset | Paper Table 2 | Paper Fig. 4 peak | abs FGD (official code) | squared FGD (paper Eq. 7) |
|---|---|---|---|---|
| Iris | 1.000 | 1.000 @ 0.85 | **1.0000** @ 0.25 (also 1.0000 @ 0.30-0.40, 0.85) | 0.9982 @ 0.35 / 0.85 |
| Wine | 0.885 | 0.885 @ 0.80 | **0.8849** @ 0.80 | 0.8992 @ 0.80 |

The abs-FGD sigma curves reproduce the Fig. 4 curves for Iris and Wine point by point
(e.g. Wine 0.855 -> 0.836 @ 0.35 -> 0.883 @ 0.6 -> 0.885 @ 0.8 -> 0.757 @ 0.95;
Iris dip to 0.963 @ 0.6 and drop to 0.900 @ 0.95). The squared-FGD curves do not.
=> Strong evidence that the published numbers were produced with the official code's
absolute-difference FGD, not Eq. 7 as printed.

All runs converged (90-91 iterations); no degenerate sigma.

## Backend benchmark (20-sigma sweep, squared FGD)

| Setting | Sate (6435x36) | Musk (3062x166) | max abs dAUC vs cpu-float64 (2 datasets x 20 sigma) |
|---|---|---|---|
| cpu float64 | 320.1 s | 306.5 s | - |
| cuda float64 | 23.6 s | 11.3 s | 0.0 |
| cuda float32 | 7.8 s | 4.6 s | 0.0058 (fails 1e-4 criterion) |

Decision (PLAN rule): **cuda float64** for all experiments. float32 rejected.
Peak GPU memory pool, Sate: 824 MB (RTX 4060 Laptop, 8 GB).
