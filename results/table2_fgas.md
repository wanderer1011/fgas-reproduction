# Table 2 — FGAS column: reproduction vs paper

Run: `20261106-145311_table2` (σ ∈ {0, 0.05, …, 1.0}, best AUC over the grid as in the paper; d = 0.1, tol = 1e-3, cuda float64).
Status: |Δ| ≤ 0.005 match, ≤ 0.02 close, else GAP. Paper values are rounded to 3 dp, so ±0.0005 is rounding.

| Dataset | Paper | **Ours abs (headline)** | best σ | Δ | % dev | status | Ours squared (Eq. 7) | best σ | Δ | status |
|---|---|---|---|---|---|---|---|---|---|---|
| Card | 0.953 | **0.9527** | 0.30 | -0.0003 | -0.03% | match | 0.9501 | 0.25 | -0.0029 | match |
| Diab | 0.989 | **0.9895** | 0.85 | +0.0005 | +0.05% | match | 0.9861 | 0.90 | -0.0029 | match |
| Ecoli | 0.955 | **0.9555** | 1.00 | +0.0005 | +0.05% | match | 0.9575 | 1.00 | +0.0025 | match |
| Ionos | 1.000 | **0.9996** | 0.85 | -0.0004 | -0.04% | match | 1.0000 | 0.75 | +0.0000 | match |
| Iris | 1.000 | **1.0000** | 0.25 | +0.0000 | +0.00% | match | 0.9982 | 0.85 | -0.0018 | match |
| Musk | 1.000 | **1.0000** | 0.40 | +0.0000 | +0.00% | match | 1.0000 | 0.00 | +0.0000 | match |
| Page | 0.960 | **0.9597** | 0.25 | -0.0003 | -0.03% | match | 0.9586 | 0.05 | -0.0014 | match |
| Pima | 0.971 | **0.9705** | 0.80 | -0.0005 | -0.05% | match | 0.9638 | 0.85 | -0.0072 | close |
| Sate | 0.734 | **0.7339** | 0.80 | -0.0001 | -0.01% | match | 0.7852 | 0.80 | +0.0512 | GAP |
| Sonar | 0.995 | **0.9948** | 0.85 | -0.0002 | -0.02% | match | 0.9928 | 0.90 | -0.0022 | match |
| Thyro | 0.985 | **0.9851** | 0.95 | +0.0001 | +0.01% | match | 0.9822 | 0.95 | -0.0028 | match |
| Wbc | 0.997 | **0.9971** | 0.00 | +0.0001 | +0.01% | match | 0.9963 | 0.00 | -0.0007 | match |
| Wdbc | 0.998 | **0.9977** | 0.75 | -0.0003 | -0.03% | match | 0.9970 | 0.80 | -0.0010 | match |
| Wine | 0.885 | **0.8849** | 0.80 | -0.0001 | -0.01% | match | 0.8992 | 0.80 | +0.0142 | close |
| Wpbc | 0.594 | **0.5941** | 1.00 | +0.0001 | +0.01% | match | 0.5704 | 1.00 | -0.0236 | GAP |
| **Average** | 0.934 (printed 0.934) | **0.9343** | | -0.0001 | -0.01% | match | 0.9358 | | +0.0014 | match |

abs: 15/15 match, 0 close, 0 gap. squared: 11/15 match, 2 close, 2 gap.

## '1st order' row with our FGAS column (abs, rounded to 3 dp) and the paper's baseline numbers

| DIS | COF | FastABOD | INFLO | kNN | LDOF | LoOP | ODIN | DCROD | ECOD | IForest | OutRanka | WNINOD | FGAS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 2 | 12 |

Paper: FGAS first on 12 datasets.
