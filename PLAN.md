# PLAN — FGAS reproduction (Phase 1)

This plan builds on the approved [NOTES.md](NOTES.md). Assumption numbers like "A#1" refer to §5 of NOTES.md.

**Scope:**
- **Full reproduction:** the FGAS column of Table 2, Fig. 4 (σ sweep), Fig. 2 (noise), and Fig. 3 (stats).
- **Partial reproduction:** the baselines and Fig. 1. Only the baselines with a standard library get re-run; the others are copied from the paper and labelled as such (A#14).

**Machine:**
- Windows 11, AMD Ryzen 7 7840HS (8 cores / 16 threads), 15.3 GB RAM.
- **NVIDIA RTX 4060 Laptop GPU:** 8 GB VRAM (about 6 GB free; the display uses about 2 GB). Driver 621.01 supports CUDA 13.5.

**GPU use (your request):**
- FGAS runs through a backend switch (`device: cpu | cuda`). The GPU path uses **CuPy**, which has the same API as numpy, so one code path serves both.
- A CPU float64 numpy path is always kept as the reference implementation that every GPU result is checked against.
- **Precision caveat:** the RTX 4060 runs float64 at 1/64 of its float32 speed. Whether the GPU actually beats the 16-thread CPU at float64 has to be measured (M3). The float32 path is only allowed if it passes the parity check in §5.

---

## 1. Repo layout

```
Fuzzy Project/
├── 2023-FGAS.pdf  NOTES.md  PLAN.md  RESULTS.md (Phase 3)  README.md
├── requirements.txt            # pinned core deps
├── requirements-gpu.txt        # cupy-cuda13x[ctk] (pinned)
├── requirements-baselines.txt  # pinned optional deps (pyod, PyNomaly), installed only after you approve
├── configs/
│   ├── base.yaml               # d, tol, max_iter, fgd variant, σ grid, scaling, imputation, seed
│   ├── datasets.yaml           # 15 entries: file, abbr, expected n/m/#outliers (Table 1), paper AUC (Table 2)
│   ├── exp_smoke.yaml          # tiny run: Iris + Wine, σ = 0.6 only
│   ├── exp_table2.yaml         # all datasets × σ grid × {squared, abs}
│   ├── exp_noise.yaml          # noise levels 0…0.5, seeds, σ policy
│   └── exp_baselines.yaml      # method → k grid / n_estimators / seeds
├── paper/
│   └── table2_paper.csv        # Table 2 transcribed from the paper (all 14 columns), plus Fig. 2/4 anchor readings
├── data/
│   ├── raw/                    # downloaded .mat files (git-ignored), with a checksum manifest that is committed
│   └── README.md               # where the data came from, SHA-256 of each file, shapes found
├── src/fgas/
│   ├── data.py                 # load .mat → (X, y); label detection; mode imputation; min-max scaling
│   ├── model.py                # FGAS: granule cardinalities, FGD (squared|abs), transition matrix, power iteration, AS
│   ├── metrics.py              # AUC (sklearn) and Eq. 13 brute force; ROC points per Eqs. 11–12
│   ├── noise.py                # attribute-noise injection (A#7)
│   ├── baselines.py            # wrappers that give every baseline the same score_samples(X) interface
│   ├── stats.py                # Friedman τ_F, critical value, Nemenyi CD, average ranks
│   ├── plots.py                # Fig. 1–4 in the paper's layout
│   └── runlog.py               # run id, seed, git commit + dirty flag, config snapshot, pip freeze, platform, timings
├── scripts/
│   ├── download_data.py        # fetch the 15 .mat files from the official FGAS repo (pinned commit) and verify checksums
│   ├── inspect_data.py         # shapes / labels / NaNs vs Table 1 → data/README.md
│   ├── run_fgas.py             # --config exp_*.yaml → runs/<id>/scores.parquet, auc.csv
│   ├── run_noise.py
│   ├── run_baselines.py
│   ├── make_table2.py          # Table 2 (ours vs paper) as markdown + CSV
│   ├── make_figures.py         # fig1_roc.png, fig2_noise.png, fig3_cd.png, fig4_sigma.png
│   └── make_results.py         # assembles RESULTS.md
├── tests/
│   ├── test_model.py           # parity with the official code; d-equivalence; row sums; edge cases
│   ├── test_metrics.py         # sklearn AUC == Eq. 13 brute force (with ties)
│   ├── test_noise.py           # exact number of corrupted cells; values stay within [min, max]
│   └── test_stats.py           # reproduces the paper's 1.5610 / 4.7659 from Table 2
├── runs/                       # one folder per run: meta.json, config.yaml, outputs (git-ignored)
└── results/                    # final CSVs and figures that RESULTS.md cites (committed)
```

## 2. Dependencies (pinned)

**Core:** `requirements.txt`. Every package is already installed on this machine, so nothing needs downloading.

| Package | Version | Use |
|---|---|---|
| Python | 3.14.0 | runtime |
| numpy | 2.4.2 | FGAS core |
| scipy | 1.17.0 | `.mat` loading, `cdist`, F-distribution |
| scikit-learn | 1.8.0 | `roc_auc_score`, `roc_curve` |
| pandas | 3.0.0 | result tables |
| matplotlib | 3.10.8 | figures |
| PyYAML | 6.0.3 | configs |
| pytest | 9.0.2 | tests |
| joblib | 1.5.3 | parallelising the σ / dataset / seed loops |
| PyMuPDF | 2.1 | (already installed; used only for reading the paper) |

**GPU, installed at the start of M2:** `requirements-gpu.txt`. Approving this plan counts as approving this install.

| Package | Version | Use | Size |
|---|---|---|---|
| cupy-cuda13x[ctk] | 14.2.0 | GPU arrays for FGAS (granules, FGD, power iteration) | 36 MB wheel, plus the CUDA 13 runtime libraries (cuBLAS, cudart, NVRTC, …) as `nvidia-*`/`cuda-toolkit` wheels. Roughly **0.6–1 GB total**; the exact size will be reported after install. |

- No system CUDA toolkit install is needed, because the `[ctk]` extra brings its own runtime.
- If CuPy fails to initialise on this driver, I'll stop and tell you. The CPU path still works, so nothing is blocked.
- PyTorch was considered and rejected: its CUDA 13.2 wheel for Python 3.14 is about 2.5 GB, and its only benefit here would be the same array maths.

**Optional, needs your approval before installing:** `requirements-baselines.txt`

| Package | Version | Use | Notes |
|---|---|---|---|
| pyod | 3.6.7 | kNN, COF, FastABOD (`ABOD(method='fast')`), ECOD, IForest | Pulls in **numba/llvmlite** (~50–100 MB). I'll check that it installs cleanly on Python 3.14 first. |
| PyNomaly | 0.4.0 | LoOP | small, pure Python |

If pyod won't install on 3.14, the fallback is to implement kNN, IForest and ECOD directly with scikit-learn/numpy (each is under 30 lines) and mark COF and FastABOD as "paper-reported".

**Environment:** use the existing Python 3.14 interpreter, with no venv, so nothing large has to be re-downloaded. Every run logs `pip freeze`, the git commit and platform info into `runs/<id>/meta.json`. If you'd rather have an isolated `.venv`, say so; it means re-installing about 150 MB of wheels.

## 3. Implementation notes (so nothing is hardcoded)

- **All numbers come from YAML:**
  - `d` (0.1) and `tol` (1e-3, A#2)
  - `max_iter` (10000)
  - `fgd: squared | abs` (A#1)
  - the σ grid as `{start: 0.0, stop: 0.95, step: 0.05}` (A#4)
  - `include_sigma_1: true` (logged only)
  - `scaling: minmax`, `impute: mode`
  - seeds and noise levels
  - **`device: cpu | cuda`**, **`dtype: float64 | float32`**, `gpu_chunk_rows` (VRAM cap)
- **Backend:** `model.py` uses `xp = cupy if device == "cuda" else numpy`, and every function is written against `xp`. Results are copied back to the host before the AUC is computed, so the metrics, stats and plots code stays CPU/numpy.
- **σ batching on GPU:** for each attribute, the n×n block of `1−|Δ|` is computed **once** and thresholded for all 20 σ values together, giving a (20, n) cardinality tensor per attribute. Power iteration then runs as a batched matrix product over the 20 σ values (`π (20×n) · P_σ`), with blocks chunked to fit about 5 GB of VRAM. This is the main speed-up, for both the sweep and the noise experiment.
- **Efficient FGAS** (same maths as Eq. 3/7, verified against a naive implementation in tests):
  - **Granule cardinality per attribute:** `|[o]_a| = Σ_p (1−|Δ|)·[1−|Δ| ≥ σ]`. Computed with an O(n²) block per attribute, chunked to cap memory at about 1 GB. Datasets with n ≤ 1000 use plain broadcasting.
  - **FGD:** build the n×m matrix of relative cardinalities `T`, then `squared → cdist(T,T,'sqeuclidean')` and `abs → cdist(T,T,'cityblock')`. That's one n×n matrix (Satellite: 330 MB).
  - **P:** row-normalise. Zero-sum rows are detected and that σ is marked NaN (A#12).
  - **Power iteration:** exactly Eq. 9 with a scalar d. Log the iteration count and final residual.
- **Determinism:** FGAS has no randomness. The noise experiment and IForest use `numpy.random.default_rng(seed)`, with seeds listed in the config and logged.
- **Run logging (`runlog.py`):** `meta.json` holds the git SHA (with a dirty-tree flag), a config hash and copy, seed(s), `pip freeze`, python/OS/CPU info, start and end times, and per-dataset wall time.
  - It also records **device and dtype, GPU name, driver/CUDA version, CuPy version and peak VRAM**.
- **GPU nondeterminism:** CuPy reductions can sum in a different order from numpy, so results agree to rounding but aren't bit-identical. The parity checks in §5 bound this. FGAS uses no random numbers on the GPU; noise is generated on the CPU with a seeded `default_rng` and then copied over.

## 4. Order of work (checklist)

Each milestone ends with tests passing and a git commit. ⏸ marks a point where I stop and ask you.

### M0 — Scaffolding
- [x] `git init`, `.gitignore` (data/raw, runs, `__pycache__`), skeleton folders, `requirements.txt`, empty configs
- [x] `paper/table2_paper.csv`, transcribed from the rendered Table 2, then checked by recomputing its column averages and "1st order" row
- [x] Commit: `M0: project scaffold and paper reference tables`

### M1 — Data
- [x] `download_data.py`: fetch the 15 `.mat` files (~2 MB total) from `BELLoney/FGAS` at a **pinned commit SHA**, plus `Code/Example.mat` (280 B) for the parity test. Write SHA-256 checksums to `data/README.md`. These files are tiny, so approving this plan counts as approving the download.
- [x] `inspect_data.py`: report the keys, shapes, label column/field, NaN count, outlier count and any constant columns, compared with Table 1 (A#9, A#10)
- [x] ⏸ **If any shape or outlier count disagrees with Table 1, or the label column is ambiguous, stop and ask.**
- [x] Commit: `M1: data download, checksums, and inspection report`

### M2 — FGAS core + tests
- [x] Install `cupy-cuda13x[ctk]==14.2.0`, check that `cupy.cuda.runtime.getDeviceCount() == 1`, and run a tiny kernel. Report the installed size.
- [x] `model.py` with both FGD variants and the cpu/cuda backend
- [x] Test **GPU = CPU:** on random data and on Iris/Wine, cuda-float64 AS equals cpu-float64 AS to ≤ 1e-8, and cuda-float32 AS to ≤ 1e-4. Tests are skipped automatically when no GPU is present.
- [x] Test that batched-σ results equal a separate run for each σ
- [x] Test **parity with the official code:** `fgd=abs, tol=1e-4` on `Example.mat` with σ = 0.6 must reproduce the 10 scores printed in the official README (0.1390, 0.0539, …, 1.0000, 0.1325, 0.1692) to ±1e-4
- [x] Test that the efficient version equals a naive implementation of Eq. 3/7/8 (both variants, random small data, to 1e-10)
- [x] Test **d-equivalence** (A#3): scalar d vs a uniform restart vector gives the same AS to 1e-6 at a tight tolerance
- [x] Test that every row of P sums to 1; AS lies in [0,1] with min 0 and max 1; zero-sum rows are handled
- [x] `metrics.py` plus a test that sklearn's AUC equals the Eq. 13 brute force, including tied scores
- [x] Commit: `M2: FGAS implementation (squared/abs FGD) with parity tests`

### M3 — Smallest viable reproduction (minutes)
- [x] `exp_smoke.yaml`: Iris (111×4) and Wine (129×13), σ = 0.6, both variants → AUC, iteration count, runtime
- [x] Then the full σ grid on Iris and Wine → best σ and best AUC vs Table 2 (Iris 1.000, Wine 0.885) and vs the Fig. 4 readings
- [x] **Benchmark on Satellite** (largest n = 6435) and Musk (largest m = 166), full 20-σ sweep, under three settings: **cpu-float64**, **cuda-float64**, **cuda-float32**. Record wall time, peak VRAM/RAM, and the best-σ AUC difference vs cpu-float64.
- [x] **Backend choice rule:** use the fastest setting that passes the parity criterion in §5. If only cuda-float32 is fast enough, the sweep runs in float32 and the **best σ for each dataset is then re-scored in float64**. All numbers reported in Table 2 and RESULTS.md come from float64.
- [x] Commit: `M3: end-to-end smoke run on Iris/Wine`
- [x] ⏸ **Short report to you:** smoke numbers, the squared-vs-abs early signal, the CPU/GPU benchmark and the chosen backend, and runtime estimates for M4/M6/M7

### M4 — Table 2 (FGAS column) + Fig. 4
- [x] `exp_table2.yaml`: 15 datasets × 20 σ × 2 FGD variants = 600 FGAS runs, plus σ = 1.0 logged
- [x] Runs on the backend chosen in M3. Estimated well under 30 min (Satellite dominates). **If M3 timing projects more than 30 min, ⏸ ask first.**
- [x] `make_table2.py` → ours vs paper per dataset, best σ, and Δ
- [x] `make_figures.py` → Fig. 4 in the paper's layout (2 panels: datasets 1–8 and 9–15, σ on 0–0.95, MATLAB-like markers)
- [x] Commit: `M4: FGAS sigma sweep, Table 2 FGAS column, Fig. 4`

### M5 — Statistics + Fig. 3
- [x] `stats.py`, plus a test reproducing the paper's critical value **1.5610** and **CD 4.7659** from the paper's Table 2. τ_F should come out at 15.07 (paper: 15.1347).
- [x] Fig. 3 for (a) the paper's Table 2, as a sanity check of the plotting code, and (b) a mixed table with our FGAS, our re-run baselines, and the remaining paper-reported baselines (labelled)
- [x] Draw the CD diagram mirrored, with 14 on the left and 1 on the right, as in the paper
- [x] Commit: `M5: Friedman/Nemenyi and CD diagram`

### M6 — Baselines + Fig. 1
- [x] ⏸ **Ask before installing pyod/PyNomaly** (heavy: numba).
- [x] `exp_baselines.yaml`:
  - kNN, COF, FastABOD, LoOP: k = 1…60
  - IForest: 100 trees, 5 seeds, report mean ± std
  - ECOD: no parameters
- [x] Cost: COF and FastABOD scale as O(n²)–O(n²k) over 60 k values. On Satellite and Page this may take more than 30 min, so **⏸ ask before launching if the estimate exceeds 30 min**. A fallback is a reduced k grid on the largest datasets, which would be logged.
- [x] Fig. 1: ROC curves (5×3 grid, FPR%/TPR%, FGAS in black) for FGAS plus the re-run baselines only. Paper-only baselines can't be plotted because their scores don't exist; that will be stated in the caption.
- [x] Commit: `M6: baselines and ROC figure`

### M7 — Noise robustness (Fig. 2)
- [x] `noise.py` per A#7: for each attribute, ⌈x·n⌉ random samples are replaced by U[min_a, max_a] on the raw data, then scaled
- [x] Main protocol: x ∈ {0, 0.05, …, 0.5} × 5 seeds × 20 σ re-tuned per level (A#6), giving 15 × 11 × 5 × 20 = 16,500 FGAS runs. Secondary protocol: σ fixed at the clean best value.
- [x] This is where the GPU helps most. Each (dataset, level, seed) runs the batched 20-σ GPU kernel once, so 825 batched runs instead of 16,500 single ones. ⏸ Ask before launching if the timing measured in M3/M4 projects more than 30 min.
- [x] Fig. 2 in the paper's layout (2 panels, noise level 0–0.5), plotting the mean over seeds with a min–max band. A single-seed version is also produced to match the paper exactly.
- [x] Commit: `M7: attribute-noise experiment and Fig. 2`

### M8 — Phase 3 write-up
- [x] `RESULTS.md`:
  - side-by-side tables (ours | paper | Δ | % deviation | status)
  - the squared-vs-abs finding
  - best σ vs Fig. 4
  - the Fig. 2/Fig. 4 inconsistencies (A#5, A#6)
  - likely causes for each gap
  - runtimes and environment
- [x] Commit: `M8: results report`

## 5. How each result is validated

AUCs are printed to 3 decimals, so rounding alone allows ±0.0005.

| Result | Target | Pass ("match") | "Close" | Otherwise |
|---|---|---|---|---|
| Parity with official code (Example.mat, σ = 0.6, abs) | README scores | max abs error ≤ 1e-4 | — | **Bug.** Stop and fix before anything else. |
| d-equivalence, naive = efficient | identical | ≤ 1e-6 / 1e-10 | — | bug |
| GPU vs CPU reference (scores) | cpu-float64 | cuda-float64 ≤ 1e-8; cuda-float32 ≤ 1e-4 max abs error in AS | — | bug, or float32 rejected |
| GPU vs CPU reference (AUC, all 15 datasets × 20 σ) | cpu-float64 | \|ΔAUC\| ≤ 1e-4 everywhere, and the same best σ | — | float32 rejected → float64 only |
| FGAS AUC per dataset (Table 2) | paper value | \|Δ\| ≤ 0.005 | ≤ 0.02 | "gap": investigate and explain in RESULTS.md, never tune to fit |
| FGAS average AUC | 0.934 | \|Δ\| ≤ 0.005 | ≤ 0.015 | gap |
| Best σ per dataset | Fig. 4 peak | within ±0.05 | ±0.10 | note it |
| Fig. 4 curve shape | readings in NOTES §3 E5 | same peak σ and same collapse at 0.95 | qualitative agreement | note it |
| Fig. 2, noise = 0 | Table 2 value (except Thyroid, A#6) | equals our Table 2 value (by construction) | — | — |
| Fig. 2 trend | flat except Ecoli/Page/Diab/Pima declines | max drop within ±0.01 of the paper's readings | same direction | note it |
| Stats on the paper's table | 1.5610 / 4.7659 / 15.1347 | exact / exact / within 0.5 % | — | bug in stats.py |
| Re-run baselines | paper column | \|Δ\| ≤ 0.01 | ≤ 0.03 | expected for some: implementations differ (A#14) |
| Ranking claim | FGAS first on 12/15, CD separation | same set of significantly different methods | — | report honestly |

**Squared vs abs (updated after M3, user decision):** both variants are always reported. **The headline is abs (official code)**, because it reproduces Fig. 4 and Table 2 on Iris/Wine while squared doesn't. ~~The spec version (squared) is the headline.~~ Whichever variant matches more datasets is identified as "most likely what the authors ran", as evidence rather than a silent choice.

## 6. Points where I'll stop and ask (summary)
1. ⏸ After M1, if the data doesn't match Table 1.
2. ⏸ After M3, with a smoke report and runtime estimates.
3. ⏸ Before M4, only if it's projected to exceed 30 min.
4. ⏸ Before installing pyod/PyNomaly (M6), and before any baseline run over 30 min.
5. ⏸ Before the M7 noise sweep (likely over 30 min).
6. ⏸ Any time a paper detail turns out to be contradictory in a way NOTES.md doesn't already cover.


---
**Status 2026-11-06:** M0–M8 complete (commits 8d28a90 … M8). Deviations from this plan:
- σ grid changed to 0…1 (A#4 resolved in M4)
- FGD headline switched to abs (user, after M3)
- noise protocol switched to one attribute (user, after the M7 pre-check)
- σ batching on GPU is done for the granule step only; FGD and power iteration run per σ, because 20 n×n matrices don't fit in VRAM for Satellite
- the noise experiment used the step-0.1 σ grid that matches Fig. 2

See RESULTS.md.
