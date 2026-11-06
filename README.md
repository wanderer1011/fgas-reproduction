# FGAS reproduction (GPU)

An independent reproduction of

> C. Liu, Z. Yuan, B. Chen, H. Chen, D. Peng. **Fuzzy granular anomaly detection using Markov random walk.**
> *Information Sciences* 646 (2023) 119400. DOI [10.1016/j.ins.2023.119400](https://doi.org/10.1016/j.ins.2023.119400).
> Official code: [BELLoney/FGAS](https://github.com/BELLoney/FGAS)

FGAS is reimplemented in numpy/CuPy, with float64 on an NVIDIA GPU and a CPU reference path. All of the paper's experiments are re-run, plus three real-world application datasets.

## Headline results

| What | Result |
|---|---|
| **Table 2, FGAS column** | **15/15 datasets match** to within the paper's 3-dp rounding (max \|Δ\| = 0.0005; average 0.9343 vs 0.934) |
| Fig. 4 (AUC vs σ) | reproduced curve for curve |
| Fig. 2 (attribute noise) | mean abs. error 0.0037 vs the paper; 160/165 points inside our seed band |
| Fig. 3 (Friedman/Nemenyi) | F_crit 1.5610 and CD 4.7659 exact |
| Baselines (pyod/PyNomaly) | ECOD and LoOP 15/15 exact; IForest 12/15; kNN 11/15; COF 7/15; **FastABOD not reproducible** |

Findings about the paper:
- **Eq. 7 as printed (squared differences) doesn't produce the published numbers.** The official code's absolute-difference distance does.
- σ must include **1.0**.
- Fig. 2 uses a step-0.1 σ grid and **one-attribute** noise.
- By its own Table 2, FGAS vs DIS is *not* significantly different (rank gap 4.667 < CD 4.7659).

Full details: [`RESULTS.md`](RESULTS.md). Application datasets (mammography, annthyroid, SECOM): [`results/EXTENSIONS.md`](results/EXTENSIONS.md).

## Repository layout

| Path | Contents |
|---|---|
| `NOTES.md` | paper digest, every experiment, and the ambiguities/assumptions log (with resolutions) |
| `PLAN.md` | implementation plan, milestones, validation tolerances |
| `RESULTS.md` | side-by-side comparison with the paper, % deviation, likely causes of gaps |
| `src/fgas/` | `model.py` (FGAS, cpu/cuda), `data.py`, `metrics.py`, `baselines.py`, `noise.py`, `stats.py`, `plots.py`, `runlog.py` |
| `scripts/` | download, run (FGAS sweep, baselines, noise), and figure/table builders |
| `configs/` | YAML for every hyperparameter and experiment |
| `results/` | regenerated tables and figures (Figs. 1–4) |
| `paper/` | Table 2 transcribed from the paper; values read off Figs. 2/4 |
| `tests/` | 28 tests, incl. parity with the official README example |

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt        # optional: NVIDIA GPU (CuPy, CUDA 13)
python -m pip install -r requirements-baselines.txt  # optional: pyod / PyNomaly baselines
python scripts/download_data.py                      # 15 .mat files from the official repo (pinned commit)
python -m pytest -q
python scripts/run_fgas.py --config configs/exp_table2.yaml        # Table 2 + Fig. 4 data (~2 min on GPU)
python scripts/make_table2.py
python scripts/make_figures.py fig4
```

Without a GPU, set `backend.device: cpu` in `configs/base.yaml` (or pass `--device cpu`). It's slower: Satellite takes about 5 min for 20 σ values.

Every run writes `runs/<timestamp>_<name>/` with `config.yaml` and `meta.json` (git commit, seeds, `pip freeze`, GPU/driver info, timings).

Tested with Python 3.14.0, numpy 2.4.2, scipy 1.17.0, scikit-learn 1.8.0, CuPy 14.2.0 (RTX 4060 Laptop GPU).

## Data and paper

- Benchmark data: downloaded from the authors' repository by `scripts/download_data.py`; not stored here. SHA-256 checksums are in `data/checksums.json`.
- Extension data: ADBench (mammography, annthyroid) and UCI SECOM, via `scripts/download_ext.py`.
- `2023-FGAS.pdf` is the published article, © 2023 Elsevier Inc., included for reference only.
