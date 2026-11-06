# FGAS reproduction

Reproduction of Liu et al., *Fuzzy granular anomaly detection using Markov random walk*, Information Sciences 646 (2023) 119400.

- `NOTES.md` — paper digest, experiments, ambiguities and assumptions
- `PLAN.md` — implementation plan and validation criteria
- `RESULTS.md` — results vs paper (Phase 3)

Quick start:

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-gpu.txt   # optional, NVIDIA GPU
    python scripts/download_data.py
    python -m pytest -q
    python scripts/run_fgas.py --config configs/exp_smoke.yaml
