"""Re-run library baselines on the paper's grids.

Usage: python scripts/run_baselines.py --config configs/exp_baselines.yaml [--datasets Iris Wine] [--methods kNN]
Writes runs/<id>/auc.csv (dataset, method, k, seed, auc, seconds, note) and scores.npz (best setting per method).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.baselines import score  # noqa: E402
from fgas.data import dataset_specs, load_dataset, preprocess  # noqa: E402
from fgas.metrics import auc  # noqa: E402
from fgas.runlog import Run, load_config  # noqa: E402


def job(abbr, method, k, seed, data_dir, pre):
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    ds = load_dataset(abbr, data_dir)
    X = preprocess(ds.X_raw, pre["impute"], pre["scaling"])
    t0 = time.perf_counter()
    s, note = score(method, X, k, seed)
    secs = time.perf_counter() - t0
    a = auc(ds.y, s) if s is not None else float("nan")
    return dict(dataset=abbr, method=method, k=k, seed=seed, auc=a, seconds=secs, note=note), s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--methods", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = {"device": "cpu", "dtype": "float64"}
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.methods:
        cfg["methods"] = {m: cfg["methods"][m] for m in args.methods}
    datasets = list(dataset_specs()) if cfg["datasets"] == "all" else cfg["datasets"]
    tasks = []
    for abbr in datasets:
        for m, p in cfg["methods"].items():
            ks = range(p["k"][0], p["k"][1] + 1) if "k" in p else [None]
            for k in ks:
                for seed in p.get("seeds", [cfg["seed"]]):
                    tasks.append((abbr, m, k, seed))
    # longest-first: big datasets and large k first, for better load balance
    size = {a: dataset_specs()[a]["n_obj"] for a in datasets}
    tasks.sort(key=lambda t: -(size[t[0]] * (t[2] or 1)))
    with Run(cfg) as run:
        print(f"run {run.id}: {len(tasks)} fits on {cfg['n_jobs']} processes", flush=True)
        out = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
            delayed(job)(a, m, k, s, ROOT / cfg["data"]["dir"], cfg["preprocess"]) for a, m, k, s in tasks)
        df = pd.DataFrame([r for r, _ in out])
        df.to_csv(run.dir / "auc.csv", index=False)
        # keep scores of the best k per (dataset, method) (seed 0 for IForest) for ROC plots
        keep = {}
        for (r, s) in out:
            if s is None or r["seed"] not in (0, cfg["seed"]):
                continue
            key = f"{r['dataset']}|{r['method']}"
            if key not in keep or r["auc"] > keep[key][0]:
                keep[key] = (r["auc"], r["k"], s)
        np.savez_compressed(run.dir / "scores.npz", **{f"{k}|k={v[1]}": v[2] for k, v in keep.items()})
        run.timings["fit_seconds_total"] = round(float(df.seconds.sum()), 1)
        run.notes["failed_fits"] = int(df.auc.isna().sum())
        run.notes["notes"] = df.note.dropna().value_counts().head(20).to_dict()
    summ = df.groupby(["dataset", "method"]).agg(best_auc=("auc", "max"), fits=("auc", "size"),
                                                 secs=("seconds", "sum"))
    print(summ.round(4).to_string())
    print(f"wrote {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
