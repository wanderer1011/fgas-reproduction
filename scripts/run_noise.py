"""Attribute-noise robustness experiment (Fig. 2).

Usage: python scripts/run_noise.py --config configs/exp_noise.yaml [--datasets Iris] [--seeds 0]
Writes runs/<id>/auc.csv with one row per (dataset, level, seed, sigma).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import all_specs, dataset_specs, load_dataset, preprocess  # noqa: E402
from fgas.metrics import auc  # noqa: E402
from fgas.model import fgas_sweep  # noqa: E402
from fgas.noise import add_attribute_noise, noise_rng  # noqa: E402
from fgas.runlog import Run, load_config  # noqa: E402


def grid(g):
    k = int(round((g["stop"] - g["start"]) / g["step"]))
    return [round(g["start"] + i * g["step"], 10) for i in range(k + 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--mode", choices=["all_attributes", "one_attribute"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.mode:
        cfg["noise_mode"] = args.mode
        cfg["name"] = f"noise_{args.mode}"
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    specs = all_specs()
    datasets = list(specs) if cfg["datasets"] == "all" else cfg["datasets"]
    levels, sigmas = grid(cfg["noise_levels"]), grid(cfg["noise_sigma_grid"])
    f, be = cfg["fgas"], cfg["backend"]
    rows = []
    with Run(cfg) as run:
        print(f"run {run.id}", flush=True)
        for abbr in datasets:
            ds = load_dataset(abbr, ROOT / cfg["data"]["dir"])
            t0 = time.perf_counter()
            for li, lvl in enumerate(levels):
                for seed in (cfg["seeds"] if lvl > 0 else cfg["seeds"][:1]):   # level 0 is deterministic
                    Xn = add_attribute_noise(ds.X_raw, lvl, noise_rng(seed, li, specs[abbr]["id"]),
                                             mode=cfg["noise_mode"])
                    X = preprocess(Xn, cfg["preprocess"]["impute"], cfg["preprocess"]["scaling"])
                    for fgd in cfg["fgd_variants"]:
                        res = fgas_sweep(X, sigmas, fgd=fgd, d=f["d"], tol=f["tol"], max_iter=f["max_iter"],
                                         device=be["device"], dtype=be["dtype"], chunk_rows=be["gpu_chunk_rows"])
                        for r in res:
                            rows.append(dict(dataset=abbr, fgd=fgd, level=lvl, seed=seed, sigma=r.sigma,
                                             auc=auc(ds.y, r.scores) if r.scores is not None else np.nan,
                                             converged=r.converged, degenerate=r.degenerate))
            secs = time.perf_counter() - t0
            run.timings[abbr] = round(secs, 1)
            d = pd.DataFrame([r for r in rows if r["dataset"] == abbr])
            best = d.groupby(["level", "seed"]).auc.max().groupby("level").mean()
            print(f"  {abbr:6s} {secs:7.1f}s  best-AUC by level: " +
                  " ".join(f"{v:.3f}" for v in best.values), flush=True)
            pd.DataFrame(rows).to_csv(run.dir / "auc.csv", index=False)   # checkpoint after each dataset
        df = pd.DataFrame(rows)
        run.notes["degenerate"] = int(df.degenerate.notna().sum())
        run.notes["nonconverged"] = int((~df.converged).sum())
    print(f"wrote {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
