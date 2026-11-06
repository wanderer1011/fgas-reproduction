"""Run FGAS over datasets x sigma grid x FGD variants (x backend settings).

Usage:
  python scripts/run_fgas.py --config configs/exp_smoke.yaml
  python scripts/run_fgas.py --config configs/exp_table2.yaml --device cuda --dtype float32

Writes runs/<id>/auc.csv (one row per dataset/variant/setting/sigma), scores.npz, meta.json.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import dataset_specs, load_dataset, preprocess  # noqa: E402
from fgas.metrics import auc  # noqa: E402
from fgas.model import fgas_sweep  # noqa: E402
from fgas.runlog import Run, load_config  # noqa: E402


def sigma_list(cfg: dict) -> tuple[list[float], list[float]]:
    """(search grid, extra logged-only sigmas)."""
    if cfg.get("sigmas", "grid") != "grid":
        return [float(s) for s in cfg["sigmas"]], []
    g = cfg["sigma_grid"]
    k = int(round((g["stop"] - g["start"]) / g["step"]))
    grid = [round(g["start"] + i * g["step"], 10) for i in range(k + 1)]
    return grid, ([1.0] if g.get("log_sigma_1") else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device")
    ap.add_argument("--dtype")
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()

    over = {"backend": {k: v for k, v in {"device": args.device, "dtype": args.dtype}.items() if v}}
    cfg = load_config(args.config, over)
    if args.datasets:
        cfg["datasets"] = args.datasets
    datasets = cfg.get("datasets", "all")
    if datasets == "all":
        datasets = list(dataset_specs())
    grid, extra = sigma_list(cfg)
    settings = cfg.get("settings") or [cfg["backend"]]
    f = cfg["fgas"]

    rows, scores = [], {}
    with Run(cfg) as run:
        print(f"run {run.id}")
        for st in settings:
            dev, dt = st["device"], st["dtype"]
            for abbr in datasets:
                ds = load_dataset(abbr, ROOT / cfg["data"]["dir"])
                X = preprocess(ds.X_raw, cfg["preprocess"]["impute"], cfg["preprocess"]["scaling"])
                for fgd in cfg.get("fgd_variants", [f["fgd"]]):
                    if dev == "cuda":  # warm-up so timings exclude CUDA context / kernel compilation
                        fgas_sweep(X[:32], [0.5], fgd=fgd, device=dev, dtype=dt)
                        import cupy as cp
                        cp.cuda.Device().synchronize()
                        cp.get_default_memory_pool().free_all_blocks()
                    t0 = time.perf_counter()
                    res = fgas_sweep(X, grid + extra, fgd=fgd, d=f["d"], tol=f["tol"],
                                     max_iter=f["max_iter"], device=dev, dtype=dt,
                                     chunk_rows=cfg["backend"]["gpu_chunk_rows"])
                    secs = time.perf_counter() - t0
                    peak = res[0].extra.get("peak_vram_mb")
                    if peak:
                        run.notes[f"peak_vram_mb|{abbr}|{fgd}|{dt}"] = round(peak, 1)
                    run.timings[f"{abbr}|{fgd}|{dev}|{dt}"] = round(secs, 3)
                    for r in res:
                        a = auc(ds.y, r.scores) if r.scores is not None else float("nan")
                        rows.append(dict(dataset=abbr, fgd=fgd, device=dev, dtype=dt, sigma=r.sigma,
                                         in_grid=r.sigma in grid, auc=a, n_iter=r.n_iter,
                                         residual=r.residual, converged=r.converged,
                                         degenerate=r.degenerate))
                        if r.scores is not None:
                            scores[f"{abbr}|{fgd}|{dev}|{dt}|{r.sigma:.2f}"] = r.scores.astype(np.float32)
                    best = max((r for r in rows[-len(res):] if r["in_grid"] and not np.isnan(r["auc"])),
                               key=lambda r: r["auc"], default=None)
                    print(f"  {abbr:6s} {fgd:7s} {dev}/{dt}: {secs:7.2f}s"
                          + (f"  best AUC {best['auc']:.4f} @ sigma={best['sigma']:.2f}" if best else "")
                          + (f"  peak VRAM pool {peak:.0f} MB" if peak else ""))
        df = pd.DataFrame(rows)
        df.to_csv(run.dir / "auc.csv", index=False)
        np.savez_compressed(run.dir / "scores.npz", **scores)
        run.notes["n_rows"] = len(df)
        run.notes["nonconverged"] = int((~df["converged"]).sum())
        run.notes["degenerate"] = int(df["degenerate"].notna().sum())
    print(f"wrote {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
