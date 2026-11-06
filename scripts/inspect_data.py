"""Inspect downloaded datasets vs Table 1 and write data/README.md."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import dataset_specs, load_dataset  # noqa: E402


def main() -> int:
    specs = dataset_specs()
    sums = json.loads((ROOT / "data" / "checksums.json").read_text())
    rows, problems = [], []
    for abbr, s in specs.items():
        ds = load_dataset(abbr)
        n, m = ds.X_raw.shape
        n_out = int(ds.y.sum())
        nan = int(np.isnan(ds.X_raw).sum())
        const = int((np.nanmax(ds.X_raw, 0) == np.nanmin(ds.X_raw, 0)).sum())
        lo, hi = np.nanmin(ds.X_raw), np.nanmax(ds.X_raw)
        dup = n - len(np.unique(ds.X_raw, axis=0))
        ok = (n, m, n_out) == (s["n_obj"], s["n_attr"], s["n_outliers"])
        if not ok:
            problems.append(abbr)
        rows.append(f"| {s['id']} | {abbr} | {s['file']} | {m} ({s['n_attr']}) | {n} ({s['n_obj']}) | "
                    f"{n_out} ({s['n_outliers']}) | {nan} | {const} | {dup} | [{lo:.3g}, {hi:.3g}] | "
                    f"{'OK' if ok else 'MISMATCH'} | `{sums[s['file']]['sha256'][:12]}` |")
    src = sums[next(iter(sums))]["source"].split(":")[0]
    md = [
        "# Data",
        "",
        f"Source: `{src}` (pinned commit), downloaded by `scripts/download_data.py`.",
        "Full SHA-256 checksums: `data/checksums.json`. Raw files are git-ignored.",
        "",
        "Layout of every file: a single matrix `trandata` (n x (m+1)); columns 1..m are features,",
        "the last column is the binary label (1 = outlier).",
        "",
        "Values in parentheses are Table 1 of the paper.",
        "",
        "| # | Abbr | File | #attr | #obj | #outliers | NaNs | const cols | dup rows | raw range | vs Table 1 | sha256 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    (ROOT / "data" / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[-len(rows) - 2:]))
    print("PROBLEMS:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
