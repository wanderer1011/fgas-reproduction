"""Per-run provenance: run id, git commit, config snapshot, environment, timings."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(exp_path: str | Path, overrides: dict | None = None) -> dict:
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    exp = yaml.safe_load(Path(exp_path).read_text())
    return deep_merge(deep_merge(base, exp), overrides or {})


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def env_info(device: str | None = None) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                                     capture_output=True).stdout.splitlines(),
    }
    if device == "cuda":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        info["gpu"] = {
            "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        }
    return info


class Run:
    """Context manager creating runs/<timestamp>_<name>/ with meta.json and config.yaml."""

    def __init__(self, cfg: dict, name: str | None = None):
        self.cfg = cfg
        name = name or cfg.get("name", "run")
        self.id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        self.dir = ROOT / "runs" / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.timings: dict[str, float] = {}
        self.notes: dict = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        cfg_text = yaml.safe_dump(self.cfg, sort_keys=True)
        (self.dir / "config.yaml").write_text(cfg_text)
        dirty = _git("status", "--porcelain")
        self.meta = {
            "run_id": self.id,
            "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(dirty) and dirty != "unknown",
            "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
            "seed": self.cfg.get("seed"),
            "env": env_info(self.cfg.get("backend", {}).get("device")),
        }
        return self

    def __exit__(self, *exc):
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self.meta["wall_seconds"] = round(time.perf_counter() - self.t0, 3)
        self.meta["timings"] = self.timings
        self.meta["notes"] = self.notes
        self.meta["status"] = "ok" if exc[0] is None else f"error: {exc[1]!r}"
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return False
