"""Download the 15 benchmark .mat files (and the README example) from the official
FGAS repo at a pinned commit, and record SHA-256 checksums.

Usage: python scripts/download_data.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    src = cfg["source"]
    base = f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}"
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    files = [(f"{src['path']}/{d['file']}", d["file"]) for d in cfg["datasets"]]
    files.append(("Code/Example.mat", "Example.mat"))  # README parity example

    manifest = {}
    for remote, local in files:
        dst = out / local
        if not dst.exists():
            url = f"{base}/{remote}"
            print(f"GET {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
        manifest[local] = {"sha256": sha256(dst), "bytes": dst.stat().st_size,
                           "source": f"{src['repo']}@{src['commit']}:{remote}"}

    man_path = ROOT / "data" / "checksums.json"
    if man_path.exists():  # verify against committed checksums
        old = json.loads(man_path.read_text())
        bad = [k for k in old if k in manifest and old[k]["sha256"] != manifest[k]["sha256"]]
        if bad:
            print("CHECKSUM MISMATCH:", bad)
            return 1
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} files OK, total {sum(v['bytes'] for v in manifest.values())/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
