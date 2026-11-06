"""Download extension datasets (configs/datasets_ext.yaml), each into its own folder, with SHA-256."""
import hashlib, json, sys, urllib.request, zipfile
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "datasets_ext.yaml").read_text())
    man = {}
    for d in cfg["datasets"]:
        out = ROOT / d["dir"]; out.mkdir(parents=True, exist_ok=True)
        dst = out / d["file"]
        if not dst.exists():
            print("GET", d["url"])
            with urllib.request.urlopen(d["url"], timeout=120) as r:
                dst.write_bytes(r.read())
        if dst.suffix == ".zip":
            with zipfile.ZipFile(dst) as z:
                for name in z.namelist():  # extract only plain files, no paths
                    if name.endswith(".data") and "/" not in name and ".." not in name:
                        (out / name).write_bytes(z.read(name))
        man[d["abbr"]] = {"file": str(dst.relative_to(ROOT)), "bytes": dst.stat().st_size,
                          "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "url": d["url"]}
    (ROOT / "data" / "checksums_ext.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
