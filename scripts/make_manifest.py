#!/usr/bin/env python3
"""Write data/processed/MANIFEST.json: per-split row counts + a content hash, for reproducibility
and for detecting silent dataset drift between a report's numbers and a later re-run."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_json, write_json


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    root = Path(args.processed_dir)
    manifest = {"splits": {}}
    for p in sorted(root.glob("*.jsonl")):
        n_lines = sum(1 for _ in open(p, encoding="utf-8"))
        manifest["splits"][p.stem] = {"rows": n_lines, "sha256_16": sha256_of(p), "bytes": p.stat().st_size}
    stats_path = root / "stats.json"
    if stats_path.exists():
        manifest["build_stats"] = read_json(stats_path)
    write_json(root / "MANIFEST.json", manifest)
    print(f"Wrote {root / 'MANIFEST.json'} covering {len(manifest['splits'])} splits.")


if __name__ == "__main__":
    main()
