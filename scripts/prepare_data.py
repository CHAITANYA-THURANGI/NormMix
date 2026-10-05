#!/usr/bin/env python3
"""Build the processed dataset (synthetic seed + optional downloaded external data).

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --external data/external --config experiments/configs/attn_bahdanau.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--external", default=None, help="Root of downloaded datasets, e.g. data/external")
    ap.add_argument("override", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    from src.datasets.pipeline import build_dataset
    stats = build_dataset(cfg, out_dir=args.out, external_root=args.external)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
