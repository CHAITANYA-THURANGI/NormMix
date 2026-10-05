#!/usr/bin/env python3
"""Train every config in experiments/configs/ (skips ones already trained), then compare on test splits.

Usage: python scripts/run_all_experiments.py [--epochs N] [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--configs-dir", default="experiments/configs")
    args = ap.parse_args()
    configs = sorted(Path(args.configs_dir).glob("*.yaml"))
    if not configs:
        print(f"No configs found in {args.configs_dir}"); return
    checkpoints = []
    for cfg_path in configs:
        run_name = cfg_path.stem
        best = ROOT / "experiments" / "checkpoints" / run_name / "best.pt"
        if best.exists() and not args.force:
            print(f"[skip] {run_name} already trained ({best})")
        else:
            cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path), f"train.run_name={run_name}"]
            if args.epochs:
                cmd.append(f"train.epochs={args.epochs}")
            print("+", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=ROOT)
        checkpoints.append(str(best))
    cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", *checkpoints,
          "--rule-baseline", "data/processed/rule_baseline.json",
          "--split", "test", "test_hard", "test_unseen_words", "gold_demo",
          "--out", "experiments/results/comparison.json"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
