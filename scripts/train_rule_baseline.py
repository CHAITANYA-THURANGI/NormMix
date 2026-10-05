#!/usr/bin/env python3
"""Fit and save the rule/dictionary baseline from the training split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.pipeline import load_split
from src.models.rule_baseline import RuleBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out", default="data/processed/rule_baseline.json")
    args = ap.parse_args()
    rows = load_split(args.processed_dir, "train")
    rb = RuleBaseline().fit(rows)
    rb.save(args.out)
    print(f"Fitted on {len(rows)} rows, {len(rb.lex)} lexicon entries, {len(rb.english)} English words -> {args.out}")


if __name__ == "__main__":
    main()
