#!/usr/bin/env python3
"""Evaluate one or more checkpoints (or the rule baseline) on one or more splits.

Usage:
    python scripts/evaluate.py --checkpoint experiments/checkpoints/attn_bahdanau/best.pt --split test
    python scripts/evaluate.py --checkpoint runA/best.pt runB/best.pt --rule-baseline data/processed/rule_baseline.json \
        --split test test_unseen_words --out experiments/results/compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.datasets.pipeline import load_split
from src.evaluation.error_analysis import analyze, build_train_vocab
from src.evaluation.evaluate import evaluate_rows, format_table
from src.inference.generate import translate_texts
from src.models.factory import load_checkpoint
from src.models.rule_baseline import RuleBaseline
from src.utils.io import write_json


def evaluate_one(name, predict_fn, splits, processed_dir, errors=False, train_vocab=None):
    per_split = {}
    for split in splits:
        rows = load_split(processed_dir, split)
        preds = predict_fn([r["src"] for r in rows])
        res = evaluate_rows(rows, preds)
        if errors:
            res["errors"] = analyze(rows, preds, train_vocab)
        per_split[split] = res
        o = res["overall"]
        print(f"[{name}][{split}] n={o['n']} EM={o['em']:.3f} CER={o['cer']:.3f} WER={o['wer']:.3f} "
             f"chrF={o['chrf']:.1f} BLEU={o['bleu']:.1f} EN_acc={o['english_token_acc']:.3f} TE_acc={o['telugu_token_acc']:.3f}")
    return per_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--rule-baseline", default=None)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--split", nargs="*", default=["test"])
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"🟢 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("🔵 Using CPU")

    train_vocab = None
    if args.errors:
        train_vocab = build_train_vocab(load_split(args.processed_dir, "train"))

    results = {}
    for ckpt in args.checkpoint:
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best.pt"
        model, src_tok, tgt_tok, meta = load_checkpoint(ckpt_path, device)
        name = ckpt_path.parent.name

        def predict_fn(texts, model=model, src_tok=src_tok, tgt_tok=tgt_tok):
            hyps = translate_texts(model, src_tok, tgt_tok, texts, device, beam_size=args.beam_size)
            return [h[0]["text"] for h in hyps]

        results[name] = evaluate_one(name, predict_fn, args.split, args.processed_dir, args.errors, train_vocab)

    if args.rule_baseline:
        rb = RuleBaseline.load(args.rule_baseline)
        results["rule_baseline"] = evaluate_one("rule_baseline", rb.normalize_many, args.split, args.processed_dir,
                                                args.errors, train_vocab)

    if len(results) > 1:
        for split in args.split:
            print(f"\n=== {split} ===")
            print(format_table(results, split))

    if args.out:
        write_json(args.out, results)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
