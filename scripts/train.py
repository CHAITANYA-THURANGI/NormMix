#!/usr/bin/env python3
"""Train one model.

Usage:
    python scripts/train.py --config experiments/configs/attn_bahdanau.yaml
    python scripts/train.py --config experiments/configs/plain_seq2seq.yaml train.epochs=5 train.run_name=smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import load_config
from src.datasets.pipeline import load_split
from src.datasets.torch_data import make_loader, mix_sources
from src.models.factory import build_model, model_kwargs, save_checkpoint
from src.tokenization.factory import build_tokenizer
from src.training.loop import train_model
from src.utils.io import read_jsonl, write_json
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("override", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))

    # ── Device status ─────────────────────────────────────────────────
    if args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
        print(f"🟢 Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"🔵 Using CPU (no CUDA GPU detected)")
        if torch.cuda.is_available():
            print(f"   GPU available but --device={args.device} was specified")
    print(f"   Device: {args.device}")

    dcfg = cfg["data"]
    train_sources = [(read_jsonl(s["path"]), s["weight"]) for s in dcfg["train_sources"]]
    train_rows = mix_sources(train_sources, cfg.get("seed", 42))
    valid_rows = read_jsonl(dcfg["valid_path"])
    print(f"train={len(train_rows)} valid={len(valid_rows)}")

    src_tok, tgt_tok = build_tokenizer(cfg, [r["src"] for r in train_rows], [r["tgt"] for r in train_rows])
    print(f"src_vocab={src_tok.vocab_size} tgt_vocab={tgt_tok.vocab_size} (kind={src_tok.kind})")

    loader = make_loader(train_rows, src_tok, tgt_tok, cfg["train"]["batch_size"], shuffle=True,
                         max_src_len=dcfg["max_src_len"], max_tgt_len=dcfg["max_tgt_len"], seed=cfg["seed"])
    mk = model_kwargs(cfg["model"])
    model = build_model(mk, src_tok, tgt_tok)
    model._mk = mk  # stashed so the loop can checkpoint architecture alongside weights
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={mk['type']} attention={cfg['model'].get('attention')} params={n_params:,}")

    run_dir = Path(cfg["train"]["out_dir"]) / cfg["train"]["run_name"]
    result = train_model(model, src_tok, tgt_tok, loader, valid_rows, cfg["train"], args.device, run_dir)
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "result.json", {"best_epoch": result["best_epoch"], "best_score": result["best_score"],
                                         "n_params": n_params})
    print(json.dumps({"run_dir": str(run_dir), "best_epoch": result["best_epoch"], "best_score": result["best_score"]}, indent=2))


if __name__ == "__main__":
    main()
