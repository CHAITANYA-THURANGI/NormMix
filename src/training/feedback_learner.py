"""Online feedback learning engine for NormMix AI.
Enables continuous active learning from user corrections on GPU (CUDA)
and re-saves updated checkpoints so future predictions incorporate user feedback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.factory import load_checkpoint, save_checkpoint
from src.tokenization.base import PAD_ID, SOS_ID, EOS_ID
from src.utils.io import read_jsonl

logger = logging.getLogger("NormMix.FeedbackLearner")
ROOT = Path(__file__).resolve().parents[2]


def detect_feedback_target_model(src: str, tgt: str) -> str:
    """Decides whether the feedback sample belongs to translation_sota or production_sota."""
    # If target has no Telugu characters and contains Latin English words, it's translation
    has_te_tgt = any("\u0C00" <= c <= "\u0C7F" for c in tgt)
    has_en_tgt = any(c.isascii() and c.isalpha() for c in tgt)
    if not has_te_tgt and has_en_tgt:
        return "translation_sota"
    return "production_sota"


def online_learn_sample(
    src: str,
    tgt: str,
    device: str | None = None,
    steps: int = 5,
    lr: float = 0.0005,
) -> dict[str, Any]:
    """Runs online few-step gradient descent adaptation on GPU for a user-provided correction."""
    src_clean = src.strip()
    tgt_clean = tgt.strip()
    if not src_clean or not tgt_clean:
        return {"status": "skipped", "reason": "Empty source or target"}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = detect_feedback_target_model(src_clean, tgt_clean)
    ckpt_path = ROOT / "experiments" / "checkpoints" / model_name / "best.pt"

    if not ckpt_path.exists():
        return {"status": "error", "reason": f"Checkpoint {ckpt_path} not found"}

    try:
        model, src_tok, tgt_tok, mk = load_checkpoint(ckpt_path, device=device)
        model.train()
        model.to(device)

        # Encode tokens
        src_ids = [SOS_ID] + src_tok.encode(src_clean) + [EOS_ID]
        tgt_in_ids = [SOS_ID] + tgt_tok.encode(tgt_clean)
        tgt_out_ids = tgt_tok.encode(tgt_clean) + [EOS_ID]

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long)
        tgt_in = torch.tensor([tgt_in_ids], dtype=torch.long, device=device)
        tgt_out = torch.tensor([tgt_out_ids], dtype=torch.long, device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(src_tensor, src_len, tgt_in, teacher_forcing_ratio=1.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Save fine-tuned checkpoint
        model_cfg = mk.get("model_cfg", mk)
        save_checkpoint(
            ckpt_path,
            model,
            model_cfg,
            src_tok,
            tgt_tok,
            meta={"feedback_learned": True, "score": losses[-1]},
            optimizer=optimizer,
            epoch=999,
        )

        logger.info(
            f"Successfully updated {model_name} on {device} from feedback sample. "
            f"Initial loss: {losses[0]:.4f} -> Final loss: {losses[-1]:.4f}"
        )

        return {
            "status": "success",
            "model_updated": model_name,
            "device": device,
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "steps": steps,
            "message": f"Model '{model_name}' successfully learned and updated on {device}.",
        }
    except Exception as e:
        logger.exception("Error during online feedback learning")
        return {"status": "error", "reason": str(e)}


def retrain_from_all_feedback(
    device: str | None = None,
    feedback_file: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregates all user feedback corrections from data/feedback.jsonl and fine-tunes models."""
    fb_path = Path(feedback_file or ROOT / "data" / "feedback.jsonl")
    if not fb_path.exists():
        return {"status": "noop", "message": "No feedback file exists yet."}

    rows = read_jsonl(fb_path)
    corrections = [
        {"src": r.get("input", ""), "tgt": r.get("correction", "")}
        for r in rows
        if r.get("correction") and r.get("input")
    ]

    if not corrections:
        return {"status": "noop", "message": "No user corrections found in feedback log."}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    updated_count = 0
    results = []

    for item in corrections:
        res = online_learn_sample(item["src"], item["tgt"], device=device, steps=3)
        if res.get("status") == "success":
            updated_count += 1
        results.append(res)

    return {
        "status": "success",
        "total_corrections": len(corrections),
        "successfully_learned": updated_count,
        "device": device,
        "details": results,
    }
