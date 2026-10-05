"""Batched generation shared by evaluation, the inference engine and the API."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def translate_texts(model, src_tok, tgt_tok, texts: list[str], device="cpu", beam_size: int = 1, n_best: int = 1,
                    batch_size: int = 64, max_len: int = 160, length_penalty: float = 1.0,
                    max_src_len: int = 128, return_attn: bool = False) -> list[list[dict]]:
    """Return, for every input text, a list of hypotheses {text, confidence, score[, attn]} (best first)."""
    model.eval()
    encoded = [src_tok.encode(t)[:max_src_len] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
    results: list[list[dict] | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        lens = torch.tensor([max(len(encoded[i]), 1) for i in idx])
        S = int(lens.max())
        src = torch.full((len(idx), S), src_tok.pad_id, dtype=torch.long)
        for k, i in enumerate(idx):
            if encoded[i]:
                src[k, : len(encoded[i])] = torch.tensor(encoded[i])
        cap = min(max_len, 2 * S + 12)
        hyps = model.generate(src.to(device), lens.to(device), max_len=cap, beam_size=beam_size, n_best=n_best,
                              length_penalty=length_penalty, return_attn=return_attn)
        for k, i in enumerate(idx):
            out = []
            for h in hyps[k]:
                d = {"text": tgt_tok.decode(h["ids"]), "score": h["score"], "confidence": math.exp(max(h["norm_score"], -50.0))}
                if "attn" in h:
                    d["attn"] = h["attn"]
                out.append(d)
            results[i] = out
    return results  # type: ignore[return-value]
