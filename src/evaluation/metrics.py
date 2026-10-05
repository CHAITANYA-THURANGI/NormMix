"""Evaluation metrics (dependency-free, so they run anywhere and are unit-tested).

Primary  : CER (character error rate), exact match, WER.
Secondary: chrF, BLEU (simple, smoothed, whitespace tokens), token accuracy.
Task-specific: English-preservation and Telugu-token accuracy (see src/evaluation/evaluate.py).
Not implemented (documented in docs/experiments.md): embedding-based semantic similarity.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def edit_distance(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds: list[str], refs: list[str]) -> float:
    """Corpus-level: total character edits / total reference characters."""
    edits = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    return edits / max(sum(len(r) for r in refs), 1)


def wer(preds: list[str], refs: list[str]) -> float:
    edits = sum(edit_distance(p.split(), r.split()) for p, r in zip(preds, refs))
    return edits / max(sum(len(r.split()) for r in refs), 1)


def exact_match(preds: list[str], refs: list[str]) -> float:
    return sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)


def normalized_edit_distance(preds: list[str], refs: list[str]) -> float:
    """Mean per-sentence edit distance / max(len)."""
    vals = [edit_distance(p, r) / max(len(p), len(r), 1) for p, r in zip(preds, refs)]
    return sum(vals) / max(len(vals), 1)


def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(preds: list[str], refs: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Sentence-averaged chrF (0-100), character n-grams 1..max_n, whitespace ignored."""
    scores = []
    for p, r in zip(preds, refs):
        f_sum, used = 0.0, 0
        for n in range(1, max_n + 1):
            pc, rc = _char_ngrams(p, n), _char_ngrams(r, n)
            if not pc and not rc:
                continue
            overlap = sum((pc & rc).values())
            prec = overlap / max(sum(pc.values()), 1)
            rec = overlap / max(sum(rc.values()), 1)
            f = 0.0 if prec + rec == 0 else (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
            f_sum += f
            used += 1
        scores.append(f_sum / used if used else (1.0 if p == r else 0.0))
    return 100.0 * sum(scores) / max(len(scores), 1)


def bleu(preds: list[str], refs: list[str], max_n: int = 4) -> float:
    """Corpus BLEU (0-100) on whitespace tokens with add-1 smoothing for n>1. Use sacreBLEU for papers."""
    match = [0] * max_n
    total = [0] * max_n
    pl = rl = 0
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        pl += len(pt)
        rl += len(rt)
        for n in range(1, max_n + 1):
            pc = Counter(tuple(pt[i:i + n]) for i in range(len(pt) - n + 1))
            rc = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
            match[n - 1] += sum((pc & rc).values())
            total[n - 1] += max(sum(pc.values()), 0)
    if pl == 0:
        return 0.0
    logs = []
    for n in range(max_n):
        if n == 0:
            if match[0] == 0:
                return 0.0
            logs.append(math.log(match[0] / total[0]))
        else:
            logs.append(math.log((match[n] + 1) / (total[n] + 1)))
    bp = 1.0 if pl > rl else math.exp(1 - rl / max(pl, 1))
    return 100.0 * bp * math.exp(sum(logs) / max_n)


def token_accuracy(preds: list[str], refs: list[str]) -> float:
    """Share of reference tokens reproduced (order-aware via LCS length), a lenient word-level view."""
    hit = tot = 0
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        tot += len(rt)
        prev = [0] * (len(pt) + 1)
        for a in rt:
            cur = [0]
            for j, b in enumerate(pt, 1):
                cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
            prev = cur
        hit += prev[-1]
    return hit / max(tot, 1)


def all_metrics(preds: list[str], refs: list[str]) -> dict:
    return {"n": len(refs), "cer": cer(preds, refs), "wer": wer(preds, refs), "em": exact_match(preds, refs),
            "ned": normalized_edit_distance(preds, refs), "chrf": chrf(preds, refs), "bleu": bleu(preds, refs),
            "token_acc": token_accuracy(preds, refs)}
