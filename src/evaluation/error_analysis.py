"""Systematic error taxonomy over (source, reference, hypothesis) triples.

Token-level alignment (difflib) between reference and hypothesis; every mismatching token is
assigned ONE category using transparent heuristics (see docs/experiments.md#error-analysis):

  english_spelling          reference English token, hypothesis Latin token within edit distance <= 2
  english_as_telugu         reference English token, hypothesis Telugu script (wrong language handling)
  english_other             reference English token, anything else
  telugu_left_romanized     reference Telugu token, hypothesis still Latin (transliteration failure)
  telugu_char_error         reference Telugu token, hypothesis Telugu within edit distance <= 2 (matra/spelling)
  telugu_vocab              reference Telugu token, unrelated Telugu hypothesis
  insertion / deletion      extra / missing tokens (segmentation or hallucination when unseen in train)
  punctuation_emoji         only edge punctuation/emoji differs
Cross-cutting flags: unknown_word (reference token never seen in training targets), hallucinated
(hypothesis token unseen in training targets and unrelated to the source).
"""
from __future__ import annotations

import difflib
from collections import Counter

from ..preprocessing.text_cleaning import has_telugu, split_edge_punct
from .metrics import edit_distance


def _core(tok: str) -> str:
    return split_edge_punct(tok)[1] or tok


def _is_latin(tok: str) -> bool:
    return any(c.isascii() and c.isalpha() for c in tok) and not has_telugu(tok)


def analyze(rows: list[dict], preds: list[str], train_vocab: set[str] | None = None, max_examples: int = 4) -> dict:
    cats: Counter = Counter()
    flags: Counter = Counter()
    examples: dict[str, list[dict]] = {}
    n_err_sent = 0

    def add(cat, src, ref, hyp, rt="", ht=""):
        cats[cat] += 1
        ex = examples.setdefault(cat, [])
        if len(ex) < max_examples:
            ex.append({"src": src, "ref": ref, "hyp": hyp, "ref_tok": rt, "hyp_tok": ht})

    for row, hyp in zip(rows, preds):
        ref, src = row["tgt"], row["src"]
        if hyp == ref:
            continue
        n_err_sent += 1
        rt, ht = ref.split(), hyp.split()
        src_cores = {_core(t).lower() for t in src.split()}
        sm = difflib.SequenceMatcher(a=rt, b=ht, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            ref_seg, hyp_seg = rt[i1:i2], ht[j1:j2]
            if op == "delete":
                for t in ref_seg:
                    add("deletion", src, ref, hyp, t, "")
                    if train_vocab is not None and _core(t) not in train_vocab:
                        flags["unknown_word"] += 1
                continue
            if op == "insert":
                for t in hyp_seg:
                    add("insertion", src, ref, hyp, "", t)
                    if train_vocab is not None and _core(t) not in train_vocab and _core(t).lower() not in src_cores:
                        flags["hallucinated"] += 1
                continue
            # replace: pair tokens positionally, leftovers become insertion/deletion
            for k in range(max(len(ref_seg), len(hyp_seg))):
                r = ref_seg[k] if k < len(ref_seg) else None
                h = hyp_seg[k] if k < len(hyp_seg) else None
                if r is None:
                    add("insertion", src, ref, hyp, "", h)
                    continue
                if h is None:
                    add("deletion", src, ref, hyp, r, "")
                    continue
                rc, hc = _core(r), _core(h)
                if rc == hc:
                    add("punctuation_emoji", src, ref, hyp, r, h)
                    continue
                d = edit_distance(rc, hc)
                if _is_latin(rc):
                    cat = "english_as_telugu" if has_telugu(hc) else "english_spelling" if (_is_latin(hc) and d <= 2) else "english_other"
                elif has_telugu(rc):
                    cat = "telugu_left_romanized" if _is_latin(hc) else "telugu_char_error" if d <= 2 else "telugu_vocab"
                else:
                    cat = "punctuation_emoji"
                add(cat, src, ref, hyp, r, h)
                if train_vocab is not None:
                    if rc not in train_vocab:
                        flags["unknown_word"] += 1
                    if hc not in train_vocab and hc.lower() not in src_cores:
                        flags["hallucinated"] += 1
    total = sum(cats.values()) or 1
    return {"n_sentences": len(rows), "n_error_sentences": n_err_sent,
            "sentence_error_rate": n_err_sent / max(len(rows), 1),
            "categories": dict(cats.most_common()), "category_share": {k: round(v / total, 3) for k, v in cats.items()},
            "flags": dict(flags), "examples": examples}


def build_train_vocab(train_rows: list[dict]) -> set[str]:
    vocab: set[str] = set()
    for r in train_rows:
        vocab.update(_core(t) for t in r["tgt"].split())
    return vocab
