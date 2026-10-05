"""Sliced evaluation: overall + mix ratio, source script, noise, length, unseen words, robustness."""
from __future__ import annotations

from ..augmentation.synthetic import composition
from ..preprocessing.text_cleaning import has_telugu, split_edge_punct
from .metrics import all_metrics


def src_script(text: str) -> str:
    lat = any(c.isascii() and c.isalpha() for c in text)
    te = has_telugu(text)
    return "mixed_script" if (lat and te) else "telugu_script" if te else "romanized"


def slice_key_fns():
    return {
        "mix": lambda r: (r.get("comp") or composition(r["tgt"]))["mix"],
        "src_script": lambda r: src_script(r["src"]),
        "kind": lambda r: r.get("kind", "noisy"),
        "length": lambda r: "short(<=4w)" if len(r["tgt"].split()) <= 4 else "long(>=8w)" if len(r["tgt"].split()) >= 8 else "medium",
    }


def token_group_accuracy(rows: list[dict], preds: list[str]) -> dict:
    """Among sentences whose token counts agree, accuracy on English vs Telugu reference tokens."""
    en_hit = en_tot = te_hit = te_tot = 0
    for r, p in zip(rows, preds):
        rt, pt = r["tgt"].split(), p.split()
        if len(rt) != len(pt):
            continue
        for a, b in zip(rt, pt):
            core = split_edge_punct(a)[1]
            if not core:
                continue
            if has_telugu(core):
                te_tot += 1
                te_hit += a == b
            elif any(c.isalpha() for c in core):
                en_tot += 1
                en_hit += a == b
    return {"english_token_acc": en_hit / max(en_tot, 1), "telugu_token_acc": te_hit / max(te_tot, 1),
            "english_tokens": en_tot, "telugu_tokens": te_tot}


def evaluate_rows(rows: list[dict], preds: list[str], slices: bool = True) -> dict:
    refs = [r["tgt"] for r in rows]
    out = {"overall": {**all_metrics(preds, refs), **token_group_accuracy(rows, preds)}}
    if slices and len(rows) >= 2:
        by = {}
        for name, fn in slice_key_fns().items():
            groups: dict[str, list[int]] = {}
            for i, r in enumerate(rows):
                groups.setdefault(fn(r), []).append(i)
            by[name] = {k: all_metrics([preds[i] for i in idx], [refs[i] for i in idx]) for k, idx in sorted(groups.items())}
        out["slices"] = by
    return out


def format_table(results: dict[str, dict], split: str, cols=("em", "cer", "wer", "chrf", "bleu")) -> str:
    """Markdown comparison table {model: {split: {'overall': {...}}}} for one split."""
    head = "| Model | n | " + " | ".join(c.upper() for c in cols) + " |\n|---|---:|" + "---:|" * len(cols) + "\n"
    lines = []
    for model, per_split in results.items():
        o = per_split.get(split, {}).get("overall")
        if not o:
            continue
        vals = [f"{o[c]:.3f}" if c in ("em", "cer", "wer") else f"{o[c]:.1f}" for c in cols]
        lines.append(f"| {model} | {o['n']} | " + " | ".join(vals) + " |")
    return head + "\n".join(lines)
