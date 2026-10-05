"""Leakage-safe, deterministic, growth-stable splitting + duplicate / conflict handling."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .seed_corpus import group_key


def bucket(group: str, salt: str = "tecm-v1") -> int:
    """Map a group id to 0..9999 deterministically (adding data never moves existing groups)."""
    return int(hashlib.sha1(f"{salt}:{group}".encode()).hexdigest()[:8], 16) % 10000


def assign_split(group: str, valid_frac: float = 0.1, test_frac: float = 0.1, salt: str = "tecm-v1") -> str:
    b = bucket(group, salt) / 10000
    if b < test_frac:
        return "test"
    if b < test_frac + valid_frac:
        return "valid"
    return "train"


def contains_word(tgt: str, words: set[str]) -> bool:
    toks = {t.strip(".,?!").lower() for t in tgt.split()}
    return bool(toks & words)


def split_rows(rows: list[dict], valid_frac=0.1, test_frac=0.1, heldout_words=None, salt="tecm-v1",
               exclude_groups: set[str] | None = None) -> dict[str, list[dict]]:
    """Group-level split. Rows sharing `group` (same clean target) always land in the same split.

    `heldout_words`: any group whose target contains one of these English words goes to
    `test_unseen_words` and is removed from train/valid (out-of-vocabulary evaluation).
    `exclude_groups`: groups to drop everywhere (e.g., gold demo set) to prevent leakage.
    """
    heldout = {w.lower() for w in (heldout_words or [])}
    out: dict[str, list[dict]] = {"train": [], "valid": [], "test": [], "test_unseen_words": []}
    for r in rows:
        g = r.get("group") or group_key(r["tgt"])
        r["group"] = g
        if exclude_groups and g in exclude_groups:
            continue
        if heldout and contains_word(r["tgt"], heldout):
            out["test_unseen_words"].append(r)
        else:
            out[assign_split(g, valid_frac, test_frac, salt)].append(r)
    return out


def dedupe_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop exact (src, tgt) duplicates; report conflicts (same src, different tgt)."""
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for r in rows:
        k = (r["src"], r["tgt"])
        if k in seen:
            continue
        seen.add(k)
        kept.append(r)
    by_src: dict[str, set[str]] = defaultdict(set)
    for r in kept:
        by_src[r["src"]].add(r["tgt"])
    conflicts = {s: sorted(t) for s, t in by_src.items() if len(t) > 1}
    return kept, {"removed_duplicates": len(rows) - len(kept), "conflicting_sources": len(conflicts),
                  "conflict_examples": dict(list(conflicts.items())[:5])}


def resolve_conflicts(rows: list[dict], policy: str = "majority") -> list[dict]:
    """Conflicting annotations: keep the majority target per source ('majority') or drop all ('drop')."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        votes[r["src"]][r["tgt"]] += 1
    out = []
    for r in rows:
        c = votes[r["src"]]
        if len(c) == 1:
            out.append(r)
        elif policy == "majority":
            best = c.most_common(1)[0]
            ties = [t for t, n in c.items() if n == best[1]]
            if len(ties) == 1 and r["tgt"] == best[0]:
                out.append(r)
    return out


def filter_noisy(rows: list[dict], min_len: int = 1, max_len: int = 300, max_len_ratio: float = 4.0) -> tuple[list[dict], int]:
    """Reject empty/very long samples and pairs whose lengths are wildly inconsistent."""
    kept = []
    for r in rows:
        s, t = r["src"].strip(), r["tgt"].strip()
        if not (min_len <= len(s) <= max_len and min_len <= len(t) <= max_len):
            continue
        ratio = max(len(s), len(t)) / max(min(len(s), len(t)), 1)
        if ratio > max_len_ratio and max(len(s), len(t)) > 12:
            continue
        kept.append(r)
    return kept, len(rows) - len(kept)


def leakage_report(splits: dict[str, list[dict]]) -> dict:
    """Verify no clean-target group and no source string is shared across train/valid/test*."""
    tr = splits.get("train", [])
    train_groups = {r["group"] for r in tr}
    train_src = {r["src"] for r in tr}
    rep = {}
    for name, rows in splits.items():
        if name == "train":
            continue
        rep[name] = {"shared_groups_with_train": len({r["group"] for r in rows} & train_groups),
                     "shared_sources_with_train": len({r["src"] for r in rows} & train_src)}
    rep["ok"] = all(v["shared_groups_with_train"] == 0 and v["shared_sources_with_train"] == 0
                    for k, v in rep.items() if k != "ok")
    return rep
