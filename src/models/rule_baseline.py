"""Baseline 1: rule + dictionary + edit-distance normalizer (no neural network).

Learns from aligned training pairs (equal token counts): romanized/noisy token -> most frequent
normalized token. At inference: exact lookup -> known-English passthrough -> Telugu-suffix split
(collegeki -> college కి) -> fuzzy match (difflib ratio) -> passthrough. It is intentionally simple:
it defines the floor that the neural models must beat, and it doubles as a safe fallback in the API.
"""
from __future__ import annotations

import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..preprocessing.text_cleaning import clean_text, has_telugu, split_edge_punct

SUFFIX_MAP = {"ki": "కి", "ku": "కు", "lo": "లో", "tho": "తో", "to": "తో", "nunchi": "నుంచి", "nundi": "నుంచి",
              "ni": "ని", "ga": "గా", "gaa": "గా"}


class RuleBaseline:
    def __init__(self, fuzzy_cutoff: float = 0.78):
        self.lex: dict[str, str] = {}
        self.english: set[str] = set()
        self.fuzzy_cutoff = fuzzy_cutoff
        self._keys_by_initial: dict[str, list[str]] = {}

    def fit(self, rows: list[dict]) -> "RuleBaseline":
        counts: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            st, tt = clean_text(r["src"]).split(), clean_text(r["tgt"], 0).split()
            for t in tt:
                core = split_edge_punct(t)[1]
                if core and core.isascii() and core.isalpha():
                    self.english.add(core.lower())
            if len(st) != len(tt):
                continue
            for s, t in zip(st, tt):
                sc, tc = split_edge_punct(s)[1], split_edge_punct(t)[1]
                if sc and tc:
                    counts[sc.lower()][tc] += 1
        self.lex = {k: c.most_common(1)[0][0] for k, c in counts.items()}
        self._index()
        return self

    def _index(self):
        self._keys_by_initial = defaultdict(list)
        for k in self.lex:
            self._keys_by_initial[k[:1]].append(k)

    def normalize_token(self, tok: str) -> str:
        pre, core, post = split_edge_punct(tok)
        if not core or has_telugu(core) or not core.isascii():
            return tok
        low = core.lower()
        if low in self.lex:
            new = self.lex[low]
        elif low in self.english:
            new = low
        else:
            new = None
            for suf, tel in SUFFIX_MAP.items():
                if low.endswith(suf) and len(low) > len(suf) + 2:
                    stem = low[: -len(suf)]
                    base = self.lex.get(stem) or (stem if stem in self.english else None)
                    if base is None:
                        m = difflib.get_close_matches(stem, [k for k in self._keys_by_initial.get(stem[:1], [])], 1, self.fuzzy_cutoff)
                        base = self.lex.get(m[0]) if m else None
                    if base:
                        new = f"{base} {tel}"
                        break
            if new is None:
                cands = self._keys_by_initial.get(low[:1], [])
                m = difflib.get_close_matches(low, cands, 1, self.fuzzy_cutoff)
                new = self.lex[m[0]] if m else core
        return pre + new + post

    def normalize(self, text: str) -> str:
        return " ".join(self.normalize_token(t) for t in clean_text(text).split())

    def normalize_many(self, texts: list[str]) -> list[str]:
        return [self.normalize(t) for t in texts]

    def to_dict(self) -> dict:
        return {"lex": self.lex, "english": sorted(self.english), "fuzzy_cutoff": self.fuzzy_cutoff}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "RuleBaseline":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(d.get("fuzzy_cutoff", 0.78))
        obj.lex, obj.english = d["lex"], set(d["english"])
        obj._index()
        return obj
