"""Lightweight token-level language identification for Telugu-English code-mixed text.

Labels follow the CMTET-LID convention (TE, EN, NE, UNIV) minus NE:
  TE   Telugu word (Telugu script, or Romanized Telugu recognised by lexicon / char n-grams)
  EN   English word
  UNIV punctuation, numbers, emoji, hashtags/mentions/URLs
Script-based rules are exact; Latin-script tokens use lexicons first and a char n-gram Naive
Bayes fallback trained on your own data. This is deliberately a separate, replaceable module.
"""
from __future__ import annotations

import math
from collections import Counter

from .text_cleaning import has_telugu, is_emoji_char, split_edge_punct


def _ngrams(word: str, ns=(1, 2, 3)) -> list[str]:
    w = f"^{word.lower()}$"
    return [w[i:i + n] for n in ns for i in range(len(w) - n + 1)]


class CharNgramLID:
    """Binary Naive Bayes over char n-grams: 'EN' vs 'TE' (Romanized Telugu)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.counts: dict[str, Counter] = {"EN": Counter(), "TE": Counter()}
        self.totals: dict[str, int] = {"EN": 0, "TE": 0}
        self.vocab: set[str] = set()
        self.priors: dict[str, float] = {"EN": 0.5, "TE": 0.5}

    def fit(self, en_words, te_roman_words) -> "CharNgramLID":
        for label, words in (("EN", en_words), ("TE", te_roman_words)):
            for w in words:
                grams = _ngrams(w)
                self.counts[label].update(grams)
                self.totals[label] += len(grams)
                self.vocab.update(grams)
        n_en, n_te = self.totals["EN"] or 1, self.totals["TE"] or 1
        z = n_en + n_te
        self.priors = {"EN": n_en / z, "TE": n_te / z}
        return self

    @property
    def fitted(self) -> bool:
        return self.totals["EN"] > 0 and self.totals["TE"] > 0

    def predict_proba(self, word: str) -> dict[str, float]:
        v = max(len(self.vocab), 1)
        scores = {}
        for label in ("EN", "TE"):
            s = math.log(self.priors[label])
            denom = self.totals[label] + self.alpha * v
            for g in _ngrams(word):
                s += math.log((self.counts[label][g] + self.alpha) / denom)
            scores[label] = s
        m = max(scores.values())
        exp = {k: math.exp(v_ - m) for k, v_ in scores.items()}
        z = sum(exp.values())
        return {k: v_ / z for k, v_ in exp.items()}

    def predict(self, word: str) -> tuple[str, float]:
        p = self.predict_proba(word)
        label = max(p, key=p.get)
        return label, p[label]

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "counts": {k: dict(v) for k, v in self.counts.items()},
                "totals": self.totals, "priors": self.priors}

    @classmethod
    def from_dict(cls, d: dict) -> "CharNgramLID":
        obj = cls(d.get("alpha", 0.5))
        obj.counts = {k: Counter(v) for k, v in d["counts"].items()}
        obj.totals = d["totals"]
        obj.priors = d["priors"]
        obj.vocab = set().union(*[set(c) for c in obj.counts.values()])
        return obj


class TokenLID:
    def __init__(self, en_lexicon=None, te_roman_lexicon=None, nb: CharNgramLID | None = None):
        self.en = {w.lower() for w in (en_lexicon or [])}
        self.te = {w.lower() for w in (te_roman_lexicon or [])}
        self.nb = nb

    def tag_token(self, token: str) -> str:
        pre, core, post = split_edge_punct(token)
        if not core:
            return "UNIV"
        if core[0] in "#@" or core.startswith(("http", "www.")):
            return "UNIV"
        if has_telugu(core):
            return "TE"
        if all(is_emoji_char(c) for c in core) or core.replace(".", "").replace(",", "").isdigit():
            return "UNIV"
        low = core.lower()
        in_en, in_te = low in self.en, low in self.te
        if in_en and not in_te:
            return "EN"
        if in_te and not in_en:
            return "TE"
        if self.nb is not None and self.nb.fitted:
            label, _ = self.nb.predict(low)
            return label
        return "EN" if in_en else "TE" if in_te else "UNK"

    def tag(self, text: str) -> list[tuple[str, str]]:
        return [(t, self.tag_token(t)) for t in text.split()]
