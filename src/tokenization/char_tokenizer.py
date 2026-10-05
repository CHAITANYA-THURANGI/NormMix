"""Unicode code-point tokenizer. No OOV for seen scripts, tiny vocabulary, ideal for noisy spelling."""
from __future__ import annotations

from collections import Counter

from .base import EOS_ID, PAD_ID, SOS_ID, SPECIALS, UNK_ID, BaseTokenizer


class CharTokenizer(BaseTokenizer):
    kind = "char"

    def __init__(self, itos: list[str] | None = None):
        self.itos = list(itos) if itos else list(SPECIALS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq: int = 1, max_size: int | None = None) -> "CharTokenizer":
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(t)
        chars = [c for c, n in cnt.most_common() if n >= min_freq]
        if max_size:
            chars = chars[: max(0, max_size - len(SPECIALS))]
        return cls(list(SPECIALS) + chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text, add_sos=False, add_eos=False):
        ids = [self.stoi.get(c, UNK_ID) for c in text]
        return ([SOS_ID] if add_sos else []) + ids + ([EOS_ID] if add_eos else [])

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if skip_special and i < len(SPECIALS):
                if i == EOS_ID:
                    break
                continue
            out.append(self.itos[i] if i < len(self.itos) else "")
        return "".join(out)

    def to_dict(self):
        return {"kind": "char", "itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        return cls(d["itos"])
