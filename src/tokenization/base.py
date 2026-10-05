"""Tokenizer interface. Special ids are fixed across all tokenizers: pad=0, sos=1, eos=2, unk=3."""
from __future__ import annotations

PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class BaseTokenizer:
    kind = "base"
    pad_id, sos_id, eos_id, unk_id = PAD_ID, SOS_ID, EOS_ID, UNK_ID

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids, skip_special: bool = True) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def unk_rate(self, texts) -> float:
        total = unk = 0
        for t in texts:
            ids = self.encode(t)
            total += len(ids)
            unk += sum(1 for i in ids if i == UNK_ID)
        return unk / max(total, 1)
