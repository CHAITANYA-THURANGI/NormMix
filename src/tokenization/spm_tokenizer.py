"""SentencePiece (unigram / BPE) wrapper with the same interface. Optional dependency."""
from __future__ import annotations

import base64
import os
import tempfile

from .base import EOS_ID, PAD_ID, SOS_ID, UNK_ID, BaseTokenizer


class SPMTokenizer(BaseTokenizer):
    kind = "spm"

    def __init__(self, model_bytes: bytes):
        import sentencepiece as spm

        self._bytes = model_bytes
        self.sp = spm.SentencePieceProcessor()
        self.sp.load_from_serialized_proto(model_bytes)

    @classmethod
    def train(cls, texts, vocab_size: int = 500, model_type: str = "unigram") -> "SPMTokenizer":
        import sentencepiece as spm

        with tempfile.TemporaryDirectory() as td:
            corpus = os.path.join(td, "corpus.txt")
            with open(corpus, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t.replace("\n", " ") + "\n")
            prefix = os.path.join(td, "spm")
            spm.SentencePieceTrainer.train(
                input=corpus, model_prefix=prefix, vocab_size=vocab_size, model_type=model_type,
                character_coverage=1.0, pad_id=PAD_ID, bos_id=SOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
                hard_vocab_limit=False, split_by_whitespace=True, byte_fallback=False, minloglevel=2)
            with open(prefix + ".model", "rb") as f:
                return cls(f.read())

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text, add_sos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        return ([SOS_ID] if add_sos else []) + ids + ([EOS_ID] if add_eos else [])

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if skip_special and i in (PAD_ID, SOS_ID, UNK_ID):
                continue
            out.append(i)
        return self.sp.decode(out)

    def to_dict(self):
        return {"kind": "spm", "model_b64": base64.b64encode(self._bytes).decode("ascii")}

    @classmethod
    def from_dict(cls, d):
        return cls(base64.b64decode(d["model_b64"]))
