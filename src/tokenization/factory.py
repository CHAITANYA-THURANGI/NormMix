from __future__ import annotations

from .char_tokenizer import CharTokenizer
from .spm_tokenizer import SPMTokenizer


def build_tokenizer(cfg: dict, src_texts, tgt_texts):
    """Returns (src_tokenizer, tgt_tokenizer). Char: separate vocabs. SPM: separate models."""
    t = cfg["tokenizer"]
    kind = t.get("type", "char")
    if kind == "char":
        return (CharTokenizer.build(src_texts, t.get("min_freq", 1)), CharTokenizer.build(tgt_texts, t.get("min_freq", 1)))
    if kind in ("spm", "unigram", "bpe"):
        mt = "bpe" if kind == "bpe" else t.get("model_type", "unigram")
        vs = t.get("vocab_size", 500)
        return SPMTokenizer.train(src_texts, vs, mt), SPMTokenizer.train(tgt_texts, vs, mt)
    raise ValueError(f"Unknown tokenizer type: {kind}")


def tokenizer_from_dict(d: dict):
    if d["kind"] == "char":
        return CharTokenizer.from_dict(d)
    if d["kind"] == "spm":
        return SPMTokenizer.from_dict(d)
    raise ValueError(d["kind"])
