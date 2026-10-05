from __future__ import annotations

from pathlib import Path

import torch

from ..tokenization.factory import tokenizer_from_dict
from .rnn_seq2seq import RNNSeq2Seq
from .transformer import TransformerSeq2Seq


def model_kwargs(cfg_model: dict) -> dict:
    """The subset of config that fully determines the architecture (stored inside checkpoints)."""
    keys_rnn = ["rnn_type", "attention", "emb_dim", "hid_dim", "enc_layers", "dec_layers", "dropout", "attn_dim", "use_copy"]
    keys_tf = ["d_model", "nhead", "ff_dim", "tf_enc_layers", "tf_dec_layers", "dropout"]
    ks = keys_tf if cfg_model["type"] == "transformer" else keys_rnn
    return {"type": cfg_model["type"], **{k: cfg_model[k] for k in ks if k in cfg_model}}


def build_model(mk: dict, src_tok, tgt_tok):
    special = dict(pad_id=src_tok.pad_id, sos_id=tgt_tok.sos_id, eos_id=tgt_tok.eos_id)
    if mk["type"] == "rnn":
        use_copy = mk.get("use_copy", mk.get("attention", "none") != "none")
        src_map = None
        if use_copy:
            src_map = torch.full((src_tok.vocab_size,), tgt_tok.unk_id, dtype=torch.long)
            for sid in range(src_tok.vocab_size):
                ch = getattr(src_tok, "id2char", {}).get(sid)
                if ch is not None and hasattr(tgt_tok, "char2id") and ch in tgt_tok.char2id:
                    src_map[sid] = tgt_tok.char2id[ch]
        return RNNSeq2Seq(src_tok.vocab_size, tgt_tok.vocab_size, mk["emb_dim"], mk["hid_dim"], mk["enc_layers"],
                          mk["dec_layers"], mk["rnn_type"], mk["attention"], mk["dropout"], mk["attn_dim"],
                          use_copy=use_copy, src_to_tgt_map=src_map, **special)
    if mk["type"] == "transformer":
        return TransformerSeq2Seq(src_tok.vocab_size, tgt_tok.vocab_size, mk["d_model"], mk["nhead"], mk["tf_enc_layers"],
                                  mk["tf_dec_layers"], mk["ff_dim"], mk["dropout"], **special)
    raise ValueError(f"Unknown model type {mk['type']}")


def save_checkpoint(path, model, mk, src_tok, tgt_tok, meta=None, optimizer=None, epoch=0):
    torch.save({"state_dict": model.state_dict(), "model_cfg": mk, "src_tok": src_tok.to_dict(),
                "tgt_tok": tgt_tok.to_dict(), "meta": meta or {}, "epoch": epoch,
                "optimizer": optimizer.state_dict() if optimizer is not None else None}, path)


def load_checkpoint(path, device="cpu"):
    p = Path(path)
    if p.is_dir():
        p = p / "best.pt"
    ck = torch.load(p, map_location=device, weights_only=False)
    src_tok, tgt_tok = tokenizer_from_dict(ck["src_tok"]), tokenizer_from_dict(ck["tgt_tok"])
    model = build_model(ck["model_cfg"], src_tok, tgt_tok)
    model.load_state_dict(ck["state_dict"], strict=False)
    return model.to(device).eval(), src_tok, tgt_tok, ck
