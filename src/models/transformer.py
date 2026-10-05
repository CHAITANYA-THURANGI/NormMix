"""Character-level Transformer encoder-decoder (comparison model; pre-LN, sinusoidal positions)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .decoding import beam_search


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, nhead=4, enc_layers=3, dec_layers=3, ff_dim=384,
                 dropout=0.1, pad_id=0, sos_id=1, eos_id=2, max_len=1024):
        super().__init__()
        self.d_model, self.pad_id, self.sos_id, self.eos_id = d_model, pad_id, sos_id, eos_id
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        el = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        dl = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, enc_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dl, dec_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_len):
        kpm = torch.arange(src.size(1), device=src.device)[None, :] >= src_len[:, None].to(src.device)
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_key_padding_mask=kpm), kpm

    def decode(self, tgt_in, memory, kpm):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        y = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        return self.out(self.decoder(y, memory, tgt_mask=causal, memory_key_padding_mask=kpm))

    def forward(self, src, src_len, tgt_in, teacher_forcing_ratio=1.0, return_attn=False):
        memory, kpm = self.encode(src, src_len)
        logits = self.decode(tgt_in, memory, kpm)
        return (logits, None) if return_attn else logits

    @torch.no_grad()
    def generate(self, src, src_len, max_len=160, beam_size=1, n_best=1, length_penalty=1.0, return_attn=False):
        was_training = self.training
        self.eval()
        try:
            memory, kpm = self.encode(src, src_len)
            B = src.size(0)
            if beam_size <= 1:
                ys = torch.full((B, 1), self.sos_id, dtype=torch.long, device=src.device)
                done = torch.zeros(B, dtype=torch.bool, device=src.device)
                lp_sum = torch.zeros(B, device=src.device)
                for _ in range(max_len):
                    lp = torch.log_softmax(self.decode(ys, memory, kpm)[:, -1], -1)
                    best, nxt = lp.max(-1)
                    lp_sum += torch.where(done, torch.zeros_like(best), best)
                    nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
                    ys = torch.cat([ys, nxt[:, None]], 1)
                    done = done | (nxt == self.eos_id)
                    if bool(done.all()):
                        break
                res = []
                for b in range(B):
                    toks = []
                    for t in ys[b, 1:].tolist():
                        if t in (self.eos_id, self.pad_id):
                            break
                        toks.append(t)
                    res.append([{"ids": toks, "score": lp_sum[b].item(), "norm_score": lp_sum[b].item() / (len(toks) + 1)}])
                return res
            results = []
            for b in range(B):
                mem, kp = memory[b:b + 1, : int(src_len[b])], kpm[b:b + 1, : int(src_len[b])]

                def step_fn(prev, prefix, mem=mem, kp=kp):
                    k = prev.size(0)
                    prefix = torch.cat([prefix, prev[:, None]], 1)
                    lg = self.decode(prefix, mem.expand(k, -1, -1), kp.expand(k, -1))[:, -1]
                    return torch.log_softmax(lg, -1), prefix

                init = torch.zeros((1, 0), dtype=torch.long, device=src.device)
                results.append(beam_search(step_fn, init, lambda st, idx: st.index_select(0, idx), self.sos_id,
                                           self.eos_id, beam_size, max_len, length_penalty, n_best, device=src.device))
            return results
        finally:
            self.train(was_training)
