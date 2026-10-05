"""Recurrent encoder-decoder with optional attention.

Encoder : embedding -> (Bi)GRU/LSTM -> outputs h_1..h_S (dim 2H) + bridge to the decoder state.
Decoder : GRU/LSTM cells, teacher forcing (optionally scheduled sampling), autoregressive decoding.
Attention (config `model.attention`):
  none            plain Seq2Seq baseline (information bottleneck: only the bridged final state)
  bahdanau        context c_t from previous state s_{t-1} *before* the RNN step; out=tanh(W[s_t;c_t;e_t])
  luong_dot|general|concat, scaled_dot
                  context from the *new* state s_t; input feeding of h~_{t-1}; h~_t=tanh(W[c_t;s_t])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..attention.attention import make_attention
from .decoding import beam_search


def _reorder(state: dict, idx: torch.Tensor) -> dict:
    return {"h": state["h"].index_select(1, idx),
            "c": state["c"].index_select(1, idx) if state["c"] is not None else None,
            "ht": state["ht"].index_select(0, idx),
            "src_mapped": state["src_mapped"].index_select(0, idx) if state.get("src_mapped") is not None else None}


class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=96, hid_dim=192, enc_layers=1, dec_layers=1, rnn_type="gru",
                 attention="bahdanau", dropout=0.2, attn_dim=96, pad_id=0, sos_id=1, eos_id=2,
                 use_copy=False, src_to_tgt_map=None):
        super().__init__()
        self.rnn_type, self.attention_kind = rnn_type.lower(), attention
        self.pad_id, self.sos_id, self.eos_id = pad_id, sos_id, eos_id
        self.hid_dim, self.dec_layers = hid_dim, dec_layers
        self.lstm = self.rnn_type == "lstm"
        self.use_attn = attention != "none"
        self.bahdanau = attention == "bahdanau"
        self.use_copy = bool(use_copy and self.use_attn)
        enc_dim = 2 * hid_dim

        self.src_emb = nn.Embedding(src_vocab, emb_dim, padding_idx=pad_id)
        rnn = nn.LSTM if self.lstm else nn.GRU
        self.encoder = rnn(emb_dim, hid_dim, num_layers=enc_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if enc_layers > 1 else 0.0)
        self.bridge_h = nn.Linear(enc_dim, hid_dim)
        self.bridge_c = nn.Linear(enc_dim, hid_dim) if self.lstm else None

        self.tgt_emb = nn.Embedding(tgt_vocab, emb_dim, padding_idx=pad_id)
        in0 = emb_dim if not self.use_attn else (emb_dim + enc_dim if self.bahdanau else emb_dim + hid_dim)
        cell = nn.LSTMCell if self.lstm else nn.GRUCell
        self.cells = nn.ModuleList([cell(in0 if l == 0 else hid_dim, hid_dim) for l in range(dec_layers)])
        if self.use_attn:
            self.attn = make_attention(attention, hid_dim, enc_dim, attn_dim)
            self.combine = nn.Linear(hid_dim + enc_dim + emb_dim, hid_dim) if self.bahdanau else nn.Linear(enc_dim + hid_dim, hid_dim)
        if self.use_copy:
            self.p_gen_layer = nn.Linear(hid_dim + emb_dim, 1)
            if src_to_tgt_map is None:
                m = torch.arange(src_vocab)
                m[m >= tgt_vocab] = pad_id
                self.register_buffer("src_to_tgt_map", m)
            else:
                self.register_buffer("src_to_tgt_map", src_to_tgt_map)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, tgt_vocab)

    # ---- encoder -------------------------------------------------------------------------
    def encode(self, src, src_len):
        S = src.size(1)
        mask = torch.arange(S, device=src.device)[None, :] < src_len[:, None].to(src.device)
        emb = self.drop(self.src_emb(src))
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, hid = self.encoder(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=S)
        h, c = hid if self.lstm else (hid, None)
        h0 = torch.tanh(self.bridge_h(torch.cat([h[-2], h[-1]], -1)))
        src_mapped = self.src_to_tgt_map[src] if self.use_copy else None
        state = {"h": h0.unsqueeze(0).repeat(self.dec_layers, 1, 1),
                 "c": torch.tanh(self.bridge_c(torch.cat([c[-2], c[-1]], -1))).unsqueeze(0).repeat(self.dec_layers, 1, 1) if self.lstm else None,
                 "ht": torch.zeros_like(h0),
                 "src_mapped": src_mapped}
        return out, mask, state

    # ---- one decoder step ----------------------------------------------------------------
    def decode_step(self, prev, state, enc_out, mask, prep):
        e = self.drop(self.tgt_emb(prev))
        h, c, ht = state["h"], state["c"], state["ht"]
        src_mapped = state.get("src_mapped")
        ctx = w = None
        if self.use_attn and self.bahdanau:
            ctx, w = self.attn(h[-1], enc_out, mask, prep)
            inp = torch.cat([e, ctx], -1)
        elif self.use_attn:
            inp = torch.cat([e, ht], -1)
        else:
            inp = e
        new_h, new_c = [], []
        for l, cell in enumerate(self.cells):
            if self.lstm:
                hl, cl = cell(inp, (h[l], c[l]))
                new_c.append(cl)
            else:
                hl = cell(inp, h[l])
            new_h.append(hl)
            inp = self.drop(hl) if l < len(self.cells) - 1 else hl
        s = new_h[-1]
        if not self.use_attn:
            feat = s
        elif self.bahdanau:
            feat = torch.tanh(self.combine(torch.cat([s, ctx, e], -1)))
        else:
            ctx, w = self.attn(s, enc_out, mask, prep)
            feat = torch.tanh(self.combine(torch.cat([ctx, s], -1)))
        logits = self.out(self.drop(feat))
        if self.use_copy and w is not None and src_mapped is not None:
            p_gen = torch.sigmoid(self.p_gen_layer(torch.cat([feat, e], -1)))
            vocab_probs = F.softmax(logits, dim=-1)
            p_final = p_gen * vocab_probs
            p_final.scatter_add_(1, src_mapped, (1.0 - p_gen) * w)
            logits = torch.log(p_final.clamp(min=1e-12))
        return logits, {"h": torch.stack(new_h), "c": torch.stack(new_c) if self.lstm else None, "ht": feat, "src_mapped": src_mapped}, w

    # ---- training forward ----------------------------------------------------------------
    def forward(self, src, src_len, tgt_in, teacher_forcing_ratio=1.0, return_attn=False):
        enc_out, mask, state = self.encode(src, src_len)
        prep = self.attn.prepare(enc_out) if self.use_attn else None
        B, T = tgt_in.shape
        prev, logits, attns = tgt_in[:, 0], [], []
        for t in range(T):
            lg, state, w = self.decode_step(prev, state, enc_out, mask, prep)
            logits.append(lg)
            if return_attn and w is not None:
                attns.append(w)
            if t + 1 < T:
                if teacher_forcing_ratio >= 1.0:
                    prev = tgt_in[:, t + 1]
                else:
                    use_tf = torch.rand(B, device=src.device) < teacher_forcing_ratio
                    prev = torch.where(use_tf, tgt_in[:, t + 1], lg.argmax(-1))
        logits = torch.stack(logits, 1)
        if return_attn:
            return logits, (torch.stack(attns, 1) if attns else None)
        return logits

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, src, src_len, max_len=160, beam_size=1, n_best=1, length_penalty=1.0, return_attn=False):
        was_training = self.training
        self.eval()
        try:
            if beam_size <= 1:
                return self._greedy(src, src_len, max_len, return_attn)
            results = []
            for b in range(src.size(0)):
                L = int(src_len[b])
                enc_out, mask, state = self.encode(src[b:b + 1, :L], src_len[b:b + 1])
                prep = self.attn.prepare(enc_out) if self.use_attn else None

                def step_fn(prev, st, enc_out=enc_out, mask=mask, prep=prep):
                    k = prev.size(0)
                    p = None if prep is None else (tuple(x.expand(k, -1, -1) for x in prep) if isinstance(prep, tuple) else prep.expand(k, -1, -1))
                    lg, ns, _ = self.decode_step(prev, st, enc_out.expand(k, -1, -1), mask.expand(k, -1), p)
                    return torch.log_softmax(lg, -1), ns

                hyps = beam_search(step_fn, state, _reorder, self.sos_id, self.eos_id, beam_size, max_len,
                                   length_penalty, n_best, device=src.device)
                results.append(hyps)
            return results
        finally:
            self.train(was_training)

    def _greedy(self, src, src_len, max_len, return_attn):
        enc_out, mask, state = self.encode(src, src_len)
        prep = self.attn.prepare(enc_out) if self.use_attn else None
        B = src.size(0)
        prev = torch.full((B,), self.sos_id, dtype=torch.long, device=src.device)
        ids = [[] for _ in range(B)]
        logp_sum = torch.zeros(B, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        attn_hist = [[] for _ in range(B)]
        for _ in range(max_len):
            lg, state, w = self.decode_step(prev, state, enc_out, mask, prep)
            lp = torch.log_softmax(lg, -1)
            best_lp, prev = lp.max(-1)
            for b in range(B):
                if not done[b]:
                    logp_sum[b] += best_lp[b]
                    if prev[b].item() == self.eos_id:
                        done[b] = True
                    else:
                        ids[b].append(prev[b].item())
                        if return_attn and w is not None:
                            attn_hist[b].append(w[b, : int(src_len[b])].cpu())
            if bool(done.all()):
                break
        out = []
        for b in range(B):
            n = len(ids[b]) + 1
            hyp = {"ids": ids[b], "score": logp_sum[b].item(), "norm_score": logp_sum[b].item() / n}
            if return_attn and attn_hist[b]:
                hyp["attn"] = torch.stack(attn_hist[b])
            out.append([hyp])
        return out
