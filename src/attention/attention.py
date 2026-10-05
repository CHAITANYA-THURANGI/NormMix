"""Attention mechanisms (all return `(context, weights)`; padding positions are masked out).

Notation: query q_t (decoder state), keys/values h_1..h_S (encoder outputs).
  Bahdanau (additive):  e_ti = v^T tanh(W_q q_t + W_k h_i)
  Luong dot:            e_ti = q_t^T (W h_i)            (W only aligns dimensions)
  Luong general:        e_ti = q_t^T W h_i
  Luong concat:         e_ti = v^T tanh(W [q_t ; h_i])
  Scaled dot-product:   e_ti = (W_q q_t)^T (W_k h_i) / sqrt(d)      values = W_v h_i
  a_ti = softmax_i(e_ti);   c_t = sum_i a_ti * value_i
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _softmax_ctx(scores, values, mask):
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg_inf)
    w = F.softmax(scores, dim=-1)
    ctx = torch.bmm(w.unsqueeze(1), values).squeeze(1)
    return ctx, w


class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(key_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def prepare(self, keys):
        return self.W_k(keys)

    def forward(self, query, keys, mask, prep=None):
        pk = prep if prep is not None else self.W_k(keys)
        e = self.v(torch.tanh(pk + self.W_q(query).unsqueeze(1))).squeeze(-1)
        return _softmax_ctx(e, keys, mask)


class LuongAttention(nn.Module):
    def __init__(self, query_dim, key_dim, mode="general", attn_dim=None):
        super().__init__()
        assert mode in ("dot", "general", "concat")
        self.mode = mode
        if mode in ("dot", "general"):
            # 'dot' needs equal dims: a bias-free projection aligns the (bidirectional) key size.
            self.W = nn.Linear(key_dim, query_dim, bias=(mode == "general"))
        else:
            a = attn_dim or query_dim
            self.W = nn.Linear(query_dim + key_dim, a)
            self.v = nn.Linear(a, 1, bias=False)

    def prepare(self, keys):
        return self.W(keys) if self.mode in ("dot", "general") else None

    def forward(self, query, keys, mask, prep=None):
        if self.mode == "concat":
            q = query.unsqueeze(1).expand(-1, keys.size(1), -1)
            e = self.v(torch.tanh(self.W(torch.cat([q, keys], -1)))).squeeze(-1)
        else:
            pk = prep if prep is not None else self.W(keys)
            e = torch.bmm(pk, query.unsqueeze(-1)).squeeze(-1)
        return _softmax_ctx(e, keys, mask)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(attn_dim)
        self.W_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(key_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(key_dim, key_dim, bias=False)

    def prepare(self, keys):
        return self.W_k(keys), self.W_v(keys)

    def forward(self, query, keys, mask, prep=None):
        k, v = prep if prep is not None else self.prepare(keys)
        e = torch.bmm(k, self.W_q(query).unsqueeze(-1)).squeeze(-1) * self.scale
        return _softmax_ctx(e, v, mask)


def make_attention(kind: str, query_dim: int, key_dim: int, attn_dim: int) -> nn.Module:
    if kind == "bahdanau":
        return BahdanauAttention(query_dim, key_dim, attn_dim)
    if kind.startswith("luong_"):
        return LuongAttention(query_dim, key_dim, kind.split("_", 1)[1], attn_dim)
    if kind == "scaled_dot":
        return ScaledDotProductAttention(query_dim, key_dim, attn_dim)
    raise ValueError(f"Unknown attention type: {kind}")
