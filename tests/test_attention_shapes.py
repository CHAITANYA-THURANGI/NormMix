import torch

from src.attention.attention import make_attention


def _run(kind):
    q_dim, k_dim, attn_dim, B, S = 12, 20, 8, 3, 5
    attn = make_attention(kind, q_dim, k_dim, attn_dim)
    q = torch.randn(B, q_dim)
    k = torch.randn(B, S, k_dim)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    ctx, w = attn(q, k, mask)
    assert w.shape == (B, S)
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-4)
    assert torch.all(w[mask.logical_not()] < 1e-6)
    return ctx.shape


def test_all_attention_kinds_shapes():
    for kind in ["bahdanau", "luong_dot", "luong_general", "luong_concat", "scaled_dot"]:
        shape = _run(kind)
        assert shape[0] == 3
