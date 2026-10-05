import torch

from src.models.rnn_seq2seq import RNNSeq2Seq
from src.models.transformer import TransformerSeq2Seq


def _batch(vocab=15, B=2, S=6, T=5):
    src = torch.randint(4, vocab, (B, S))
    src_len = torch.tensor([S, S - 2])
    tin = torch.randint(4, vocab, (B, T))
    return src, src_len, tin, vocab


def test_rnn_forward_and_generate_all_attentions():
    src, src_len, tin, vocab = _batch()
    for att in ["none", "bahdanau", "luong_dot", "luong_general", "luong_concat", "scaled_dot"]:
        m = RNNSeq2Seq(vocab, vocab, emb_dim=8, hid_dim=10, attention=att)
        logits = m(src, src_len, tin)
        assert logits.shape == (2, 5, vocab)
        hyps = m.generate(src, src_len, max_len=6)
        assert len(hyps) == 2 and len(hyps[0]) == 1
        beam = m.generate(src, src_len, max_len=6, beam_size=3, n_best=2)
        assert len(beam[0]) <= 2


def test_transformer_forward_and_generate():
    src, src_len, tin, vocab = _batch()
    m = TransformerSeq2Seq(vocab, vocab, d_model=16, nhead=2, enc_layers=1, dec_layers=1, ff_dim=32)
    logits = m(src, src_len, tin)
    assert logits.shape == (2, 5, vocab)
    hyps = m.generate(src, src_len, max_len=6)
    assert len(hyps) == 2
    beam = m.generate(src, src_len, max_len=6, beam_size=2, n_best=2)
    assert len(beam[0]) <= 2


def test_lstm_variant_runs():
    src, src_len, tin, vocab = _batch()
    m = RNNSeq2Seq(vocab, vocab, emb_dim=8, hid_dim=10, rnn_type="lstm", attention="luong_general")
    assert m(src, src_len, tin).shape == (2, 5, vocab)
