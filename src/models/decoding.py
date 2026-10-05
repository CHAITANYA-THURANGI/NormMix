"""Model-agnostic beam search. `step_fn(prev_tokens[k], state) -> (log_probs[k, V], new_state)`."""
from __future__ import annotations

import torch


def beam_search(step_fn, init_state, reorder_fn, sos_id: int, eos_id: int, beam_size: int = 4,
                max_len: int = 160, length_penalty: float = 1.0, n_best: int = 1, device="cpu") -> list[dict]:
    seqs: list[list[int]] = [[]]
    scores = torch.zeros(1, device=device)
    state = init_state
    prev = torch.tensor([sos_id], device=device)
    finished: list[tuple[float, float, list[int]]] = []
    for _ in range(max_len):
        logp, state = step_fn(prev, state)
        k, V = logp.shape
        cand = (scores[:, None] + logp).reshape(-1)
        topv, topi = cand.topk(min(cand.numel(), 2 * beam_size))
        new_seqs, new_scores, parents, new_prev = [], [], [], []
        for v, i in zip(topv.tolist(), topi.tolist()):
            b, tok = divmod(i, V)
            if tok == eos_id:
                toks = seqs[b]
                finished.append((v / (len(toks) + 1) ** length_penalty, v, toks))
            elif len(new_seqs) < beam_size:
                new_seqs.append(seqs[b] + [tok])
                new_scores.append(v)
                parents.append(b)
                new_prev.append(tok)
        if not new_seqs:
            break
        best_alive = max(new_scores) / (len(new_seqs[0]) + 1) ** length_penalty
        if len(finished) >= beam_size and max(f[0] for f in finished) >= best_alive:
            break
        seqs, scores = new_seqs, torch.tensor(new_scores, device=device)
        state = reorder_fn(state, torch.tensor(parents, device=device))
        prev = torch.tensor(new_prev, device=device)
    if len(finished) < n_best:
        for s, sc in zip(seqs, scores.tolist()):
            finished.append((sc / max(len(s), 1) ** length_penalty, sc, s))
    finished.sort(key=lambda f: f[0], reverse=True)
    return [{"ids": f[2], "score": f[1], "norm_score": f[0]} for f in finished[:n_best]]
