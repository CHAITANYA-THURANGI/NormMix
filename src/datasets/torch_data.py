"""PyTorch datasets, bucketing samplers and collation."""
from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from ..tokenization.base import PAD_ID


def mix_sources(sources: list[tuple[list[dict], float]], seed: int = 42) -> list[dict]:
    """Weighted concatenation: weight 2.0 = every row twice; 0.5 = a random half."""
    rng = random.Random(seed)
    out: list[dict] = []
    for rows, w in sources:
        whole, frac = int(w), w - int(w)
        out += rows * whole
        if frac > 0:
            out += rng.sample(rows, int(len(rows) * frac))
    return out


class PairDataset(Dataset):
    def __init__(self, rows, src_tok, tgt_tok, max_src_len=128, max_tgt_len=128, drop_long=True):
        self.items = []
        self.dropped = 0
        for i, r in enumerate(rows):
            s = src_tok.encode(r["src"])
            t = tgt_tok.encode(r["tgt"])
            if drop_long and (len(s) > max_src_len or len(t) + 1 > max_tgt_len):
                self.dropped += 1
                continue
            self.items.append((i, s[:max_src_len], t[:max_tgt_len - 1]))
        self.sos, self.eos = tgt_tok.sos_id, tgt_tok.eos_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, s, t = self.items[k]
        return i, s, [self.sos] + t, t + [self.eos]


def collate(batch):
    idx = [b[0] for b in batch]
    src_len = torch.tensor([max(len(b[1]), 1) for b in batch])
    S = int(src_len.max())
    T = max(len(b[2]) for b in batch)
    src = torch.full((len(batch), S), PAD_ID, dtype=torch.long)
    tin = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    tout = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    for k, (_, s, ti, to) in enumerate(batch):
        if s:
            src[k, : len(s)] = torch.tensor(s)
        tin[k, : len(ti)] = torch.tensor(ti)
        tout[k, : len(to)] = torch.tensor(to)
    return {"idx": idx, "src": src, "src_len": src_len, "tgt_in": tin, "tgt_out": tout}


def _batches(ds: PairDataset, batch_size: int, shuffle: bool, seed: int):
    order = list(range(len(ds)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)
        chunk = batch_size * 50
        groups = [sorted(order[i:i + chunk], key=lambda k: len(ds.items[k][1])) for i in range(0, len(order), chunk)]
        batches = [g[i:i + batch_size] for g in groups for i in range(0, len(g), batch_size)]
        rng.shuffle(batches)
    else:
        order.sort(key=lambda k: len(ds.items[k][1]))
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    return batches


class BucketLoader:
    """Length-bucketed batches, re-shuffled every epoch (call set_epoch)."""

    def __init__(self, ds: PairDataset, batch_size: int, shuffle: bool, seed: int = 42):
        self.ds, self.batch_size, self.shuffle, self.seed, self.epoch = ds, batch_size, shuffle, seed, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return (len(self.ds) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for b in _batches(self.ds, self.batch_size, self.shuffle, self.seed + self.epoch):
            yield collate([self.ds[k] for k in b])


def make_loader(rows, src_tok, tgt_tok, batch_size, shuffle, max_src_len=128, max_tgt_len=128, seed=42, drop_long=True):
    ds = PairDataset(rows, src_tok, tgt_tok, max_src_len, max_tgt_len, drop_long)
    return BucketLoader(ds, batch_size, shuffle, seed)
