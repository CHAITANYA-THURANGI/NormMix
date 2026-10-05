"""Loads and caches checkpoints + the rule baseline; picks a default model; simple LRU-free cache."""
from __future__ import annotations

import threading
from pathlib import Path

import torch

from src.inference.generate import translate_texts
from src.models.factory import load_checkpoint
from src.models.rule_baseline import RuleBaseline


class ModelRegistry:
    def __init__(self, checkpoints_dir: str = "experiments/checkpoints", rule_path: str | None = None,
                default_model: str | None = None, device: str | None = None):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._cache: dict[str, tuple] = {}
        self.rule_baseline: RuleBaseline | None = None
        if rule_path and Path(rule_path).exists():
            self.rule_baseline = RuleBaseline.load(rule_path)
        self.default_model = default_model or self._discover_default()

    def _discover_default(self) -> str | None:
        if not self.checkpoints_dir.exists():
            return None
        preferred = ["production_sota", "attn_scaled_dot", "attn_bahdanau", "attn_luong_general", "attn_luong_concat",
                    "transformer_small", "plain_seq2seq"]
        names = {p.name for p in self.checkpoints_dir.iterdir() if (p / "best.pt").exists()}
        for p in preferred:
            if p in names:
                return p
        return sorted(names)[0] if names else None

    def available_models(self) -> list[str]:
        if not self.checkpoints_dir.exists():
            return []
        return sorted(p.name for p in self.checkpoints_dir.iterdir() if (p / "best.pt").exists())

    def get(self, name: str):
        with self._lock:
            if name not in self._cache:
                ckpt = self.checkpoints_dir / name / "best.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(f"No checkpoint at {ckpt}")
                self._cache[name] = load_checkpoint(ckpt, self.device)
            return self._cache[name]

    def normalize(self, texts: list[str], model_name: str | None, engine: str, beam_size: int, n_best: int):
        """Returns (per_text_hypotheses, engine_used, model_used)."""
        name = model_name or self.default_model
        use_rule = engine == "rule" or (engine == "auto" and name is None)
        if use_rule:
            if self.rule_baseline is None:
                raise RuntimeError("No neural model available and rule baseline is not loaded.")
            preds = self.rule_baseline.normalize_many(texts)
            return [[{"text": p, "confidence": 1.0}] for p in preds], "rule", "rule_baseline"
        model, src_tok, tgt_tok, _ = self.get(name)
        hyps = translate_texts(model, src_tok, tgt_tok, texts, self.device, beam_size=beam_size, n_best=n_best)
        return [[{"text": h["text"], "confidence": h["confidence"]} for h in hs] for hs in hyps], "neural", name
