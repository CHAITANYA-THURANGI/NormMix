"""Logging + lightweight experiment tracking (JSONL, optional TensorBoard / W&B)."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


def get_logger(name: str = "tecm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class ExperimentTracker:
    """Always writes metrics.jsonl; TensorBoard / W&B are opt-in and failure-tolerant."""

    def __init__(self, run_dir, use_tensorboard=False, use_wandb=False,
                 project="telugu-english-normalization", run_name=None, config=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self._tb = None
        self._wandb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception as e:  # pragma: no cover
                get_logger().warning("TensorBoard unavailable (%s); continuing without it.", e)
        if use_wandb:
            try:  # pragma: no cover
                import wandb

                self._wandb = wandb.init(project=project, name=run_name, config=config, dir=str(self.run_dir))
            except Exception as e:  # pragma: no cover
                get_logger().warning("W&B unavailable (%s); continuing without it.", e)

    def log(self, step, metrics):
        rec = {"step": step, "time": round(time.time(), 2), **metrics}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)
        if self._wandb is not None:  # pragma: no cover
            self._wandb.log(metrics, step=step)

    def close(self):
        self._f.close()
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:  # pragma: no cover
            self._wandb.finish()
