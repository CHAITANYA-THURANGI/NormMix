"""YAML configuration with deep-merge defaults and dotted CLI overrides.

Usage:
    cfg = load_config("config.yaml", overrides=["train.epochs=5", "model.attention=luong_general"])
    cfg["train"]["epochs"]
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s: str) -> Any:
    return yaml.safe_load(s)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                base_path: str | Path | None = None) -> dict:
    """Load `config.yaml` (defaults) then merge an optional experiment config, then CLI overrides."""
    base_file = Path(base_path) if base_path else ROOT / "config.yaml"
    with open(base_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path and Path(path).resolve() != base_file.resolve():
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
