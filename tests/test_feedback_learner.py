"""Unit tests for the active feedback learning system (continuous online fine-tuning on GPU/CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from src.training.feedback_learner import (
    detect_feedback_target_model,
    online_learn_sample,
)


def test_detect_feedback_target_model():
    # English target -> translation_sota
    assert detect_feedback_target_model("repu college undi", "There is college tomorrow.") == "translation_sota"
    assert detect_feedback_target_model("Hi ella vunnav", "Greetings. How do you do?") == "translation_sota"

    # Telugu target -> production_sota
    assert detect_feedback_target_model("repu college undi", "రేపు college ఉంది") == "production_sota"


def test_online_learn_sample_translation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = online_learn_sample(
        src="repu college ki veltaanu",
        tgt="I shall proceed to college tomorrow",
        device=device,
        steps=2,
    )
    assert res["status"] == "success"
    assert res["model_updated"] == "translation_sota"
    assert "initial_loss" in res
    assert "final_loss" in res
    assert res["final_loss"] <= res["initial_loss"] + 0.05
