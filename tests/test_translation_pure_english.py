"""Tests for English and Pure English translation & GPU neural model integration."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
from fastapi.testclient import TestClient

from src.pipeline.omni_processor import (
    translate_to_english,
    translate_to_pure_english,
    elevate_to_pure_english,
    detransliterate_telugu_loanwords,
    OmniProcessor,
)
from api.main import app
from api.services.model_registry import ModelRegistry


def test_detransliterate_telugu_loanwords():
    raw = "మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్"
    res = detransliterate_telugu_loanwords(raw)
    assert "convert" in res.lower()
    assert "sentence" in res.lower()
    assert "english" in res.lower()


def test_user_sentence_translation_to_english():
    raw = "మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్"
    en = translate_to_english(raw)
    assert "You look very good" in en or "very good" in en
    assert "convert this sentence into english" in en.lower()


def test_user_sentence_translation_to_pure_english():
    raw = "మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్"
    pure_en = translate_to_pure_english(raw)
    assert "You appear exceptionally well" in pure_en or "well" in pure_en
    assert "please" in pure_en.lower() or "translate" in pure_en.lower() or "convert" in pure_en.lower()
    assert "English" in pure_en or "english" in pure_en.lower()


def test_pure_telugu_and_tanglish_translations():
    # Greetings
    en1 = translate_to_english("మీరు ఎలా ఉన్నారు?")
    pure_en1 = translate_to_pure_english("మీరు ఎలా ఉన్నారు?")
    assert "How are you" in en1
    assert "How do you do" in pure_en1

    en2 = translate_to_english("meeru ela unnaru?")
    pure_en2 = translate_to_pure_english("meeru ela unnaru?")
    assert "How are you" in en2
    assert "How do you do" in pure_en2

    # Status
    en3 = translate_to_english("మీరు చాలా బాగున్నారు")
    pure_en3 = translate_to_pure_english("మీరు చాలా బాగున్నారు")
    assert "You look very good" in en3 or "good" in en3
    assert "You appear exceptionally well" in pure_en3 or "well" in pure_en3


def test_elevate_to_pure_english():
    elevated = elevate_to_pure_english("I can't come today, please give me some help.")
    assert "cannot" in elevated
    assert "assistance" in elevated


def test_omni_processor_returns_pure_english():
    op = OmniProcessor(registry=None)
    res = op.process("మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్")
    opts = res["options"]
    assert opts["english_translation"] != ""
    assert opts["pure_english"] != ""
    assert opts["english_translation"] != opts["all_telugu_script"]
    assert "English" in opts["pure_english"] or "english" in opts["pure_english"].lower()


def test_translate_api_endpoint():
    with TestClient(app) as client:
        # Standard English target
        r1 = client.post("/translate", json={
            "text": "మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్",
            "target": "english"
        })
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["target"] == "english"
        assert "convert this sentence into english" in body1["translation"].lower()

        # Pure English target
        r2 = client.post("/translate", json={
            "text": "మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్",
            "target": "pure_english"
        })
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["target"] == "pure_english"
        assert "English" in body2["translation"]
        assert body2["pure_english"] != ""

        # Pure Telugu target
        r3 = client.post("/translate", json={
            "text": "meeru ela unnaru?",
            "target": "pure_telugu"
        })
        assert r3.status_code == 200
        body3 = r3.json()
        assert body3["target"] == "pure_telugu"
        assert len(body3["translation"]) > 0


def test_gpu_neural_translation_model():
    chk_dir = ROOT / "experiments" / "checkpoints" / "translation_sota"
    if not (chk_dir / "best.pt").exists():
        pytest.skip("translation_sota checkpoint not found on disk")

    registry = ModelRegistry(ROOT / "experiments" / "checkpoints", ROOT / "data" / "lexicons" / "high_frequency_lexicon.json")
    preds = registry.translate(["మీరు ఎలా ఉన్నారు?", "naku ivala college lo interview undi"], model_name="translation_sota")
    assert len(preds) == 2
    assert all(isinstance(p, str) and len(p) > 0 for p in preds)
