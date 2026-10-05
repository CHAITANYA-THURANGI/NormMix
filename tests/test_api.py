"""API tests using FastAPI's TestClient (httpx) against the in-process app.

Uses whatever checkpoints/rule baseline exist in experiments/ and data/processed/ at test time --
run `python scripts/prepare_data.py && python scripts/train_rule_baseline.py` first (the CI config
does this; see .github/workflows/ci.yml).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

RULE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "rule_baseline.json"
pytestmark = pytest.mark.skipif(not RULE_PATH.exists(), reason="run scripts/train_rule_baseline.py first")


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["rule_baseline_loaded"] is True


def test_normalize_rule_engine(client):
    r = client.post("/normalize", json={"text": "naku ivala college ki vellali", "engine": "rule"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine_used"] == "rule"
    assert len(body["hypotheses"]) == 1
    assert "college" in body["hypotheses"][0]["text"]


def test_normalize_rejects_blank_text(client):
    r = client.post("/normalize", json={"text": "   "})
    assert r.status_code == 422


def test_normalize_batch(client):
    r = client.post("/normalize/batch", json={"texts": ["nuvvu ekkada unnav", "class ki vastunnava?"], "engine": "rule"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2


def test_models_endpoint_lists_rule_baseline(client):
    r = client.get("/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "rule_baseline" in names


def test_batch_rejects_oversized_batch(client):
    r = client.post("/normalize/batch", json={"texts": ["hi"] * 100, "engine": "rule"})
    assert r.status_code == 422


def test_romanize_endpoint(client):
    r = client.post("/romanize", json={"text": "నాకు ఇవాళ కాలేజీ కి వెళ్లాలి"})
    assert r.status_code == 200
    body = r.json()
    assert "romanized" in body
    assert body["mode"] == "telugu_to_roman"
    assert len(body["romanized"]) > 0


def test_dictionary_lookup(client):
    r = client.get("/dictionary/lookup?q=college")
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "college"
    assert "matches" in body


def test_omni_process_endpoint(client):
    # 1. Test Romanized Tanglish
    r1 = client.post("/omni/process", json={"text": "naku ivala college lo important interview undi", "engine": "rule"})
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["detected"]["modality"] == "romanized_tanglish"
    assert "all_telugu_script" in b1["options"]
    assert "all_romanized_tanglish" in b1["options"]
    assert "english_gloss" in b1["options"]

    # 2. Test Pure Telugu
    r2 = client.post("/omni/process", json={"text": "నాకు ఇవాళ ముఖ్యమైన పని ఉంది"})
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["detected"]["modality"] == "telugu_script_pure"

    # 3. Test Pure English
    r3 = client.post("/omni/process", json={"text": "Where are you going today?"})
    assert r3.status_code == 200
    b3 = r3.json()
    assert b3["detected"]["modality"] == "pure_english"


def test_feedback_endpoint(client):
    r = client.post("/feedback", json={
        "input_text": "naku ivala college undi",
        "output_text": "నాకు ఇవాళ college ఉంది",
        "rating": "positive",
        "comments": "Great copy attention"
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "feedback_id" in body


def test_feedback_endpoint_with_learning(client):
    r = client.post("/feedback", json={
        "input_text": "repu college ki veltaanu",
        "output_text": "I will go to college tomorrow",
        "rating": "negative",
        "correction": "I shall proceed to college tomorrow",
        "comments": "User suggested formal pure English"
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["learned"] is True
    assert body["model_updated"] == "translation_sota"


def test_translate_endpoint(client):
    r = client.post("/translate", json={
        "text": "Hi Ella, vunnavu. Are you coming to college?",
        "target": "english"
    })
    assert r.status_code == 200
    body = r.json()
    assert "english_translation" in body
    assert "pure_english" in body
    assert len(body["translation"]) > 0




