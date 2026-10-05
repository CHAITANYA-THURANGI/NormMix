"""FastAPI service for Telugu-English code-mixed text normalization.

Run:  uvicorn api.main:app --reload --port 8000    (from the project root)
Docs: http://127.0.0.1:8000/docs

Falls back to the rule baseline automatically if no trained checkpoint exists yet, so the API,
web demo and Chrome extension are usable immediately after `python scripts/prepare_data.py` +
`python scripts/train_rule_baseline.py`, before any neural model is trained.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import datetime
import json
import uuid

from api.schemas import (BatchNormalizeRequest, BatchNormalizeResponse, HealthResponse, ModelInfo,
                         NormalizeRequest, NormalizeResponse, RomanizeResponse, Hypothesis,
                         OmniProcessRequest, OmniProcessResponse, FeedbackRequest, FeedbackResponse)

from api.services.model_registry import ModelRegistry
from api.services.rate_limit import RateLimiter
from src.config import load_config
from src.pipeline.omni_processor import OmniProcessor
from src.preprocessing.translit_te import romanize_text
from src.utils.io import read_json

CFG = load_config(base_path=ROOT / "config.yaml")
registry = ModelRegistry(ROOT / "experiments" / "checkpoints", ROOT / CFG["inference"]["lexicon"])
omni = OmniProcessor(registry)
limiter = RateLimiter(per_minute=CFG["api"].get("rate_limit_per_min", 120))


app = FastAPI(title="Telugu-English Code-Mixed Text Normalization API", version="0.5.0",
              description="Enterprise Attention Seq2Seq normalization & bidirectional transliteration of Telugu-English code-mixed text.")
app.add_middleware(CORSMiddleware, allow_origins=CFG["api"].get("cors_origins", ["*"]), allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def rate_limit_mw(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        return JSONResponse(status_code=429, content={"error": "rate_limited", "detail": "Too many requests."})
    return await call_next(request)


def _run(texts, req) -> list[NormalizeResponse]:
    t0 = time.perf_counter()
    if getattr(req, "mode", "normalize") == "romanize":
        dt = (time.perf_counter() - t0) * 1000
        per_item = dt / max(len(texts), 1)
        return [NormalizeResponse(input=t, hypotheses=[Hypothesis(text=romanize_text(t), confidence=0.99)],
                                  engine_used="rule", model="rule_romanizer",
                                  latency_ms=round(per_item, 2), mode="romanize") for t in texts]
    try:
        per_text, engine_used, model_used = registry.normalize(texts, req.model, req.engine, req.beam_size, req.n_best)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    dt = (time.perf_counter() - t0) * 1000
    per_item = dt / max(len(texts), 1)
    return [NormalizeResponse(input=t, hypotheses=hs, engine_used=engine_used, model=model_used,
                              latency_ms=round(per_item, 2), mode="normalize")
            for t, hs in zip(texts, per_text)]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", default_model=registry.default_model, models_loaded=list(registry._cache),
                          rule_baseline_loaded=registry.rule_baseline is not None, device=registry.device)


@app.get("/models", response_model=list[ModelInfo])
def models():
    out = []
    for name in registry.available_models():
        try:
            cfg = read_json(registry.checkpoints_dir / name / "config.json")
            mcfg = cfg.get("model", {})
        except FileNotFoundError:
            mcfg = {}
        out.append(ModelInfo(name=name, type=mcfg.get("type", "?"), attention=mcfg.get("attention"),
                             is_default=(name == registry.default_model)))
    if registry.rule_baseline is not None:
        out.append(ModelInfo(name="rule_baseline", type="rule", is_default=registry.default_model is None))
    return out


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(req: NormalizeRequest):
    return _run([req.text], req)[0]


@app.post("/normalize/batch", response_model=BatchNormalizeResponse)
def normalize_batch(req: BatchNormalizeRequest):
    if len(req.texts) > CFG["api"].get("max_batch", 32):
        raise HTTPException(422, f"Batch too large (max {CFG['api'].get('max_batch', 32)}).")
    t0 = time.perf_counter()
    fake_req = NormalizeRequest(text="x", n_best=req.n_best, beam_size=req.beam_size, model=req.model, engine=req.engine)
    results = _run(req.texts, fake_req)
    return BatchNormalizeResponse(results=results, total_latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/romanize", response_model=RomanizeResponse)
def romanize_endpoint(req: NormalizeRequest):
    t0 = time.perf_counter()
    rom = romanize_text(req.text)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return RomanizeResponse(input=req.text, romanized=rom, latency_ms=latency, mode="telugu_to_roman")


@app.post("/omni/process", response_model=OmniProcessResponse)
def omni_process_endpoint(req: OmniProcessRequest):
    return omni.process(req.text, engine=req.engine, beam_size=req.beam_size, n_best=req.n_best)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(req: FeedbackRequest, request: Request):
    feedback_id = str(uuid.uuid4())[:8]
    entry = {
        "id": feedback_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "client": request.client.host if request.client else "unknown",
        "input": req.input_text,
        "output": req.output_text,
        "rating": req.rating,
        "mode": req.mode,
        "correction": req.correction,
        "comments": req.comments,
    }
    feedback_file = ROOT / "data" / "feedback.jsonl"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return FeedbackResponse(status="success", message="Thank you! Feedback recorded for model reinforcement.", feedback_id=feedback_id)




@app.get("/dictionary/lookup")
def dict_lookup(q: str):
    matches = []
    if registry.rule_baseline and registry.rule_baseline.lex:
        lex = registry.rule_baseline.lex
        query = q.strip().lower()
        if query in lex:
            matches.append({"src": query, "tgt": lex[query], "match": "exact"})
        for k, v in lex.items():
            if len(matches) >= 8:
                break
            if k.startswith(query) and k != query:
                matches.append({"src": k, "tgt": v, "match": "prefix"})
            elif v == q.strip():
                matches.append({"src": k, "tgt": v, "match": "reverse"})
    return {"query": q, "count": len(matches), "matches": matches}


@app.get("/")
def root():
    index_path = ROOT / "web" / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"name": "Telugu-English Code-Mixed Text Normalization API", "docs": "/docs", "health": "/health"}


@app.get("/report.pdf")
def download_report():
    pdf_path = ROOT / "docs" / "project_report.pdf"
    if pdf_path.exists():
        return FileResponse(pdf_path, media_type="application/pdf", filename="Telugu_English_CodeMixed_Normalization_Report.pdf")
    raise HTTPException(status_code=404, detail="Project report PDF not generated yet.")


@app.get("/api")
def api_info():
    return {"name": "Telugu-English Code-Mixed Text Normalization API", "docs": "/docs", "health": "/health"}

