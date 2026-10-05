"""Pydantic request/response schemas for the normalization API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 2000
MAX_BATCH = 32


class Hypothesis(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    n_best: int = Field(1, ge=1, le=5)
    beam_size: int = Field(4, ge=1, le=8)
    model: str | None = Field(None, description="Run name under experiments/checkpoints/, e.g. 'production_sota'")
    engine: Literal["neural", "rule", "auto"] = "auto"
    mode: Literal["normalize", "romanize"] = "normalize"

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class NormalizeResponse(BaseModel):
    input: str
    hypotheses: list[Hypothesis]
    engine_used: Literal["neural", "rule"]
    model: str
    latency_ms: float
    mode: str = "normalize"


class RomanizeResponse(BaseModel):
    input: str
    romanized: str
    latency_ms: float
    mode: str = "telugu_to_roman"


class BatchNormalizeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    n_best: int = 1
    beam_size: int = 4
    model: str | None = None
    engine: Literal["neural", "rule", "auto"] = "auto"

    @field_validator("texts")
    @classmethod
    def _lens(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip():
                raise ValueError("texts must not contain blank strings")
            if len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"each text must be <= {MAX_TEXT_CHARS} chars")
        return v


class BatchNormalizeResponse(BaseModel):
    results: list[NormalizeResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    name: str
    type: str
    attention: str | None = None
    params: int | None = None
    is_default: bool = False


class HealthResponse(BaseModel):
    status: str
    default_model: str | None
    models_loaded: list[str]
    rule_baseline_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class DetectedModality(BaseModel):
    modality: str
    label: str
    confidence: float
    telugu_char_pct: float
    latin_char_pct: float
    token_count: int
    tokens: list[dict] = Field(default_factory=list)


class RecommendedOutput(BaseModel):
    text: str
    confidence: float
    engine: str
    model: str


class OmniOptions(BaseModel):
    normalized_code_mixed: str
    english_translation: str = ""
    pure_english: str = ""
    pure_telugu: str = ""
    all_telugu_script: str
    all_romanized_tanglish: str
    english_gloss: str
    prescribed_meaning: str = ""


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    target: Literal["english", "pure_english", "pure_telugu"] = "english"
    engine: Literal["neural", "rule", "auto"] = "auto"


class TranslateResponse(BaseModel):
    input: str
    target: str
    translation: str
    english_translation: str = ""
    pure_english: str = ""
    latency_ms: float



class OmniProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    engine: Literal["neural", "rule", "auto"] = "auto"
    beam_size: int = Field(4, ge=1, le=8)
    n_best: int = Field(1, ge=1, le=5)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class OmniProcessResponse(BaseModel):
    input: str
    detected: DetectedModality
    recommended: RecommendedOutput
    options: OmniOptions
    tokens: list[dict] = Field(default_factory=list)
    latency_ms: float


class FeedbackRequest(BaseModel):
    input_text: str = Field(..., max_length=MAX_TEXT_CHARS)
    output_text: str = Field(..., max_length=MAX_TEXT_CHARS)
    rating: Literal["positive", "negative"]
    mode: str = "auto"
    correction: str | None = None
    comments: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str


