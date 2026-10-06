# NormMix API Reference

### Live Production Endpoints
- **Live Cloud API Base:** `https://normmix-api.onrender.com`
- **Interactive Swagger UI:** [https://normmix-api.onrender.com/docs](https://normmix-api.onrender.com/docs)
- **OpenAPI JSON Schema:** [https://normmix-api.onrender.com/openapi.json](https://normmix-api.onrender.com/openapi.json)
- **Live System Health:** [https://normmix-api.onrender.com/health](https://normmix-api.onrender.com/health)

### Local Development Server
From the project root:
```bash
uvicorn api.main:app --reload --port 8000
```
Local Swagger docs: `http://127.0.0.1:8000/docs`

---

## 1. System Telemetry & Health

### `GET /health`
Returns the status of loaded neural checkpoints, lexicon, and hardware accelerator.

**Response:**
```json
{
  "status": "ok",
  "default_model": "attn_bahdanau",
  "models_loaded": ["attn_bahdanau"],
  "rule_baseline_loaded": true,
  "device": "cpu"
}
```

### `GET /models`
Lists all trained checkpoints under `experiments/checkpoints/*/best.pt` and the rule baseline with metadata.

---

## 2. Universal Omni-Directional Processor

### `POST /omni/process`
Auto-detects the input modality (*Pure English*, *Native Telugu Script*, *Romanized Tanglish*, or *Bi-scriptal Code-Mixed*) and simultaneously computes all target representations.

**Request:**
```json
{
  "text": "college lo project report ready ayindi",
  "engine": "auto",
  "beam_size": 4,
  "n_best": 1
}
```

**Response:**
```json
{
  "source_text": "college lo project report ready ayindi",
  "detected": {
    "modality": "romanized_tanglish",
    "label": "Romanized Tanglish (Telugu-English Code-Mixed)",
    "confidence": 0.98,
    "telugu_char_pct": 0,
    "latin_char_pct": 100
  },
  "options": {
    "normalized_code_mixed": "college లో project report ready అయింది",
    "english_translation": "The project report in college is ready.",
    "pure_telugu": "కళాశాలలో కార్య నివేదిక సిద్ధమైంది.",
    "all_telugu_script": "కాలేజీ లో ప్రాజెక్ట్ రిపోర్ట్ రెడీ అయింది",
    "all_romanized_tanglish": "college lo project report ready ayindi",
    "english_gloss": "college in project report ready became",
    "prescribed_meaning": "The project report in college is ready."
  },
  "tokens": [
    {
      "source": "college",
      "normalized": "college",
      "telugu_script": "కాలేజీ",
      "romanized": "college",
      "tag": "EN",
      "gloss": "college"
    },
    {
      "source": "lo",
      "normalized": "లో",
      "telugu_script": "లో",
      "romanized": "lo",
      "tag": "TE",
      "gloss": "in"
    }
  ],
  "recommended": {
    "engine": "SOTA Pointer-Generator BiGRU",
    "model": "attn_bahdanau",
    "confidence": 0.98
  },
  "latency_ms": 7.4
}
```

---

## 3. Code-Mixed Normalization

### `POST /normalize`
Normalizes Romanized Tanglish into Telugu script while preserving English loanwords.

**Request:**
```json
{
  "text": "naku ivala college ki vellali",
  "n_best": 1,
  "beam_size": 4,
  "engine": "auto",
  "model": null
}
```

**Response:**
```json
{
  "input": "naku ivala college ki vellali",
  "hypotheses": [
    {
      "text": "నాకు ఇవాళ college కి వెళ్ళాలి",
      "confidence": 0.94
    }
  ],
  "engine_used": "neural",
  "model": "attn_bahdanau",
  "latency_ms": 8.2
}
```

### `POST /normalize/batch`
Batch normalization supporting up to 32 inputs per request.

---

## 4. Reverse Phonetic Transliteration

### `POST /romanize`
Transliterates Telugu script text into Romanized phonetic Tanglish.

**Request:**
```json
{
  "text": "నాకు ఇవాళ కాలేజీ కి వెళ్ళాలి"
}
```

**Response:**
```json
{
  "source": "నాకు ఇవాళ కాలేజీ కి వెళ్ళాలి",
  "romanized": "naku ivala kaleji ki vellali",
  "latency_ms": 1.2
}
```

---

## 5. Active Learning & Reinforcement Feedback

### `POST /feedback`
Records user ratings and ground-truth corrections to `data/feedback.jsonl` for continuous model reinforcement.

**Request:**
```json
{
  "input_text": "hello ella vunnavuu",
  "output_text": "హలో ఎలా ఉన్నావు?",
  "rating": "positive",
  "mode": "auto",
  "correction": null,
  "comments": null
}
```

**Response:**
```json
{
  "status": "recorded",
  "feedback_id": "fb-9a2c1f-4b",
  "learned": true,
  "model_updated": "attn_bahdanau"
}
```

---

## 6. Lexicon Explorer

### `GET /dictionary/lookup?q={query}`
Queries the 64,421 verified lexicon knowledge base.

**Response:**
```json
{
  "query": "college",
  "count": 4,
  "matches": [
    { "src": "college", "tgt": "కాలేజీ", "match": "exact_loanword" },
    { "src": "college lo", "tgt": "కాలేజీలో", "match": "inflected_locative" }
  ]
}
```

---

## 7. Error Handling & Rate Limiting

| Status Code | Description | Solution |
| :--- | :--- | :--- |
| **422 Unprocessable Entity** | Validation error (blank input, batch > 32, text > 2000 chars) | Provide valid UTF-8 string input. |
| **429 Too Many Requests** | Rate limit exceeded (Default: 120 req/min per IP) | Back off requests or request IP whitelisting. |
| **503 Service Unavailable** | Cold start / model loading | Retry after 3 seconds or rely on in-browser engine. |

**CORS Policy:** `allow_origins: ["*"]` is enabled in production, allowing requests from `https://chaitanya-thurangi.github.io` and Chrome Extension origins (`chrome-extension://*`).
