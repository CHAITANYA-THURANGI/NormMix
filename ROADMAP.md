# Project Roadmap

## Telugu-English Code-Mixed Text Normalization

---

### v0.1 — Dataset Pipeline ✅ DONE
- Built-in seed corpus with 120+ template-generated Telugu-English sentences
- Noise generation engine (spelling, phonetic, keyboard, elongation)
- Synthetic code-mixed pair generation
- Leakage-safe splitting by target group hash
- JSONL serialization with statistics

### v0.2 — Rule Baseline ✅ DONE
- Dictionary-based normalization from aligned training pairs
- Telugu suffix decomposition (collegeki → college కి)
- Fuzzy matching via difflib
- JSON serialization for API fallback
- Baseline evaluation metrics

### v0.3 — Basic Seq2Seq ✅ DONE
- BiGRU/LSTM encoder with packed sequences
- GRU/LSTM decoder with teacher forcing
- Bridge layer (encoder final state → decoder init)
- Greedy decoding
- Config-driven architecture (`model.attention: none`)

### v0.4 — Attention-Based Seq2Seq ✅ DONE (Core Model)
- Bahdanau (additive) attention
- Luong attention (dot, general, concat)
- Scaled dot-product attention
- Attention weight extraction for visualization
- Beam search decoding with length penalty

### v0.5 — Model Comparison Framework ✅ DONE
- Transformer encoder-decoder (Pre-LN, sinusoidal positions)
- Per-experiment YAML config overrides
- `run_all_experiments.py` automation script
- Unified evaluation across all model variants
- Result JSON output

### v0.6 — API Server ✅ DONE
- FastAPI with `/normalize`, `/normalize/batch`, `/health`, `/models`
- IP-based rate limiting (120 req/min)
- CORS middleware
- Auto-fallback to rule baseline if no neural checkpoint
- Pydantic request/response schemas

### v0.7 — Web Interface ✅ DONE
- Single-file HTML/CSS/JS demo (`web/index.html`)
- Real-time normalization via API
- Multiple candidate display
- Confidence score visualization

### v0.8 — Chrome Extension + Mobile ✅ DONE
- Manifest V3 Chrome extension with context menu
- Popup UI for normalization
- Content script for text selection
- Flutter mobile scaffold with API integration

### v0.9 — Optimization & Real Dataset Integration 🔄 IN PROGRESS
**Next steps:**
- Download and integrate Dakshina transliteration dataset
- Download and integrate Aksharantar transliteration dataset
- Re-train models on combined seed + real data
- Hyperparameter tuning (embedding dim, hidden dim)
- Scheduled sampling experiments
- Mixed precision training (now implemented)

### v1.0 — Complete AAN Mini-Project Release 📋 PLANNED
**Deliverables:**
- All experiments run and results documented
- Academic report written
- Attention visualization notebook
- Error analysis with taxonomy
- Clean GitHub repository with documentation
- Project presentation slides

### v2.0 — Production/Research Version 🔮 FUTURE
**Possible additions:**
- Fine-tuned ByT5 or IndicBART comparison
- Large-scale real social media dataset
- Cloud deployment (Google Cloud Run)
- ONNX model export for on-device inference
- User feedback loop and continual learning
- Multilingual support (Hindi-English, Tamil-English)
