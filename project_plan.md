# Project Plan — Telugu-English Code-Mixed Text Normalization

## AAN Mini-Project / Case Study

### Course: Advanced Artificial Intelligence and Neural Networks

---

## Project Title

**"An Attention-Based Sequence-to-Sequence Framework for Telugu-English Code-Mixed Text Normalization"**

---

## Project Objectives

1. Build a character-level Seq2Seq model with Bahdanau attention that normalizes noisy Telugu-English
   code-mixed text (romanized Telugu → Telugu script, preserve English, fix spelling).
2. Compare five attention mechanisms (Bahdanau, Luong dot/general/concat, scaled dot-product) and a
   no-attention baseline to demonstrate attention's value for code-mixed text.
3. Implement a rule/dictionary baseline as a lower bound and API fallback.
4. Include a Transformer comparison model to contextualize RNN-based approaches.
5. Evaluate using CER, WER, exact match, chrF, and BLEU across multiple test slices.
6. Deploy as a FastAPI web service with a browser demo and Chrome extension scaffold.
7. Document the complete system for academic submission.

---

## Experiment Plan

### Experiment Table

| # | Experiment | Hypothesis | Config | Primary Metric |
|---|-----------|-----------|--------|---------------|
| 1 | Rule baseline | Establishes lower bound | `train_rule_baseline.py` | CER |
| 2 | Seq2Seq (no attention) | Information bottleneck limits performance | `plain_seq2seq.yaml` | CER |
| 3 | Seq2Seq + Bahdanau | Attention significantly reduces CER | `attn_bahdanau.yaml` | CER |
| 4 | Seq2Seq + Luong (general) | Multiplicative attention competitive | `attn_luong_general.yaml` | CER |
| 5 | Seq2Seq + Luong (concat) | Concat similar to Bahdanau | `attn_luong_concat.yaml` | CER |
| 6 | Seq2Seq + Scaled dot-product | Transformer-style attention | `attn_scaled_dot.yaml` | CER |
| 7 | Transformer | More params, needs more data | `transformer_small.yaml` | CER |
| 8 | GRU vs LSTM | GRU faster, similar accuracy | Override `model.rnn_type` | CER + train time |
| 9 | Emb dim 64 vs 96 vs 128 | 96 is sweet spot | Override `model.emb_dim` | CER |
| 10 | Hidden dim 128 vs 192 vs 256 | 192 balances capacity/speed | Override `model.hid_dim` | CER |
| 11 | Char vs BPE tokenization | Char better for noisy text | Override `tokenizer.type` | CER |
| 12 | Label smoothing 0 vs 0.05 vs 0.1 | Mild smoothing helps | Override `train.label_smoothing` | CER |
| 13 | With vs without augmentation | Augmentation improves generalization | Override `data.variants_per_sentence` | CER on test_unseen |
| 14 | Scheduled sampling (tf 1.0→0.5) | Reduces exposure bias | Override `train.tf_ratio_end: 0.5` | CER |

### How to Run All Experiments

```bash
python scripts/run_all_experiments.py
```

Or individual experiments:
```bash
python scripts/train.py --config experiments/configs/attn_bahdanau.yaml
python scripts/train.py --config experiments/configs/plain_seq2seq.yaml
```

---

## Evaluation Strategy

### Metrics (implemented in `src/evaluation/metrics.py`)

| Metric | Implementation | What it Measures |
|--------|---------------|-----------------|
| CER | `cer(preds, refs)` | Character-level edit distance ratio |
| WER | `wer(preds, refs)` | Word-level edit distance ratio |
| Exact Match | `exact_match(preds, refs)` | % of perfect predictions |
| chrF | `chrf(preds, refs)` | Char n-gram F-score (0-100) |
| BLEU | `bleu(preds, refs)` | Word n-gram precision (0-100) |
| Token Accuracy | `token_accuracy(preds, refs)` | LCS-based word reproduction |
| NED | `normalized_edit_distance(preds, refs)` | Mean normalized edit distance |

### Test Slices (from `src/datasets/pipeline.py`)

| Split | Purpose |
|-------|---------|
| `test` | Standard held-out test set |
| `test_unseen_words` | English words never seen during training |
| `test_hard` | Heavily noisy inputs |
| `gold_demo` | 4 hand-verified gold pairs |

### Evaluation Command

```bash
python scripts/evaluate.py \
  --checkpoint experiments/checkpoints/attn_bahdanau \
  --rule-baseline data/processed/rule_baseline.json \
  --split test test_unseen_words gold_demo \
  --errors
```

---

## Error Analysis Framework

### Error Categories (from `src/evaluation/error_analysis.py`)

1. **english_spelling** — English word with minor spelling error (edit distance ≤ 2)
2. **english_as_telugu** — English word incorrectly transliterated to Telugu
3. **english_other** — Other English word errors
4. **telugu_left_romanized** — Telugu word left in Roman script (not transliterated)
5. **telugu_char_error** — Telugu word with minor character errors
6. **telugu_vocab** — Telugu word replaced with wrong word entirely
7. **insertion** — Extra tokens in prediction
8. **deletion** — Missing tokens in prediction
9. **punctuation_emoji** — Punctuation/emoji handling errors

### Additional Flags
- **unknown_word** — Token not in training vocabulary
- **hallucinated** — Token in prediction not in input or training vocabulary

---

## Hardware Requirements

| Setup | CPU | GPU | RAM | Storage | Train Time (seed corpus) |
|-------|-----|-----|-----|---------|-------------------------|
| **Laptop (minimum)** | 4 cores | None | 8 GB | 2 GB | ~30 min (CPU) |
| **Colab Free** | 2 cores | T4 16GB | 12 GB | 15 GB | ~5-10 min |
| **Colab Pro** | 4+ cores | A100 40GB | 32+ GB | 50+ GB | ~2 min |

---

## Academic Report Outline (for AAN case study)

### Suggested Structure

1. **Title Page** — Project title, student name, course, date
2. **Abstract** (200 words) — Problem, approach, key results
3. **Introduction** (1-2 pages) — Code-mixing prevalence, motivation, objectives
4. **Literature Survey** (2-3 pages) — 10-15 cited papers on:
   - Code-mixed text processing
   - Telugu NLP
   - Attention mechanisms
   - Sequence-to-sequence models
   - Text normalization
5. **Problem Statement** (0.5 pages) — Formal definition
6. **Proposed System** (3-4 pages) — Architecture, math, components
7. **Dataset** (1-2 pages) — Sources, statistics, preprocessing, limitations
8. **Implementation** (2-3 pages) — Tools, code structure, training details
9. **Experiments & Results** (3-4 pages) — All comparisons with tables and charts
10. **Error Analysis** (1-2 pages) — Error taxonomy with examples
11. **Deployment** (1 page) — API, web demo, extension
12. **Limitations** (0.5 pages) — Honest assessment
13. **Future Scope** (1 page) — Research directions
14. **Conclusion** (0.5 pages) — Summary
15. **References** (1-2 pages) — 15-25 works

### Required Figures/Tables

- [ ] System architecture diagram
- [ ] Encoder-Decoder architecture with attention
- [ ] Attention weight heatmap visualization
- [ ] Training/validation loss curves
- [ ] Model comparison bar chart (CER across all models)
- [ ] Attention mechanism comparison table
- [ ] Error distribution pie chart
- [ ] Sample predictions table (input → predicted → expected)
- [ ] Dataset statistics table
- [ ] Hyperparameter configuration table

---

## Deliverable Checklist

### Core (Required for AAN submission)

- [x] Problem definition
- [x] Dataset pipeline (seed corpus + augmentation)
- [x] Data preprocessing (Unicode NFC, cleaning, char clamping)
- [x] Character-level tokenizer
- [x] Rule/dictionary baseline
- [x] Seq2Seq without attention (baseline 2)
- [x] Seq2Seq + Bahdanau attention (core model)
- [x] Seq2Seq + Luong attention variants
- [x] Seq2Seq + Scaled dot-product attention
- [x] Transformer comparison model
- [x] Training pipeline (early stopping, gradient clipping, label smoothing)
- [x] Evaluation pipeline (CER, WER, chrF, BLEU, EM)
- [x] Error analysis framework
- [x] Experiment configuration system
- [x] Inference pipeline (greedy + beam search)
- [x] FastAPI backend
- [x] Web application
- [x] Chrome extension scaffold
- [x] Unit tests (39 tests)
- [x] README
- [x] Configuration system (YAML + CLI overrides)

### Extended (Recommended)

- [x] Dataset adapters (Dakshina, Aksharantar, CMTET-LID)
- [x] SentencePiece tokenizer support
- [x] Scheduled sampling support
- [x] Beam search with length penalty
- [x] Multiple experiment configs
- [x] Experiment runner script
- [x] Flutter mobile scaffold
- [ ] Real dataset integration (requires download)
- [ ] Attention visualization notebook
- [ ] Academic report (write based on results)

### Future

- [ ] Fine-tune ByT5/IndicBART
- [ ] Large-scale real dataset
- [ ] Cloud deployment
- [ ] ONNX export
- [ ] On-device inference
- [ ] User feedback loop

---

## Version History

| Version | Status | Description |
|---------|--------|-------------|
| v0.1 | ✅ Done | Dataset pipeline (seed corpus, noise, splits) |
| v0.2 | ✅ Done | Rule/dictionary baseline |
| v0.3 | ✅ Done | Basic Seq2Seq without attention |
| v0.4 | ✅ Done | Attention-based Seq2Seq (all 5 variants) |
| v0.5 | ✅ Done | Transformer comparison + experiment framework |
| v0.6 | ✅ Done | FastAPI server |
| v0.7 | ✅ Done | Web interface |
| v0.8 | ✅ Done | Chrome extension + mobile scaffold |
| v0.9 | 🔄 In Progress | Real dataset integration + optimization |
| v1.0 | 📋 Planned | Complete AAN mini-project release |
| v2.0 | 🔮 Future | Production version with pretrained models |
