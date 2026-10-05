# Pretrained Models (Optional — Future Work)

These models are **not required** for the core AAN mini-project. The project's Seq2Seq + Attention
model trains from scratch on the seed corpus. These pretrained models are listed for **future
experiments** where fine-tuning may significantly improve performance.

## Recommended Models

### 1. ByT5-small (Google)
- **HuggingFace ID**: `google/byt5-small`
- **Parameters**: ~300M
- **Languages**: 100+ (byte-level, no tokenizer needed)
- **License**: Apache-2.0
- **Why**: Byte-level processing handles noisy/code-mixed text exceptionally well. No OOV issues.
- **Fine-tuning**: Treat normalization as a seq2seq task. Input = noisy code-mixed text,
  output = normalized text. Fine-tune with `transformers.Seq2SeqTrainer`.
- **GPU requirement**: T4 (16GB) minimum; A100 recommended for faster training.

### 2. IndicBART (AI4Bharat)
- **HuggingFace ID**: `ai4bharat/IndicBART`
- **Parameters**: ~244M
- **Languages**: 11 Indic languages + English
- **License**: MIT
- **Why**: Pre-trained on Indic languages with denoising objectives. Natural fit for normalization.
- **Fine-tuning**: Same seq2seq approach. The denoising pretraining objective aligns well with
  normalization (input = corrupted text, output = clean text).
- **GPU requirement**: T4 (16GB) sufficient.

### 3. IndicBERT (AI4Bharat)
- **HuggingFace ID**: `ai4bharat/indic-bert`
- **Parameters**: ~18M (ALBERT architecture)
- **Languages**: 12 Indic languages
- **License**: MIT
- **Why**: Lightweight encoder for downstream tasks like language identification.
- **Use case**: Encoder-only (not seq2seq). Useful for token-level language ID, not normalization.
- **GPU requirement**: Runs on CPU.

### 4. mT5-small (Google)
- **HuggingFace ID**: `google/mt5-small`
- **Parameters**: ~300M
- **Languages**: 101 languages (SentencePiece tokenizer)
- **License**: Apache-2.0
- **Why**: Multilingual T5 with good Telugu support. Subword tokenization.
- **Fine-tuning**: Similar to ByT5 but uses subword tokens (may struggle with noisy spellings).
- **GPU requirement**: T4 (16GB) sufficient.

### 5. L3Cube Telugu-BERT (L3Cube-Pune)
- **HuggingFace ID**: `l3cube-pune/telugu-bert`
- **Parameters**: ~110M
- **Languages**: Telugu
- **License**: Check repository
- **Why**: Telugu-specific BERT. Good for Telugu-only tasks but not seq2seq generation.
- **Use case**: Embeddings, classification. Not directly applicable to normalization.

## How to Download (when ready for fine-tuning)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Example: ByT5
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# Example: IndicBART
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
```

## Storage Requirements

| Model | Download Size | GPU Memory (inference) | GPU Memory (fine-tuning) |
|-------|--------------|----------------------|------------------------|
| ByT5-small | ~1.2 GB | ~2 GB | ~8-12 GB |
| IndicBART | ~1.0 GB | ~1.5 GB | ~6-10 GB |
| IndicBERT | ~70 MB | ~200 MB | ~2 GB |
| mT5-small | ~1.2 GB | ~2 GB | ~8-12 GB |
