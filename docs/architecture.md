# Architecture

## Task
**Input**: noisy, romanized and/or code-mixed Telugu-English text a user actually types
(`naku ivala college ki vellali`).
**Output**: normalized code-mixed text: Telugu words in Telugu script, English words in
lowercase Latin script, code-mixing structure preserved (`నాకు ఇవాళ college కి వెళ్ళాలి`).
This is normalization, not translation: English words are *kept as English* (not translated to
Telugu) and Telugu words are *rendered in Telugu script* (not transliterated to Roman) --
see `docs/dataset.md` for the exact target-format decision and its trade-offs.

## Pipeline
```
raw/synthetic text -> src/preprocessing (clean, protect URLs/mentions, langid)
                    -> src/augmentation (noise models simulate real typing patterns)
                    -> src/datasets (leakage-safe group-level train/valid/test split)
                    -> src/tokenization (char-level by default; SentencePiece optional)
                    -> src/models (RNN Seq2Seq with pluggable attention | Transformer)
                    -> src/training (teacher forcing / scheduled sampling, early stopping)
                    -> src/evaluation (CER/WER/chrF/BLEU + sliced + error taxonomy)
                    -> src/inference (batched beam search)
                    -> api/ (FastAPI) -> web/ (HTML/JS) + chrome-extension/ + mobile/
```

## Models (see `docs/experiments.md` for how to compare them)
1. **Rule/dictionary baseline** (`src/models/rule_baseline.py`) -- floor to beat, and a safe
   fallback when no neural checkpoint is trained yet.
2. **Plain Seq2Seq** (`model.attention: none`) -- encoder's final state is the *only* information
   the decoder gets; demonstrates the fixed-context bottleneck attention was invented to fix.
3. **Attention-based Seq2Seq** (the focus of this project) -- `src/attention/attention.py`
   implements Bahdanau (additive) and four Luong-family variants (dot / general / concat /
   scaled-dot-product) behind one interface (`make_attention`), so `model.attention` in
   `config.yaml` swaps the mechanism without touching training code. Both GRU and LSTM cells are
   supported (`model.rnn_type`).
4. **Transformer** (`model.type: transformer`) -- self-attention comparison point; same
   tokenizer/data/eval pipeline, so numbers are directly comparable.
5. *(Optional, not run in this repo)* a pretrained multilingual/Indic seq2seq model
   (e.g. ByT5/mT5/IndicBART) fine-tuned on this task -- see `docs/experiments.md#pretrained-model`
   for why it's scoped out of default CPU runs and how to add it.

## Attention, precisely
All variants return `(context, weights)`, with padding positions masked to `-inf` before softmax
(`src/attention/attention.py`):
  * **Bahdanau (additive)**: `e_ti = v^T tanh(W_q s_{t-1} + W_k h_i)` -- context computed from
    the *previous* decoder state, concatenated with the embedding and fed as extra decoder input
    (the original formulation).
  * **Luong dot / general / concat**: context computed from the decoder's *new* state
    `s_t`, then combined via "input feeding" (`h~_t = tanh(W[c_t; s_t])`, fed into the *next*
    step) -- the "global attention" family.
  * **Scaled dot-product**: Transformer-style `softmax(QK^T / sqrt(d)) V`, adapted to a
    single-query RNN decoder step for a fair architectural comparison.
Beam search (`src/models/decoding.py`) is model-agnostic: it only needs a `step_fn` closure, so
both the RNN and Transformer models share one implementation.

## Design decisions worth knowing about
* **Character-level tokenization by default.** Telugu is an abugida (conjunct consonant clusters,
  vowel signs, virama) and the input is full of unpredictable Latin spelling variants -- a
  closed word vocabulary would drown in `<unk>`. SentencePiece (unigram/BPE) is implemented and
  selectable (`tokenizer.type: unigram`) for experimentation, but needs more data to beat char-level
  here.
* **Leakage-safe splitting is by *clean target*, not by row** (`src/datasets/splits.py`): every
  noisy variant of the same sentence is deterministically hashed to one split, so train and
  test/valid never share a target sentence (`leakage_report()` asserts this and the pipeline
  raises if it's violated).
* **`test_unseen_words`** holds out sentences containing specific English words
  (`data.heldout_words` in `config.yaml`) entirely from train/valid, to measure open-vocabulary
  generalization -- the scenario a closed lexicon baseline handles worst.
* **`test_hard`** re-generates the *same* test-set target sentences at 2x noise strength with a
  different seed, to separate "hard because the sentence is inherently ambiguous" from "hard
  because the noise happened to be severe."
