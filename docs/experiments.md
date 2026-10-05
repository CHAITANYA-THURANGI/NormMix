# Experiments: how to run them, and what was actually observed

## Reproduce from scratch
```bash
pip install -r requirements.txt
python scripts/prepare_data.py            # builds data/processed/*.jsonl (synthetic corpus)
python scripts/train_rule_baseline.py      # fits + saves the rule/dictionary baseline
python scripts/run_all_experiments.py --epochs 40   # trains every config in experiments/configs/,
                                                      # then writes experiments/results/comparison.json
```
Each config in `experiments/configs/*.yaml` only overrides `model` / `train` keys on top of
`config.yaml`'s defaults; `run_name` there becomes the checkpoint directory name. Compare a
specific pair yourself:
```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/attn_bahdanau experiments/checkpoints/plain_seq2seq \
  --rule-baseline data/processed/rule_baseline.json --split test test_unseen_words test_hard gold_demo --errors \
  --out experiments/results/compare.json
```

## What was actually observed in this repository (CPU, char tokenizer, synthetic corpus)
Real numbers from a run inside this sandbox: single CPU core, no GPU, char tokenizer, GRU+Bahdanau
attention, `hid_dim=256`/`emb_dim=128`, early-stopped at epoch 12/26 (~5 min wall clock);
plain Seq2Seq (no attention) same size, early-stopped at epoch 9/19 (~2 min). Full breakdown in
`experiments/results/comparison_final.json`. **Not** a claim about real-world performance -- see
`docs/dataset.md` for why (small synthetic, largely closed-vocabulary corpus).

| Model | test EM | test CER | test_unseen_words EM | test_unseen_words CER | gold_demo EM |
|---|---:|---:|---:|---:|---:|
| Rule/dictionary baseline | 0.714 | 0.032 | 0.199 | 0.106 | 0.250 |
| **Attention Seq2Seq (Bahdanau)** | **0.579** | **0.140** | 0.000 | 0.249 | 0.250 |
| Plain Seq2Seq (no attention) | 0.018 | 0.406 | 0.000 | 0.340 | 0.000 |

Two findings, both worth reporting rather than only the flattering one:

**1. Attention decisively beats no attention.** CER 0.140 vs 0.406, EM 0.579 vs 0.018, on the same
data/tokenizer/training budget, changing only `model.attention: bahdanau` vs `none`. This is the
information bottleneck argument made concrete: a plain Seq2Seq compresses the whole source
sentence into one fixed-size vector, and even at this small scale that costs the model dearly (it
degrades badly on `gold_demo`'s longer sentences: EM 0.000, CER 0.634). Bahdanau attention lets the
decoder look back at every encoder position, and that alone recovers most of the gap to the rule
baseline on `test` (from 0.406 to 0.140 CER) and closes it almost entirely on `test_hard`.

**2. The rule/dictionary baseline still leads on raw accuracy, and wins outright on unseen words.**
The corpus is built from ~25 templates over a few dozen slot values (`docs/dataset.md`), so it has
a small, largely *closed* vocabulary that a token-level lookup table can near-memorize from very
few examples. On `test_unseen_words` specifically (English words never seen in training), the
attention model scores **EM 0.000** -- inspecting `error_analysis` output shows *why*:
given `nenu repu mall కి welthanu` (a novel word, "mall"), the model outputs
`నేను రేపు మాట్లా వస్తాను` -- it has never learned "when a Latin token doesn't match anything
memorized, just copy it through unchanged," so it instead maps the unfamiliar spelling to the
closest *Telugu* word it does know (`మాట్లా`, from "మాట్లాడు"/talk). The rule baseline handles
this case far better (EM 0.199, CER 0.106) via its fuzzy-match/suffix-split fallback, precisely
because open-vocabulary copy-through is a design feature of a lookup approach and something a
from-scratch neural model needs *many more examples* (or, more reliably, a copy mechanism, or
pretraining) to learn implicitly.

This is exactly the scenario the research literature predicts for low-resource
transliteration/normalization: neural sequence models overtake rule/lookup baselines once the
training set is substantially larger (real corpora -- `data/DATASETS.md`: Dakshina, Aksharantar)
and/or initialized from pretraining (see below), not from a few thousand synthetic pairs alone.

**Report all three numbers in your write-up, not just the neural model's** -- "attention beats no
attention decisively; the rule baseline still wins on this toy corpus, especially on unseen words,
and here's the specific failure mode and what would fix it" is a more defensible piece of analysis
than only reporting the neural model's own score.

## Attention mechanism ablation
All six attention configurations (`none`, `bahdanau`, `luong_dot`, `luong_general`,
`luong_concat`, `scaled_dot`) train and decode correctly (see `tests/test_models_forward.py` and
`tests/test_attention_shapes.py` for the shape/normalization guarantees checked), and the
`bahdanau` vs `none` comparison above is measured directly in this repository. Comparing the
*Luong variants against each other* meaningfully needs more data/epochs than this CPU smoke test
budgeted -- run `python scripts/run_all_experiments.py` with a larger `--epochs` and
real/expanded data, then inspect `experiments/results/comparison.json`; differences between
attention *variants* (as opposed to attention vs none) are typically much smaller than the
none-vs-any-attention gap shown above.

## Error analysis
`src/evaluation/error_analysis.py` buckets every mismatched token into one of: English spelling
error, English wrongly rendered in Telugu script, Telugu left un-transliterated, Telugu character
error, Telugu vocabulary error, insertion, deletion, or punctuation/emoji-only, plus
`unknown_word` / `hallucinated` flags against the training vocabulary. Run with `--errors` on
`scripts/evaluate.py`; `result["errors"]["examples"]` gives up to 4 concrete (source, reference,
hypothesis) triples per category for qualitative discussion in a report.

## Robustness (`test_hard`)
Re-generates the test set's target sentences at 2x the default noise strength (a different random
seed, so it's not simply duplicate data). Comparing `test` vs `test_hard` numbers for the same
model isolates *robustness to noise severity* from *raw task difficulty*.

## Pretrained model (optional, not run in this repo)
Fine-tuning a pretrained multilingual/Indic seq2seq checkpoint (candidates: `google/byt5-small`
for a script-agnostic byte-level model with no Telugu-specific tokenizer needed, or an
AI4Bharat IndicBART/IndicTrans2 checkpoint if you want native Telugu subword pretraining -- check
current model cards on Hugging Face for licenses/sizes before choosing) is a natural extension:
low-resource sequence tasks typically benefit substantially from pretraining. It's scoped out of
the default pipeline here because (a) it needs a GPU and a download of a few hundred MB-1GB+ to be
worth running at all, which conflicts with "workable on free/low-cost compute," and (b) it's best
fine-tuned on the real external data (`data/DATASETS.md`) rather than the small synthetic corpus,
where a large pretrained model would simply overfit or gain little. To add it: write a thin
wrapper implementing the same `.generate(...)` interface as `src/models/rnn_seq2seq.py` (HF
`generate()` underneath), a config entry (`model.type: pretrained`), and reuse
`src/evaluation/evaluate.py` unchanged.
