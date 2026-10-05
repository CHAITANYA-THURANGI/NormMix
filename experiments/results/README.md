# Results

`sample_results.json` is the output of `scripts/evaluate.py` run in this project's development
sandbox (single CPU core, no GPU) against the `attn_bahdanau` and `plain_seq2seq` runs described
in `experiments/checkpoints/README.md`, plus the rule/dictionary baseline. It's the source for the
numbers discussed in `docs/experiments.md` -- read that file for the interpretation, not just the
raw JSON here.

Regenerate it (after retraining, since checkpoints aren't shipped -- see
`experiments/checkpoints/README.md`) with:
```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/attn_bahdanau experiments/checkpoints/plain_seq2seq \
  --rule-baseline data/processed/rule_baseline.json --split test test_hard test_unseen_words gold_demo \
  --errors --out experiments/results/sample_results.json
```

For a full sweep across every config in `experiments/configs/`, use
`python scripts/run_all_experiments.py` instead, which writes `comparison.json`.
