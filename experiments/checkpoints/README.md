# Checkpoints

Trained model weights (`.pt` files) are **not included** in this ZIP by design (see the project
brief / `README.md`) -- they are reproducible in a few minutes on a CPU and would otherwise bloat
the deliverable. What *is* kept here, per run, is the lightweight metadata that documents what was
actually run and what it achieved:

* `config.json` -- the fully-resolved config (defaults + experiment YAML + CLI overrides) used
* `result.json` -- `best_epoch` and `best_score` (the early-stopping monitor value) reached

`attn_bahdanau/` and `plain_seq2seq/` correspond to the runs discussed in `docs/experiments.md`
and scored in `experiments/results/sample_results.json`.

To regenerate the actual weights:
```bash
python scripts/prepare_data.py
python scripts/train_rule_baseline.py
python scripts/train.py --config experiments/configs/attn_bahdanau.yaml
python scripts/train.py --config experiments/configs/plain_seq2seq.yaml
```
