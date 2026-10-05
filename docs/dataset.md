# Dataset: what this project actually trains on

**Read this before quoting any number from `experiments/results/`.**

## The honest summary
By default, this project trains on a **synthetic, template-driven toy corpus**
(`src/datasets/seed_corpus.py`): ~25 sentence templates x a handful of slot values (places,
events, objects...) plus ~38 hand-written free sentences, giving ~197 distinct clean Telugu-English
target sentences. Each is expanded into several *noisy* input variants via
`src/augmentation/noise.py` (typos, abbreviations, elongation, romanization variation, punctuation
noise, suffix-merging like `collegeki`). No claim is made that this represents the true
distribution of real Telugu-English code-mixed social media / chat text, and results on it should
be read as **pipeline validation**, not as a benchmark of real-world performance.

## Why this default, instead of stopping to find "the" dataset
No single, license-clear, publicly downloadable **(noisy code-mixed text, normalized target)**
parallel corpus for Telugu-English was found during a real search of the literature (RANLP/ACL
code-mixing papers, AI4Bharat, HuggingFace) at the time this project was built -- normalization
*targets* (as opposed to language-ID tags, or clean monolingual text, or transliteration word
pairs) for code-mixed Indian-language text are not a solved, standardized benchmark. Rather than
invent one and pretend it's real, this project:
1. Says so explicitly (this file).
2. Builds a small but *inspectable, deterministic, and reproducible* synthetic corpus, so the full
   pipeline (data -> train -> eval -> serve) is genuinely exercised end-to-end.
3. Documents exactly how to fold in real resources that *do* exist and *are* verifiable
   (`data/DATASETS.md`: Dakshina, Aksharantar, CMTET-LID) to make the training distribution more
   realistic, without requiring them for the code to run.
4. Ships the tooling (`src/datasets/adapters.py`, `src/datasets/pipeline.py`) so plugging in a
   better dataset later -- including one you annotate yourself -- is a small adapter function, not
   a rewrite.

## Target-format decision (and its trade-offs)
The normalization target keeps Telugu words in Telugu script and English words in lowercase Latin
(`నాకు ఇవాళ college కి వెళ్ళాలి`), with a Telugu case-marker glued onto an English word split into
its own token (`college కి`, not `collegeki` and not `collegeకి`). This was chosen because:
* It matches how fluent Telugu-English typists actually write when *not* constrained to one
  script (mixed-script chat messages), which is a more information-preserving normalization than
  forcing everything into one script.
* It keeps English named entities/borrowings recognizable (a "translate everything to Telugu"
  target would lose information: "college" and "కళాశాల" are not always interchangeable in how
  people use them).
* Trade-off: it requires the model to make a script decision per word, not just a spelling
  decision -- this is exactly the language-boundary problem `src/preprocessing/langid.py` and the
  `english_as_telugu` / `telugu_left_romanized` error categories in
  `src/evaluation/error_analysis.py` are built to detect and measure.
An "everything in Telugu script" or "everything romanized" convention is a one-line change to
`seed_corpus.py`'s templates if your use case prefers it -- the rest of the pipeline doesn't
assume the current convention.

## Known limitations of the synthetic corpus (and their evaluation-time symptoms)
* **Closed, small vocabulary.** ~40 English content words and ~50 Telugu grammatical/content
  words recur across templates. This inflates scores for *any* lookup-capable method (including
  the rule baseline) relative to genuinely open-vocabulary text, and is precisely why
  `test_unseen_words` exists -- read that slice's numbers, not just the aggregate `test` numbers.
* **Template-shaped syntax.** Real code-mixed sentences have far more varied structure. The
  Dakshina-Wikipedia code-mixing augmentation (`code_mix_from_telugu`, active when
  `data/external/dakshina` is downloaded) helps but is itself a heuristic (naive noun substitution
  with a ~20-word bilingual seed lexicon), not real bilingual authorship.
* **Rule-based noise, not human noise.** `src/augmentation/noise.py`'s typo/abbreviation models
  are plausible but hand-designed; real typos, autocorrect artifacts, and keyboard-specific errors
  (e.g. Telugu phonetic keyboards) will differ in ways this corpus doesn't capture.
* **Small size.** A few thousand pairs after augmentation is enough to validate the pipeline on a
  laptop CPU in minutes, not enough to expect strong generalization from the neural models --
  see `docs/experiments.md` for what was actually observed and why, and for the (straightforward)
  path to more data via `data/DATASETS.md`.

## Splits (all in `data/processed/*.jsonl` after `python scripts/prepare_data.py`)
| Split | Purpose |
|---|---|
| `train` / `valid` / `test` | Standard 80/10/10 **by clean-target group** (never by row) |
| `test_unseen_words` | Targets containing English words never seen in train/valid (`config.yaml: data.heldout_words`) |
| `test_hard` | Same test-set targets, regenerated at 2x noise strength, different seed |
| `gold_demo` | The 4 example sentences from the project brief, held out from all training data |
