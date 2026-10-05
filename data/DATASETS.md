# External datasets (optional)

The pipeline runs fully **without** any of these -- `scripts/prepare_data.py` builds a synthetic
seed corpus from templates + hand-written sentences (see `src/datasets/seed_corpus.py` and
`src/augmentation/`). This is what lets everything (data -> train -> eval -> API -> web -> Chrome
extension) work out of the box. The datasets below make the synthetic data more realistic and
larger; **none of their files are included in this ZIP** (they are hundreds of MB to a few GB and
carry their own licenses/redistribution terms), so download them yourself if you want to use them.

## Dakshina (Google Research) -- recommended first
Native-script and romanized text + word-level transliteration lexicons for 12 South Asian
languages, Telugu (`te`) included. Apache License 2.0.
  * Paper: Roark et al., 2020, "Processing South Asian Languages Written in the Latin Script..."
  * Repo / download: https://github.com/google-research-datasets/dakshina
  * Direct tarball: https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar (~2GB, all 12 languages)
  * Used here for: `dakshina_lexicon()` -> attested Telugu word romanizations (feeds
    `RomanizationSampler` for less arbitrary synthetic noise) and `dakshina_native_sentences()` ->
    clean Telugu Wikipedia sentences (fed into `code_mix_from_telugu()` for more code-mixing
    variety than the ~40 hand-written sentences in the seed corpus).
  * Expected layout after download: `data/external/dakshina/te/{lexicons,romanized,native_script_wikipedia}/...`
    (matches the paths `src/datasets/adapters.py` reads; `scripts/download_datasets.py --only dakshina`
    fetches and extracts just the `te/` subset automatically, network permitting).

## Aksharantar (AI4Bharat) -- optional, more word coverage
21.7M native-Roman word pairs across 21 Indic languages (CC0). Larger and more diverse than
Dakshina's lexicon, useful for enriching `RomanizationSampler` further.
  * Repo card: https://huggingface.co/datasets/ai4bharat/Aksharantar
  * Paper: Madhani et al., 2023 (ACL Findings), "Aksharantar: Open Indic-language Transliteration
    datasets and models for the Next Billion Users"
  * Expected layout: `data/external/aksharantar/te.json` (or the unzipped per-language JSONL the
    dataset viewer provides) -- read by `src/datasets/adapters.py::aksharantar_words()`.

## CMTET-LID -- real noisy Telugu-English text (source-side only, no normalized targets)
A community-annotated word-level language-ID corpus of real Telugu-English code-mixed social
media text (TE/EN/NE/UNIV tags). **It has no clean "normalized" target**, so it cannot directly
extend the (source, target) training pairs -- but it is valuable as (a) real noisy input for
qualitative testing / the API demo, and (b) ground truth for evaluating the language-ID module in
`src/preprocessing/langid.py` on real data instead of only synthetic pairs.
  * Repo: https://github.com/ksubbu199/cmtet-lid (check the repo's own license file before reuse)
  * Read with `src/datasets/adapters.py::cmtet_lid()`.

## If you have your own labeled data
Any TSV/JSONL of (noisy_text, normalized_text) pairs can be folded in directly: write an adapter
in `src/datasets/adapters.py` following the existing ones (return
`{"src": ..., "tgt": ..., "source": "your_name"}` dicts), add it to
`src/datasets/pipeline.py::build_dataset()`, and it will flow through cleaning, dedup, leakage-safe
splitting and stats automatically.

## Why nothing is invented
No fabricated "Telugu-English code-mixed benchmark" is bundled or referenced as if it were a
standard dataset -- see `docs/dataset.md` for exactly what data this project trains on by default
and its known limitations.
