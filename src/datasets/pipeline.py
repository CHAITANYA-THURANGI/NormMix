"""End-to-end dataset build: seed + synthetic (+ optional external) -> cleaned, deduped, split JSONL."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from ..augmentation.noise import NoiseConfig, RomanizationSampler
from ..augmentation.synthetic import code_mix_from_telugu, composition, generate_pairs
from ..preprocessing.text_cleaning import clean_text
from ..utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from . import adapters
from .seed_corpus import build_clean_corpus, group_key
from .splits import dedupe_pairs, filter_noisy, leakage_report, resolve_conflicts, split_rows

GOLD_DEMO = [
    ("naku ivala college ki vellali", "నాకు ఇవాళ college కి వెళ్ళాలి"),
    ("nuvvu ekkada unnav", "నువ్వు ఎక్కడ ఉన్నావు?"),
    ("class ki vastunnava?", "class కి వస్తున్నావా?"),
    ("naku telugu chala istam", "నాకు తెలుగు చాలా ఇష్టం"),
]


def gold_demo_rows() -> list[dict]:
    return [{"src": s, "tgt": t, "source": "gold_demo", "group": group_key(t), "kind": "gold"} for s, t in GOLD_DEMO]


def build_dataset(cfg: dict, out_dir: str | Path | None = None, external_root: str | Path | None = None) -> dict:
    """Build processed splits. Returns a stats dict (also written to `stats.json`)."""
    dcfg = cfg["data"]
    out_dir = ensure_dir(out_dir or dcfg["processed_dir"])
    seed = cfg.get("seed", 42)
    noise = NoiseConfig(**dcfg.get("noise", {}))

    attested = None
    native_sents: list[str] = []
    ext = Path(external_root) if external_root else None
    extra_pairs: list[dict] = []
    if ext and (ext / "dakshina" / "te").exists():
        root = ext / "dakshina"
        try:
            lex = adapters.dakshina_lexicon(root, "te", "train")
            attested = adapters.attested_from_lexicon(lex)
            native_sents = adapters.dakshina_native_sentences(root, "te", "train", dcfg.get("max_wiki_sentences", 10000))
            dak_sents = adapters.dakshina_sentences(root, "te", dcfg.get("max_dakshina_sentences", 8000))
            extra_pairs += dak_sents
            print(f"[prepare_data] Loaded {len(dak_sents)} sentences from Dakshina")
        except FileNotFoundError as e:  # partial download is not fatal
            print(f"[prepare_data] Dakshina files incomplete, skipping some parts: {e}")

    if ext and (ext / "aksharantar" / "te").exists():
        ak_root = ext / "aksharantar" / "te"
        max_ak = dcfg.get("max_aksharantar_words", 12000)
        try:
            for fname in ["tel_valid.json", "tel_train.json"]:
                fpath = ak_root / fname
                if fpath.exists():
                    ak_words = adapters.aksharantar_words(fpath, max_rows=max_ak // 2)
                    extra_pairs += ak_words
                    print(f"[prepare_data] Loaded {len(ak_words)} word pairs from Aksharantar ({fname})")
                    if len(extra_pairs) > max_ak * 2:
                        break
        except Exception as e:
            print(f"[prepare_data] Aksharantar loading warning: {e}")
    sampler = RomanizationSampler(attested)

    clean = build_clean_corpus()
    if native_sents:
        clean += code_mix_from_telugu(native_sents, seed=seed, max_out=dcfg.get("max_wiki_codemix", 20000))
    pairs = generate_pairs(clean, variants=dcfg.get("variants_per_sentence", 8), noise=noise, sampler=sampler,
                           seed=seed, p_identity=dcfg.get("p_identity", 0.04))
    for r in extra_pairs:
        r["group"] = group_key(r["tgt"])
    pairs += extra_pairs

    for r in pairs:
        r["src"], r["tgt"] = clean_text(r["src"], dcfg.get("clamp_repeats", 3)), clean_text(r["tgt"], 0)
    pairs, n_filtered = filter_noisy(pairs, max_len=dcfg.get("max_chars", 300))
    pairs, dedupe_info = dedupe_pairs(pairs)
    pairs = resolve_conflicts(pairs, dcfg.get("conflict_policy", "majority"))

    gold = gold_demo_rows()
    splits = split_rows(pairs, dcfg.get("valid_frac", 0.1), dcfg.get("test_frac", 0.1),
                        heldout_words=dcfg.get("heldout_words", []), exclude_groups={g["group"] for g in gold})
    # robustness set: same test targets, doubled noise strength, different seed
    hard_targets = [{"tgt": r["tgt"], "group": r["group"], "template_id": r.get("template_id"), "slot": r.get("slot")}
                    for r in {r["group"]: r for r in splits["test"]}.values() if r.get("source") == "synthetic_seed"]
    splits["test_hard"] = generate_pairs(hard_targets, variants=2, noise=noise.scaled(2.0), sampler=sampler,
                                         seed=seed + 1, p_identity=0.0, source_name="synthetic_seed_hard")
    splits["gold_demo"] = gold

    rep = leakage_report({k: v for k, v in splits.items() if k != "gold_demo"})
    if not rep["ok"]:
        # test_hard reuses test groups (fine); only train-vs-eval overlap is fatal
        bad = {k: v for k, v in rep.items() if k not in ("ok",) and (v["shared_groups_with_train"] or v["shared_sources_with_train"])}
        if bad:
            raise RuntimeError(f"Train/eval leakage detected: {bad}")

    for name, rows in splits.items():
        for r in rows:
            r["comp"] = composition(r["tgt"])
        write_jsonl(Path(out_dir) / f"{name}.jsonl", rows)

    stats = {
        "n_pairs": {k: len(v) for k, v in splits.items()},
        "n_clean_targets": len(clean),
        "filtered_out": n_filtered,
        "dedupe": dedupe_info,
        "leakage": rep,
        "mix_distribution_train": dict(Counter(r["comp"]["mix"] for r in splits["train"])),
        "external_used": bool(ext and ext.exists()),
        "warning": "Seed data is a template-driven toy corpus; see docs/dataset.md before reporting results.",
    }
    write_json(Path(out_dir) / "stats.json", stats)
    return stats


def load_split(processed_dir: str | Path, name: str) -> list[dict]:
    p = Path(processed_dir) / f"{name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/prepare_data.py")
    return read_jsonl(p)
