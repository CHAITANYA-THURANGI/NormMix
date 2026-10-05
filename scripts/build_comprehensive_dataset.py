#!/usr/bin/env python3
"""Build a comprehensive training and validation dataset for SOTA Seq2Seq normalization.

Merges:
1. Google Dakshina Dataset (10,000 human-annotated Romanized Telugu to Native Telugu sentences)
2. Google Dakshina Lexicon (15,000 attested word-level transliterations)
3. AI4Bharat Aksharantar (15,000 curated word-level pairs)
4. NormMix Code-Mixed Corpus (25,803 synthetic and colloquial code-mixed Telugu-English pairs)
5. Pure Telugu (అచ్చ తెలుగు) pairs & conversational idioms
6. Dakshina Dev & NormMix Valid splits for comprehensive validation.

Outputs:
- data/processed/train_comprehensive.jsonl
- data/processed/valid_comprehensive.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean(s: str) -> str:
    return " ".join(s.strip().split())


def build_comprehensive_dataset():
    random.seed(42)
    train_rows = []
    valid_rows = []

    # ── 1. NormMix Existing Processed Code-Mixed Pairs ──────────────────────────
    base_train = ROOT / "data" / "processed" / "train.jsonl"
    if base_train.exists():
        with open(base_train, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    train_rows.append({
                        "src": clean(item["src"]),
                        "tgt": clean(item["tgt"]),
                        "source": item.get("source", "normmix_codemix"),
                        "kind": item.get("kind", "code_mixed")
                    })
        print(f"[1] Loaded {len(train_rows)} base code-mixed pairs from {base_train}")

    # ── 2. Google Dakshina Full Sentences ───────────────────────────────────────
    dak_sent_tsv = ROOT / "data" / "external" / "dakshina_dataset_v1.0" / "te" / "romanized" / "te.romanized.rejoined.tsv"
    dak_count = 0
    if dak_sent_tsv.exists():
        with open(dak_sent_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    tgt, src = clean(parts[0]), clean(parts[1])
                    if tgt and src and len(src) < 250 and len(tgt) < 250:
                        train_rows.append({
                            "src": src,
                            "tgt": tgt,
                            "source": "dakshina_sentence",
                            "kind": "roman_to_telugu_sentence"
                        })
                        dak_count += 1
        print(f"[2] Loaded {dak_count} full sentence pairs from Google Dakshina ({dak_sent_tsv.name})")

    # ── 3. Google Dakshina Word Lexicon (Attested Count >= 2) ───────────────────
    dak_lex_tsv = ROOT / "data" / "external" / "dakshina_dataset_v1.0" / "te" / "lexicons" / "te.translit.sampled.train.tsv"
    dak_lex_count = 0
    if dak_lex_tsv.exists():
        candidates = []
        with open(dak_lex_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    count = int(parts[2]) if parts[2].isdigit() else 1
                    if count >= 2:
                        tgt, src = clean(parts[0]), clean(parts[1])
                        if tgt and src:
                            candidates.append((src, tgt))
        random.shuffle(candidates)
        for src, tgt in candidates[:15000]:
            train_rows.append({
                "src": src,
                "tgt": tgt,
                "source": "dakshina_lexicon",
                "kind": "word_translit"
            })
            dak_lex_count += 1
        print(f"[3] Loaded {dak_lex_count} high-confidence word pairs from Dakshina Lexicon")

    # ── 4. AI4Bharat Aksharantar Transliteration Pairs ──────────────────────────
    ak_json = ROOT / "data" / "external" / "aksharantar" / "te" / "tel_train.json"
    ak_count = 0
    if ak_json.exists():
        candidates = []
        with open(ak_json, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        src, tgt = clean(d.get("english word", "")), clean(d.get("native word", ""))
                        if src and tgt and len(src) < 50:
                            candidates.append((src, tgt))
                    except Exception:
                        continue
                if len(candidates) >= 60000:
                    break
        random.shuffle(candidates)
        for src, tgt in candidates[:15000]:
            train_rows.append({
                "src": src,
                "tgt": tgt,
                "source": "aksharantar",
                "kind": "word_translit"
            })
            ak_count += 1
        print(f"[4] Loaded {ak_count} word pairs from AI4Bharat Aksharantar")

    # ── 5. Pure Telugu (అచ్చ తెలుగు) Pairs ──────────────────────────────────────
    pure_dict_path = ROOT / "data" / "processed" / "pure_telugu_dictionary.json"
    pure_count = 0
    if pure_dict_path.exists():
        with open(pure_dict_path, "r", encoding="utf-8") as f:
            pure_data = json.load(f)
            loanwords = pure_data.get("loanwords", {})
            for eng, forms in loanwords.items():
                if isinstance(forms, dict):
                    base = forms.get("base", "")
                    if base:
                        train_rows.append({"src": eng, "tgt": base, "source": "pure_telugu_dict", "kind": "pure_telugu"})
                        pure_count += 1
                    for inflect_key in ["in", "to", "from", "with"]:
                        inflect_val = forms.get(inflect_key, "")
                        if inflect_val:
                            # e.g. "college lo" -> "కళాశాలలో"
                            train_rows.append({
                                "src": f"{eng} {inflect_key}",
                                "tgt": inflect_val,
                                "source": "pure_telugu_dict",
                                "kind": "pure_telugu_inflection"
                            })
                            pure_count += 1
        print(f"[5] Loaded {pure_count} Pure Telugu dictionary pairs")

    # ── 6. Conversational Tanglish & Pure Telugu Sentence Templates ─────────────
    conversational_templates = [
        ("nenu repu pusthaka bhandagaraniki velthanu", "నేను రేపు పుస్తక భాండాగారానికి వెళ్తాను"),
        ("naku me sahayam chala avasaram", "నాకు మీ సహాయం చాలా అవసరం"),
        ("aayana karyalayaniki vellaru", "ఆయన కార్యాలయానికి వెళ్లారు"),
        ("eppudu bhojanam cheddam?", "ఎప్పుడు భోజనం చేద్దాం?"),
        ("meeru elaa unnaru?", "మీరు ఎలా ఉన్నారు?"),
        ("nenu kshemanga unnanu", "నేను క్షేమంగా ఉన్నాను"),
        ("dhanyavadhamulu andi", "ధన్యవాదములు అండి"),
        ("idi chala manchi vishayam", "ఇది చాలా మంచి విషయం"),
        ("vidyarthulu pathasala lo unnaru", "విద్యార్థులు పాఠశాలలో ఉన్నారు"),
        ("chikitsalayam ekkada undi?", "చికిత్సాలయం ఎక్కడ ఉంది?"),
        ("nenu intiki velthunnanu", "నేను ఇంటికి వెళ్తున్నాను"),
        ("ivvala weather chala bagundi", "ఇవ్వాళ weather చాలా బాగుంది"),
        ("meeting eppudu start avuthundi?", "meeting ఎప్పుడు start అవుతుంది?"),
        ("project submit chesava?", "project submit చేశావా?"),
        ("naku ee topic chala istam", "నాకు ఈ topic చాలా ఇష్టం"),
    ]
    for src, tgt in conversational_templates:
        for _ in range(5):  # Re-sample to reinforce essential conversational anchors
            train_rows.append({
                "src": src,
                "tgt": tgt,
                "source": "conversational_anchors",
                "kind": "conversational_gold"
            })

    # ── Deduplicate Train Rows ──────────────────────────────────────────────────
    seen = set()
    deduped_train = []
    for r in train_rows:
        key = (r["src"].lower(), r["tgt"])
        if key not in seen and r["src"] and r["tgt"]:
            seen.add(key)
            deduped_train.append(r)
    random.shuffle(deduped_train)
    print(f"Total Unique Train Rows: {len(deduped_train)}")

    # ── 7. Validation Data ──────────────────────────────────────────────────────
    base_valid = ROOT / "data" / "processed" / "valid.jsonl"
    if base_valid.exists():
        with open(base_valid, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    valid_rows.append({
                        "src": clean(item["src"]),
                        "tgt": clean(item["tgt"]),
                        "source": item.get("source", "normmix_valid"),
                        "kind": item.get("kind", "code_mixed")
                    })

    # Add Dakshina dev sentences
    dak_dev_roman = ROOT / "data" / "external" / "dakshina_dataset_v1.0" / "te" / "romanized" / "te.romanized.rejoined.dev.roman.txt"
    dak_dev_native = ROOT / "data" / "external" / "dakshina_dataset_v1.0" / "te" / "romanized" / "te.romanized.rejoined.dev.native.txt"
    if dak_dev_roman.exists() and dak_dev_native.exists():
        with open(dak_dev_roman, "r", encoding="utf-8") as f_r, open(dak_dev_native, "r", encoding="utf-8") as f_n:
            dev_r = [line.strip() for line in f_r]
            dev_n = [line.strip() for line in f_n]
            for s, t in zip(dev_r[:1500], dev_n[:1500]):
                if s and t and len(s) < 250 and len(t) < 250:
                    valid_rows.append({
                        "src": clean(s),
                        "tgt": clean(t),
                        "source": "dakshina_dev",
                        "kind": "roman_to_telugu_sentence"
                    })

    seen_valid = set()
    deduped_valid = []
    for r in valid_rows:
        key = (r["src"].lower(), r["tgt"])
        if key not in seen_valid and key not in seen and r["src"] and r["tgt"]:
            seen_valid.add(key)
            deduped_valid.append(r)
    random.shuffle(deduped_valid)
    print(f"Total Unique Valid Rows: {len(deduped_valid)}")

    # ── Write Outputs ───────────────────────────────────────────────────────────
    out_train = ROOT / "data" / "processed" / "train_comprehensive.jsonl"
    out_valid = ROOT / "data" / "processed" / "valid_comprehensive.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for r in deduped_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_valid, "w", encoding="utf-8") as f:
        for r in deduped_valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Successfully wrote {len(deduped_train)} rows to {out_train}")
    print(f"✅ Successfully wrote {len(deduped_valid)} rows to {out_valid}")


if __name__ == "__main__":
    build_comprehensive_dataset()
