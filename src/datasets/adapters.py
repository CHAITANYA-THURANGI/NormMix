"""Adapters turning external resources into the unified pair schema.

Unified row: {"src": str, "tgt": str, "source": str, "task": str, ...}
Formats below follow the official READMEs (Dakshina, Aksharantar cards, CMTET-LID README).
Nothing here downloads data; see scripts/download_datasets.py and DATASET_DOWNLOAD_REQUIRED.md.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..preprocessing.text_cleaning import clean_text


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def dakshina_sentences(root: str | Path, lang: str = "te", max_rows: int | None = None) -> list[dict]:
    """Dakshina full-sentence romanization: `<lang>/romanized/<lang>.romanized.rejoined.tsv` (native<TAB>roman).

    NOTE: monolingual Telugu (Wikipedia) -> teaches Roman->Telugu script transliteration, NOT code-mixing.
    """
    p = Path(root) / lang / "romanized" / f"{lang}.romanized.rejoined.tsv"
    rows = []
    with _open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            native, roman = clean_text(parts[0]), clean_text(parts[1])
            if native and roman:
                rows.append({"src": roman, "tgt": native, "source": "dakshina", "task": "sentence_translit"})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def dakshina_native_sentences(root: str | Path, lang: str = "te", split: str = "train", max_rows: int | None = None) -> list[str]:
    """Native-script Wikipedia strings (clean Telugu) for synthetic code-mix generation."""
    p = Path(root) / lang / "native_script_wikipedia" / f"{lang}.wiki-filt.{split}.text.shuf.txt.gz"
    out = []
    with _open(p) as f:
        for line in f:
            s = clean_text(line)
            if s:
                out.append(s)
            if max_rows and len(out) >= max_rows:
                break
    return out


def dakshina_lexicon(root: str | Path, lang: str = "te", split: str = "train") -> list[dict]:
    """`<lang>/lexicons/<lang>.translit.sampled.<split>.tsv`: native<TAB>roman<TAB>count."""
    p = Path(root) / lang / "lexicons" / f"{lang}.translit.sampled.{split}.tsv"
    rows = []
    with _open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                cnt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                rows.append({"src": parts[1].strip(), "tgt": parts[0].strip(), "source": "dakshina_lexicon",
                             "task": "word_translit", "weight": cnt})
    return rows


def attested_from_lexicon(rows: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """native word -> [(attested romanization, count)] for RomanizationSampler."""
    d: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        d.setdefault(r["tgt"], []).append((r["src"], float(r.get("weight", 1))))
    return d


def aksharantar_words(path: str | Path, min_score: float | None = None, max_rows: int | None = None) -> list[dict]:
    """Aksharantar JSONL: {"native word", "english word", "source", "score", ...} (Telugu file)."""
    rows = []
    with _open(Path(path)) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            score = d.get("score")
            if min_score is not None and score is not None and score < min_score:
                continue
            rows.append({"src": d["english word"], "tgt": d["native word"], "source": "aksharantar",
                         "task": "word_translit", "origin": d.get("source")})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def cmtet_lid(directory: str | Path) -> list[dict]:
    """CMTET-LID: each line `word<whitespace>TAG`, blank line ends a sentence (per repo README).

    Returns sentence rows {"tokens": [...], "tags": [...], "text": "..."}: real noisy Telugu-English text
    (a source-side pool + LID evaluation data). There are NO normalized targets in this dataset.
    """
    rows = []
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file() or p.suffix.lower() in {".md", ".json", ".py"}:
            continue
        toks, tags = [], []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in list(f) + [""]:
                line = line.strip()
                if not line:
                    if toks:
                        rows.append({"tokens": toks, "tags": tags, "text": " ".join(toks), "file": p.name})
                    toks, tags = [], []
                    continue
                parts = line.split()
                toks.append(parts[0])
                tags.append(parts[-1] if len(parts) > 1 else "UNK")
    return rows
