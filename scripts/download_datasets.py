#!/usr/bin/env python3
"""Best-effort downloader for optional external datasets. Never required for the pipeline to run.

Dakshina (Google Research, Apache-2.0): includes Telugu romanization lexicons + full-sentence
romanized Wikipedia. Aksharantar (AI4Bharat, CC0): large native<->Roman word pairs for 21 languages.
Both are 100MB-few GB; on slow connections use --only dakshina or --only aksharantar.
See data/DATASETS.md for manual download instructions and exact expected paths if this fails
(e.g. no internet access in your environment / GitHub LFS quota / Hugging Face rate limits).
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"

# Multiple fallback URLs for Aksharantar (HuggingFace URLs change over time)
AKSHARANTAR_URLS = [
    "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/te.zip",
    "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/data/te.zip",
]


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


def get_dakshina(root: Path):
    tar_path = root / "dakshina_dataset_v1.0.tar"
    if not (root / "te").exists():
        if _download(DAKSHINA_URL, tar_path):
            print("Extracting (Telugu ('te') subset only)...")
            with tarfile.open(tar_path) as tf:
                members = [m for m in tf.getmembers() if "/te/" in m.name or m.name.endswith("/te")]
                tf.extractall(root, members=members)
            nested = root / "dakshina_dataset_v1.0"
            if nested.exists():
                for p in nested.iterdir():
                    p.rename(root / p.name)
                nested.rmdir()
            tar_path.unlink(missing_ok=True)
    else:
        print("Dakshina 'te' already present, skipping.")


def _try_hf_datasets_aksharantar(root: Path) -> bool:
    """Try downloading via the `datasets` library (pip install datasets). Most reliable method."""
    try:
        from datasets import load_dataset
        print("Downloading Aksharantar Telugu via HuggingFace `datasets` library...")
        ds = load_dataset("ai4bharat/Aksharantar", "te", trust_remote_code=True)
        out_dir = root / "te"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ds:
            out_path = out_dir / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in ds[split_name]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Saved {len(ds[split_name])} rows -> {out_path}")
        return True
    except ImportError:
        print("  `datasets` library not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"  HuggingFace datasets download failed: {e}")
        return False


def get_aksharantar(root: Path):
    te_dir = root / "te"
    if te_dir.exists() and any(te_dir.iterdir()):
        print("Aksharantar Telugu data already present, skipping.")
        return

    # Method 1: Try HuggingFace datasets library (most reliable)
    if _try_hf_datasets_aksharantar(root):
        return

    # Method 2: Try direct URL download (may fail if HF changes paths)
    zpath = root / "aksharantar_te.zip"
    for url in AKSHARANTAR_URLS:
        if _download(url, zpath):
            print("Download succeeded! Extracting...")
            import zipfile
            try:
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(root)
                zpath.unlink(missing_ok=True)
                return
            except Exception as e:
                print(f"  Extraction failed: {e}")

    # Method 3: Manual instructions
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED for Aksharantar:")
    print("  1. Visit: https://huggingface.co/datasets/ai4bharat/Aksharantar")
    print("  2. Download the Telugu ('te') subset")
    print(f"  3. Extract to: {root / 'te'}/")
    print("  4. Or install the datasets library: pip install datasets")
    print("     Then re-run this script.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external")
    ap.add_argument("--only", choices=["dakshina", "aksharantar"], default=None)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "dakshina"):
        get_dakshina(root / "dakshina")
    if args.only in (None, "aksharantar"):
        get_aksharantar(root / "aksharantar")
    print("\nDone. Re-run scripts/prepare_data.py --external", root, "to fold this into the dataset.")


if __name__ == "__main__":
    main()
