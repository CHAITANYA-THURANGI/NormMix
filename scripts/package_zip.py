#!/usr/bin/env python3
"""Packages the project into a clean distribution ZIP archive.

Excludes virtual environments (.venv), Git metadata, cached files,
large external datasets (IMDb, Dakshina raw dumps), and heavy binary model weights (.pt).
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "telugu-english-code-mixed-normalization-project.zip"

EXCLUDE_DIRS = {".venv", ".venv-gpu", ".git", "__pycache__", ".pytest_cache", "external", "checkpoints", "archive (17)"}
SKIP_EXTS = {".pt", ".pyc", ".zip", ".tar", ".gz", ".tsv"}


def main():
    print(f"Packaging {ROOT} -> {DEST}", flush=True)
    file_count = 0
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Prune excluded directories in-place
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith("archive")]
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in SKIP_EXTS or f.startswith("archive"):
                    continue
                full = Path(dirpath) / f
                # Skip any single file > 2MB just in case
                if full.stat().st_size > 2 * 1024 * 1024:
                    continue
                rel = full.relative_to(ROOT)
                zf.write(full, str(rel))
                file_count += 1

    size_kb = os.path.getsize(DEST) / 1024
    print(f"Success! Clean ZIP package created with {file_count} files.")
    print(f"Location: {DEST}")
    print(f"Size: {size_kb:,.1f} KB ({size_kb / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
