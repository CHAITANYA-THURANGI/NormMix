#!/usr/bin/env python3
"""Packages the Chrome Extension into a clean distribution ZIP ready for:
1. Google Chrome Web Store Developer Dashboard upload
2. Direct GitHub Release distribution for users to install for free
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "chrome-extension"
OUT_ZIP = ROOT / "normmix-chrome-extension-v0.5.0.zip"


def main():
    print(f"Packaging Chrome Extension from: {EXT_DIR}")
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir, _, files in os.walk(EXT_DIR):
            for file in files:
                full_path = Path(root_dir) / file
                rel_path = full_path.relative_to(EXT_DIR)
                zf.write(full_path, str(rel_path))
                count += 1
                print(f"  + Added: {rel_path}")

    size_kb = OUT_ZIP.stat().st_size / 1024
    print(f"\n[OK] Package created successfully with {count} files!")
    print(f"Location: {OUT_ZIP}")
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
