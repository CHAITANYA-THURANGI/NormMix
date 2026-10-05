"""Conservative, information-preserving text cleaning.

Design rules (see docs/00_master_report.md, section 5):
  * DO: Unicode NFC, remove zero-width space/BOM, unify whitespace, clamp absurd repeats,
        protect URLs / @mentions / #hashtags so the model never mangles them.
  * DO NOT: lowercase the target, strip emojis, remove punctuation, collapse *all* repeated
        letters (elongation is a normalization signal), stem, or drop Telugu joiners (ZWJ/ZWNJ).
"""
from __future__ import annotations

import re
import unicodedata

TELUGU_RANGE = (0x0C00, 0x0C7F)

URL_RE = r"(?:https?://\S+|www\.\S+)"
MENTION_RE = r"@[A-Za-z0-9_]+"
HASHTAG_RE = r"#[A-Za-z0-9_\u0C00-\u0C7F]+"
PROTECT_RE = re.compile(f"({URL_RE}|{MENTION_RE}|{HASHTAG_RE})")
_ZERO_WIDTH_RE = re.compile("[\u200b\ufeff\u2060]")  # ZWSP, BOM, word-joiner (ZWJ/ZWNJ are kept)
_WS_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def is_telugu_char(ch: str) -> bool:
    return TELUGU_RANGE[0] <= ord(ch) <= TELUGU_RANGE[1]


def has_telugu(text: str) -> bool:
    return any(is_telugu_char(c) for c in text)


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF) or cp in (0x2764, 0x200D, 0xFE0F)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clamp_repeats(text: str, max_run: int = 3) -> str:
    """Limit runs of the same non-digit character to `max_run` ('sooooo' -> 'sooo')."""
    if max_run <= 0:
        return text
    return re.sub(r"(\D)\1{%d,}" % max_run, lambda m: m.group(1) * max_run, text)


def clean_text(text: str, clamp: int = 3) -> str:
    return collapse_whitespace(clamp_repeats(normalize_unicode(text), clamp))


def split_protected(text: str) -> list[tuple[str, bool]]:
    """Split into (segment, is_protected) parts; protected spans bypass the model."""
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in PROTECT_RE.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts or [(text, False)]


_EDGE_PUNCT_RE = re.compile(r"^([\W_]*)(.*?)([\W_]*)$", re.UNICODE | re.DOTALL)


def split_edge_punct(token: str) -> tuple[str, str, str]:
    """'college?!' -> ('', 'college', '?!'). Telugu combining marks are kept inside the core."""
    # \W treats Telugu vowel signs as non-word, so peel punctuation manually.
    i, j = 0, len(token)
    while i < j and _is_punct(token[i]):
        i += 1
    while j > i and _is_punct(token[j - 1]):
        j -= 1
    return token[:i], token[i:j], token[j:]


def _is_punct(ch: str) -> bool:
    if is_telugu_char(ch) or ch.isalnum():
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S") or cat.startswith("Z") or is_emoji_char(ch)


def word_tokens(text: str) -> list[str]:
    return text.split()
