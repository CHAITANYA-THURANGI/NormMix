"""Rule-based Telugu-script -> Roman transliteration with controllable spelling variation.

Used ONLY to synthesize realistic noisy Romanized input from clean Telugu targets (data
augmentation). It is deterministic in canonical mode (`rng=None`) which is what the unit tests
check. It is not a claim of linguistic completeness (no ISO 15919 diacritics, no schwa-deletion
model); real attested romanizations (Dakshina / Aksharantar lexicons) should be preferred when
available -- see `RomanizationSampler` in src/augmentation/noise.py.
"""
from __future__ import annotations

import random

VIRAMA = "\u0c4d"
ANUSVARA = "\u0c02"
CANDRABINDU = "\u0c01"
VISARGA = "\u0c03"

# First option is the canonical romanization; later ones are attested informal variants.
CONSONANTS: dict[str, list[str]] = {
    "క": ["k"], "ఖ": ["kh", "k"], "గ": ["g"], "ఘ": ["gh", "g"], "ఙ": ["ng", "n"],
    "చ": ["ch", "c"], "ఛ": ["chh", "ch"], "జ": ["j"], "ఝ": ["jh", "j"], "ఞ": ["nj", "n"],
    "ట": ["t"], "ఠ": ["th", "t"], "డ": ["d"], "ఢ": ["dh", "d"], "ణ": ["n"],
    "త": ["t", "th"], "థ": ["th", "t"], "ద": ["d", "dh"], "ధ": ["dh", "d"], "న": ["n"],
    "ప": ["p"], "ఫ": ["ph", "f", "p"], "బ": ["b"], "భ": ["bh", "b"], "మ": ["m"],
    "య": ["y"], "ర": ["r"], "ల": ["l"], "వ": ["v", "w"], "శ": ["sh", "s"], "ష": ["sh", "s"],
    "స": ["s"], "హ": ["h"], "ళ": ["l"], "ఱ": ["rr", "r"],
}
INDEPENDENT_VOWELS: dict[str, list[str]] = {
    "అ": ["a"], "ఆ": ["aa", "a"], "ఇ": ["i", "e"], "ఈ": ["ee", "i"], "ఉ": ["u"], "ఊ": ["oo", "u"],
    "ఋ": ["ru"], "ఎ": ["e"], "ఏ": ["e", "ee", "ae"], "ఐ": ["ai", "ay"], "ఒ": ["o"],
    "ఓ": ["o", "oo"], "ఔ": ["au", "ow"],
}
VOWEL_SIGNS: dict[str, list[str]] = {
    "ా": ["aa", "a"], "ి": ["i", "e"], "ీ": ["ee", "i"], "ు": ["u"], "ూ": ["oo", "u"],
    "ృ": ["ru"], "ె": ["e"], "ే": ["e", "ee", "ae"], "ై": ["ai", "ay"], "ొ": ["o"],
    "ో": ["o", "oo"], "ౌ": ["au", "ow"],
}
LABIALS = set("పఫబభమ")


def _pick(options: list[str], rng: random.Random | None, variation: float) -> str:
    if rng is None or variation <= 0 or len(options) == 1:
        return options[0]
    if rng.random() < variation:
        return options[rng.randrange(1, len(options))]
    return options[0]


def romanize_word(word: str, rng: random.Random | None = None, variation: float = 0.0) -> str:
    """Romanize one Telugu-script word. Non-Telugu characters pass through unchanged."""
    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            base = _pick(CONSONANTS[ch], rng, variation)
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                # geminate (C + virama + C): optionally write the consonant once
                if i + 2 < n and word[i + 2] == ch and rng is not None and rng.random() < variation * 0.5:
                    base = ""
                out.append(base)
                i += 2
            elif nxt in VOWEL_SIGNS:
                out.append(base + _pick(VOWEL_SIGNS[nxt], rng, variation))
                i += 2
            else:
                out.append(base + "a")
                i += 1
        elif ch in INDEPENDENT_VOWELS:
            out.append(_pick(INDEPENDENT_VOWELS[ch], rng, variation))
            i += 1
        elif ch == ANUSVARA:
            nxt = word[i + 1] if i + 1 < n else ""
            out.append("m" if (not nxt or nxt in LABIALS) else "n")
            i += 1
        elif ch == CANDRABINDU:
            out.append("n")
            i += 1
        elif ch == VISARGA:
            out.append("h")
            i += 1
        elif ch in ("\u200c", "\u200d"):
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def romanize_text(text: str, rng: random.Random | None = None, variation: float = 0.0) -> str:
    return " ".join(romanize_word(w, rng, variation) for w in text.split())
