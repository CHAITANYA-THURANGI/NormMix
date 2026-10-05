"""Noise models that turn a clean code-mixed target into realistic noisy user input.

All randomness flows through an explicit `random.Random` so datasets are reproducible.
Every operation here is *label preserving* for the normalization target: it changes surface form
(spelling, spacing, casing, repeated letters, script) but never the intended words.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from ..preprocessing.text_cleaning import has_telugu, split_edge_punct
from ..preprocessing.translit_te import romanize_word

# Telugu case markers / postpositions that users often glue onto English words: college+ki
SUFFIX_ROMAN = {"కి": ["ki", "ku"], "లో": ["lo"], "కు": ["ku", "ki"], "తో": ["tho", "to"],
                "నుంచి": ["nunchi", "nundi"], "ని": ["ni"], "గా": ["ga", "gaa"]}

ENGLISH_ABBREV = {
    "college": ["clg", "colg", "collage"], "please": ["pls", "plz"], "thanks": ["thanx", "thx", "tnx"],
    "sorry": ["sry", "soory"], "class": ["clas", "klass"], "hostel": ["hostl", "hostle"],
    "office": ["ofc", "offce"], "project": ["projct", "prjct"], "assignment": ["asignment", "assgnmnt"],
    "exam": ["exm", "eksam"], "meeting": ["mtg", "meting"], "library": ["lib", "libary"],
    "hospital": ["hospitl", "hosptal"], "internet": ["net", "intrnet"], "help": ["hlp", "helpp"],
    "late": ["lt", "lateee"], "phone": ["fone", "phn"], "laptop": ["lappy", "labtop"],
    "charger": ["chrgr", "charjer"], "results": ["resuts", "rslts"], "submit": ["submt", "sumbit"],
    "complete": ["complet", "cmplt"], "problem": ["prob", "problm"], "ticket": ["tkt", "tiket"],
    "party": ["partyy", "paarty"], "movie": ["mvie", "muvie"], "interview": ["intrvw", "interveiw"],
}
_KEY_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj", "i": "uo",
    "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o", "r": "et", "s": "ad",
    "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}
VOWELS = set("aeiou")
EMOJIS = ["😂", "🙏", "😊", "👍", "🔥", "😅"]


@dataclass
class NoiseConfig:
    roman_variation: float = 0.35      # per-unit chance of picking a non-canonical romanization
    p_keep_telugu_script: float = 0.05  # leave a Telugu token in script (mixed-script input)
    p_en_typo: float = 0.22            # generic typo on an English token
    p_en_abbrev: float = 0.30          # known abbreviation for an English token
    p_te_vowel_drop: float = 0.03      # drop an inner vowel in a Romanized Telugu token
    p_repeat: float = 0.04             # elongate a letter ("chalaaaa")
    p_merge_suffix: float = 0.25       # glue Telugu suffix to previous English word
    p_drop_punct: float = 0.35
    p_dup_punct: float = 0.15
    p_capitalize: float = 0.10
    p_upper_token: float = 0.03
    p_emoji: float = 0.08
    strength: float = 1.0              # global multiplier for the probabilities above

    def scaled(self, factor: float) -> "NoiseConfig":
        return replace(self, strength=self.strength * factor)

    def p(self, name: str) -> float:
        return min(0.95, getattr(self, name) * self.strength)


class RomanizationSampler:
    """Sample a Roman spelling for a Telugu word: attested (if provided) or rule-based."""

    def __init__(self, attested: dict[str, list[tuple[str, float]]] | None = None, p_attested: float = 0.6):
        self.attested = attested or {}
        self.p_attested = p_attested

    def sample(self, native: str, rng: random.Random, variation: float) -> str:
        options = self.attested.get(native)
        if options and rng.random() < self.p_attested:
            words, weights = zip(*options)
            return rng.choices(words, weights=weights, k=1)[0]
        return romanize_word(native, rng, variation)


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    op = rng.choice(["del_vowel", "del_char", "swap", "double", "neighbor", "phonetic"])
    chars = list(word)
    idx = list(range(1, len(chars)))  # keep the first letter
    if op == "del_vowel":
        cand = [i for i in idx if chars[i] in VOWELS]
        if cand:
            del chars[rng.choice(cand)]
    elif op == "del_char":
        del chars[rng.choice(idx)]
    elif op == "swap" and len(chars) > 3:
        i = rng.randrange(1, len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "double":
        i = rng.choice(idx)
        chars.insert(i, chars[i])
    elif op == "neighbor":
        i = rng.choice(idx)
        nb = _KEY_NEIGHBORS.get(chars[i])
        if nb:
            chars[i] = rng.choice(nb)
    elif op == "phonetic":
        w = "".join(chars)
        for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("ee", "i"), ("oo", "u")):
            if a in w[1:]:
                return w[0] + w[1:].replace(a, b, 1)
    return "".join(chars)


def _elongate(word: str, rng: random.Random) -> str:
    if not word:
        return word
    pos = [i for i, c in enumerate(word) if c in VOWELS] or [len(word) - 1]
    i = rng.choice(pos)
    return word[:i] + word[i] * rng.randint(2, 3) + word[i + 1:]


def _drop_inner_vowel(word: str, rng: random.Random) -> str:
    cand = [i for i in range(1, len(word) - 1) if word[i] in VOWELS]
    if len(word) < 5 or not cand:
        return word
    i = rng.choice(cand)
    return word[:i] + word[i + 1:]


class SourceNoiser:
    """Build a noisy source string from a clean target string."""

    def __init__(self, cfg: NoiseConfig | None = None, sampler: RomanizationSampler | None = None):
        self.cfg = cfg or NoiseConfig()
        self.sampler = sampler or RomanizationSampler()

    def _english(self, core: str, rng: random.Random) -> str:
        c = self.cfg
        low = core.lower()
        if low in ENGLISH_ABBREV and rng.random() < c.p("p_en_abbrev"):
            return rng.choice(ENGLISH_ABBREV[low])
        if rng.random() < c.p("p_en_typo"):
            return _typo(core, rng)
        return core

    def _telugu(self, core: str, rng: random.Random) -> str:
        c = self.cfg
        if rng.random() < c.p("p_keep_telugu_script"):
            return core
        r = self.sampler.sample(core, rng, c.roman_variation)
        if rng.random() < c.p("p_te_vowel_drop"):
            r = _drop_inner_vowel(r, rng)
        if rng.random() < c.p("p_repeat"):
            r = _elongate(r, rng)
        return r

    def make_source(self, tgt: str, rng: random.Random) -> str:
        c = self.cfg
        out: list[str] = []
        prev_english_open = False  # previous token is an English word with no trailing punctuation
        for tok in tgt.split():
            pre, core, post = split_edge_punct(tok)
            if not core:
                out.append(tok)
                prev_english_open = False
                continue
            if has_telugu(core):
                if prev_english_open and core in SUFFIX_ROMAN and not pre and rng.random() < c.p("p_merge_suffix"):
                    out[-1] = out[-1] + rng.choice(SUFFIX_ROMAN[core]) + post
                    prev_english_open = False
                    continue
                new = self._telugu(core, rng)
                prev_english_open = False
            elif core.isascii() and core.isalpha():
                new = self._english(core, rng)
                prev_english_open = not post
            else:
                new = core
                prev_english_open = False
            if rng.random() < c.p("p_upper_token") and new.isalpha():
                new = new.upper()
            if post:
                if all(ch in ".?!," for ch in post) and rng.random() < c.p("p_drop_punct"):
                    post = ""
                elif post[-1] in "?!." and rng.random() < c.p("p_dup_punct"):
                    post = post + post[-1] * rng.randint(1, 2)
            out.append(pre + new + post)
        text = " ".join(out)
        if text and rng.random() < c.p("p_capitalize"):
            text = text[0].upper() + text[1:]
        return text
