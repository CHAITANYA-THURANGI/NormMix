"""Synthetic pair generation: (noisy source, clean target) with leakage-safe group ids."""
from __future__ import annotations

import random

from ..datasets.seed_corpus import group_key
from ..preprocessing.text_cleaning import has_telugu, is_emoji_char, split_edge_punct
from .noise import EMOJIS, NoiseConfig, RomanizationSampler, SourceNoiser


def generate_pairs(clean_rows: list[dict], variants: int = 8, noise: NoiseConfig | None = None,
                   sampler: RomanizationSampler | None = None, seed: int = 42,
                   p_identity: float = 0.04, source_name: str = "synthetic_seed") -> list[dict]:
    """For each clean target create up to `variants` distinct noisy sources.

    A fraction `p_identity` of samples is Telugu-script/clean input (src == tgt up to casing and
    punctuation) so the model also learns to leave already-normalized text alone.
    """
    rng = random.Random(seed)
    noiser = SourceNoiser(noise, sampler)
    out: list[dict] = []
    for row in clean_rows:
        base_tgt = row["tgt"]
        seen: set[str] = set()
        attempts = 0
        while len(seen) < variants and attempts < variants * 4:
            attempts += 1
            tgt = base_tgt
            emoji = ""
            if rng.random() < noiser.cfg.p("p_emoji"):
                emoji = rng.choice(EMOJIS)
                tgt = f"{base_tgt} {emoji}"
            if rng.random() < p_identity:
                src = tgt.replace("?", "?" * rng.randint(1, 2)) if "?" in tgt else tgt
                kind = "identity"
            else:
                src = noiser.make_source(base_tgt, rng) + (f" {emoji}" if emoji else "")
                kind = "noisy"
            if src in seen:
                continue
            seen.add(src)
            out.append({"src": src, "tgt": tgt, "source": source_name, "kind": kind,
                        "group": row.get("group") or group_key(base_tgt),
                        "template_id": row.get("template_id"), "slot": row.get("slot")})
    return out


# Comprehensive bilingual seed lexicon for realistic English insertion into Telugu sentences.
# Covers education, tech, workplace, transit, daily life, food, and modern communication.
BILINGUAL_SEED = {
    # Education & College
    "కళాశాల": "college", "పాఠశాల": "school", "విశ్వవిద్యాలయం": "university", "పరీక్ష": "exam",
    "తరగతి": "class", "ఉపన్యాసం": "lecture", "విద్యార్థి": "student", "అధ్యాపకుడు": "professor",
    "పుస్తకం": "book", "ప్రాజెక్టు": "project", "సమర్పణ": "presentation", "నివేదిక": "report",
    "గమనికలు": "notes", "మార్కులు": "marks", "ఫలితాలు": "results", "సర్టిఫికెట్": "certificate",
    "గ్రంథాలయం": "library", "వసతి గృహం": "hostel", "ప్రయోగశాల": "lab", "ప్రవేశం": "admission",

    # Workplace & Career
    "ఆఫీసు": "office", "కార్యాలయం": "office", "సమావేశం": "meeting", "సందర్శన": "interview",
    "ఉద్యోగం": "job", "వేతనం": "salary", "సెలవు": "leave", "విరామం": "break", "వివరాలు": "details",
    "పని": "work", "అనుమతి": "permission", "మేనేజర్": "manager", "బృందం": "team",
    "సహోద్యోగి": "colleague", "కంపెనీ": "company", "నిర్ణయం": "decision",

    # Technology, Devices & Digital Life
    "ఫోన్": "phone", "కంప్యూటర్": "computer", "ల్యాప్‌టాప్": "laptop", "ఇంటర్నెట్": "internet",
    "వైఫై": "wifi", "ఛార్జర్": "charger", "బ్యాటరీ": "battery", "కెమెరా": "camera",
    "స్క్రీన్": "screen", "సందేశం": "message", "కాల్": "call", "యాప్": "app",
    "సాఫ్ట్‌వేర్": "software", "పాస్‌వర్డ్": "password", "లింక్": "link", "కోడ్": "code",
    "డేటాబేస్": "database", "సర్వర్": "server", "నోటిఫికేషన్": "notification",

    # Transit, Travel & Places
    "బస్సు": "bus", "రైలు": "train", "కారు": "car", "బైక్": "bike", "విమానం": "flight",
    "టికెట్": "ticket", "స్టేషన్": "station", "స్టాప్": "stop", "రహదారి": "road",
    "ట్రాఫిక్": "traffic", "సిగ్నల్": "signal", "ఆసుపత్రి": "hospital", "బ్యాంకు": "bank",
    "హోటల్": "hotel", "రెస్టారెంట్": "restaurant", "దుకాణం": "shop", "మార్కెట్": "market",
    "షాపింగ్ మాల్": "mall", "సినిమా హాల్": "theatre",

    # Entertainment & Daily Living
    "సినిమా": "movie", "పాట": "song", "వీడియో": "video", "క్రీడ": "game", "మ్యాచ్": "match",
    "పార్టీ": "party", "స్నేహితుడు": "friend", "కుటుంబం": "family", "సమయం": "time",
    "తేదీ": "date", "డబ్బు": "money", "సహాయం": "help", "సమస్య": "problem", "ప్రణాళిక": "plan",
    "ఆలోచన": "idea", "అవకాశం": "chance", "చిరునామా": "address", "కాఫీ": "coffee", "టీ": "tea",
    "ఆర్డర్": "order", "బిల్లు": "bill", "చెల్లింపు": "payment", "నగదు": "cash",
}
_SUFFIXES = ["లో", "కి", "కు", "తో", "నుంచి", "నుండి", "ని", "ను", "గా", "ల", "లు", "ఉ", "ము"]


def insert_english(tgt: str, rng: random.Random, p: float = 0.7, lex: dict[str, str] | None = None) -> str | None:
    """Replace known Telugu nouns with English, splitting off a Telugu suffix as its own token.

    Returns None if nothing was replaced. Morphology beyond simple suffixes is skipped on
    purpose (unknown remainders keep the token unchanged) to avoid producing bad Telugu.
    """
    lex = lex or BILINGUAL_SEED
    changed = False
    out: list[str] = []
    for tok in tgt.split():
        pre, core, post = split_edge_punct(tok)
        replaced = False
        for stem, en in lex.items():
            if core.startswith(stem) and rng.random() < p:
                rest = core[len(stem):]
                if rest == "" or rest in _SUFFIXES:
                    out.append(pre + en + (" " + rest if rest else "") + post)
                    replaced = changed = True
                    break
        if not replaced:
            out.append(tok)
    return " ".join(out) if changed else None


def code_mix_from_telugu(sentences: list[str], seed: int = 42, max_out: int | None = None) -> list[dict]:
    """Turn clean Telugu sentences into code-mixed targets via English insertion."""
    rng = random.Random(seed)
    rows = []
    for s in sentences:
        if not has_telugu(s):
            continue
        cm = insert_english(s, rng)
        if cm and any(c.isascii() and c.isalpha() for c in cm):
            rows.append({"tgt": cm, "template_id": "WIKI_INSERT", "slot": None, "group": group_key(cm)})
        if max_out and len(rows) >= max_out:
            break
    return rows


def composition(tgt: str) -> dict:
    """Language composition of a normalized target: counts of TE / EN tokens and switch points."""
    labels = []
    for tok in tgt.split():
        _, core, _ = split_edge_punct(tok)
        if not core or all(is_emoji_char(c) for c in core):
            continue
        if has_telugu(core):
            labels.append("TE")
        elif any(c.isalpha() for c in core):
            labels.append("EN")
    n_te, n_en = labels.count("TE"), labels.count("EN")
    total = max(n_te + n_en, 1)
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    en_ratio = n_en / total
    mix = "telugu_heavy" if en_ratio <= 0.25 else "english_heavy" if en_ratio >= 0.6 else "balanced"
    return {"n_te": n_te, "n_en": n_en, "en_ratio": round(en_ratio, 3), "switches": switches, "mix": mix}
