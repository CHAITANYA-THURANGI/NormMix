"""Omni-Directional Telugu-English Code-Mixed Universal Processor.

Auto-detects input modality (Pure English, Pure Telugu Script, Romanized Tanglish,
Bi-scriptal Code-Mixed) and generates ALL possible output transformations simultaneously:
1. Standard Normalized Code-Mixed (Telugu Script + Preserved English Loanwords via SOTA BiGRU)
2. Full Telugu Script (all words, including English loanwords, transliterated into Telugu Unicode)
3. Full Romanized / Tanglish (all Telugu script phonetically transliterated into Latin characters)
4. English Semantic Gloss / Translation (bilingual vocabulary mapping and glossing)
5. Comprehensive Token-by-Token Language and Script Diagnostic Breakdown.
"""
from __future__ import annotations

import re
import time
from typing import Any, Literal

from ..preprocessing.text_cleaning import clean_text, has_telugu, split_edge_punct
from ..preprocessing.translit_te import romanize_text, romanize_word

# Common English loanwords to authentic Telugu script transliterations
LOANWORD_TO_TELUGU: dict[str, str] = {
    # Education & Campus
    "college": "కాలేజీ", "collage": "కాలేజీ", "school": "స్కూల్", "university": "యూనివర్సిటీ",
    "exam": "ఎగ్జామ్", "class": "క్లాస్", "student": "స్టూడెంట్", "teacher": "టీచర్",
    "professor": "ప్రొఫెసర్", "notes": "నోట్స్", "project": "ప్రాజెక్ట్", "presentation": "ప్రెజెంటేషన్",
    "report": "రిపోర్ట్", "marks": "మార్క్స్", "results": "రిజల్ట్స్", "certificate": "సర్టిఫికెట్",
    "admission": "అడ్మిషన్", "lab": "ల్యాబ్", "hostel": "హాస్టల్", "library": "లైబ్రరీ",
    "course": "కోర్స్", "subject": "సబ్జెక్ట్", "campus": "క్యాంపస్", "degree": "డిగ్రీ",
    "interview": "ఇంటర్వ్యూ", "important": "ఇంపార్టెంట్",

    # Workplace & Career
    "office": "ఆఫీస్", "meeting": "మీటింగ్", "salary": "శాలరీ", "leave": "లీవ్",
    "manager": "మేనేజర్", "team": "టీమ్", "company": "కంపెనీ", "job": "జాబ్",
    "work": "వర్క్", "boss": "బాస్", "client": "క్లయింట్", "schedule": "షెడ్యూల్",
    "deadline": "డెడ్‌లైన్", "task": "టాస్క్", "break": "బ్రేక్", "shift": "షిఫ్ట్",

    # Technology & Digital
    "mobile": "మొబైల్", "phone": "ఫోన్", "wifi": "వైఫై", "router": "రౌటర్",
    "charger": "ఛార్జర్", "laptop": "ల్యాప్‌టాప్", "computer": "కంప్యూటర్", "internet": "ఇంటర్నెట్",
    "password": "పాస్‌వర్డ్", "app": "యాప్", "link": "లింక్", "call": "కాల్",
    "message": "మెసేజ్", "screen": "స్క్రీన్", "battery": "బ్యాటరీ", "code": "కోడ్",
    "server": "సర్వర్", "database": "డేటాబేస్", "system": "సిస్టమ్", "data": "డేటా",
    "network": "నెట్‌వర్క్", "online": "ఆన్‌లైన్", "offline": "ఆఫ్‌లైన్", "update": "అప్‌డేట్",
    "switch": "స్విచ్", "off": "ఆఫ్", "on": "ఆన్", "download": "డౌన్‌లోడ్",

    # Transit & Travel
    "train": "ట్రైన్", "bus": "బస్సు", "car": "కారు", "bike": "బైక్", "flight": "ఫ్లైట్",
    "ticket": "టికెట్", "station": "స్టేషన్", "stop": "స్టాప్", "traffic": "ట్రాఫిక్",
    "signal": "సిగ్నల్", "road": "రోడ్డు", "auto": "ఆటో", "cab": "క్యాబ్", "driver": "డ్రైవర్",

    # Daily Living & Expressions
    "coffee": "కాఫీ", "tea": "టీ", "order": "ఆర్డర్", "bill": "బిల్లు", "hotel": "హోటల్",
    "restaurant": "రెస్టారెంట్", "bank": "బ్యాంక్", "hospital": "హాస్పిటల్", "doctor": "డాక్టర్",
    "movie": "మూవీ", "theatre": "థియేటర్", "party": "పార్టీ", "time": "టైమ్",
    "morning": "మార్నింగ్", "evening": "ఈవెనింగ్", "night": "నైట్", "please": "ప్లీజ్",
    "thanks": "థాంక్స్", "sorry": "సారీ", "ok": "ఓకే", "confirm": "కన్ఫర్మ్",
    "cancel": "క్యాన్సల్", "help": "హెల్ప్", "problem": "ప్రాబ్లెమ్", "idea": "ఐడియా",
    "plan": "ప్లాన్", "food": "ఫుడ్", "water": "వాటర్", "money": "మనీ",
}

COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "are", "where", "today",
    "tomorrow", "yesterday", "going", "send", "sent", "please", "help", "need", "urgent", "report",
    "hi", "hello", "hey", "morning", "evening", "night", "afternoon", "fine", "thank", "thanks",
    "welcome", "bye", "okay", "ok", "yes", "great", "awesome", "dear", "friend", "sir", "madam",
    "is", "am", "was", "were", "has", "had", "does", "did", "very", "much", "too", "here", "why"
}


# High-frequency Telugu roots/lemmas to English gloss translations
TELUGU_TO_ENGLISH_GLOSS: dict[str, str] = {
    # Pronouns
    "naku": "to me", "నాకు": "to me", "nenu": "I", "నేను": "I",
    "nuvvu": "you", "నువ్వు": "you", "meeru": "you (formal)", "మీరు": "you (formal)",
    "athanu": "he", "అతను": "he", "aame": "she", "ఆమె": "she",
    "adi": "that / it", "అది": "that / it", "idi": "this", "ఇది": "this",
    "vaallu": "they", "వాళ్లు": "they", "manamu": "we", "మనము": "we",

    # Time & Questions
    "ivala": "today", "ఇవాళ": "today", "eroju": "today", "eeroju": "today", "ఈరోజు": "today", "repu": "tomorrow", "రేపు": "tomorrow",
    "ninna": "yesterday", "నిన్న": "yesterday", "ippudu": "now", "ఇప్పుడు": "now",
    "appudu": "then", "అప్పుడు": "then", "eppudu": "when", "ఎప్పుడు": "when",
    "ekkada": "where", "ఎక్కడ": "where", "ikkada": "here", "ఇక్కడ": "here",
    "akkada": "there", "అక్కడ": "there", "ela": "how", "ella": "how", "yela": "how", "ఎలా": "how",
    "enti": "what", "ఏంటి": "what", "enduku": "why", "ఎందుకు": "why",

    # Adverbs & Postpositions
    "chala": "very", "చాలా": "very", "baga": "well", "బాగా": "well",
    "lo": "in", "లో": "in", "ki": "to / for", "కి": "to / for", "ku": "to", "కు": "to",
    "tho": "with", "తో": "with", "nunchi": "from", "నుంచి": "from", "nundi": "from", "నుండి": "from",
    "kosam": "for", "కోసం": "for", "mariyu": "and", "మరియు": "and", "leda": "or", "లేదా": "or",

    # Verbs & Predicates
    "undi": "has / is there", "ఉంది": "has / is there", "ledu": "is not / no", "లేదు": "is not / no",
    "unnav": "are (you)", "unnavu": "are (you)", "vunnav": "are (you)", "vunnavu": "are (you)", "vunnvu": "are (you)", "unnvu": "are (you)", "vunvu": "are (you)", "unvu": "are (you)", "ఉన్నావ్": "are (you)", "unnaru": "are (you/they)", "ఉన్నారు": "are (you/they)",
    "unnanu": "am (I)", "ఉన్నాను": "am (I)",
    "vellali": "need to go", "వెళ్లాలి": "need to go", "vellanu": "went", "వెళ్లాను": "went",
    "vastanu": "will come", "వస్తాను": "will come", "vastunnava": "are you coming?", "vastunava": "are you coming?", "వస్తున్నావా": "are you coming?", "vastara": "will you come?", "వస్తారా": "will you come?",
    "vastunnanu": "am coming", "వస్తున్నాను": "am coming",
    "pani": "work", "పని": "work", "cheyali": "must do", "చేయాలి": "must do",
    "chesanu": "did", "చేశాను": "did", "chesaru": "did (they)", "చేశారు": "did (they)",
    "chey": "do", "చేయ్": "do", "cheyandi": "please do", "చేయండి": "please do",
    "cheyatledu": "is not working", "చేయట్లేదు": "is not working",
    "bagundi": "is good", "బాగుంది": "is good",
    "ainda": "is it done?", "అయిందా": "is it done?", "aina": "became", "అయినా": "became",
    "marchipoyanu": "forgot", "మర్చిపోయాను": "forgot",
    "aipotundi": "getting finished / running out", "అయిపోతుంది": "getting finished / running out",
    "namaskaram": "greetings / hello", "నమస్కారం": "greetings / hello",
    "dhanyavadalu": "thank you", "ధన్యవాదాలు": "thank you",
    "kavali": "need / want", "కావాలి": "need / want",
    "istam": "like", "ఇష్టం": "like",
    "bhojanam": "meal / food", "భోజనం": "meal / food",
    "ardam": "understood", "అర్థం": "understood", "ardham": "understood",
    "nidra": "sleep", "నిద్ర": "sleep",
    "akali": "hunger / hungry", "ఆకలి": "hunger / hungry",
    "sahayam": "help", "సహాయం": "help",
    "intlo": "at home", "ఇంట్లో": "at home",
    "intiki": "to home", "ఇంటికి": "to home",
    "amma": "mother / mom", "అమ్మ": "mother / mom",
    "nanna": "father / dad", "నాన్న": "father / dad",
}

# Rule-based fallback phonetics for arbitrary English words into Telugu script
_PHONETIC_CONSONANTS = {
    "b": "బ", "c": "క", "d": "డ", "f": "ఫ", "g": "గ", "h": "హ", "j": "జ", "k": "క",
    "l": "ల", "m": "మ", "n": "న", "p": "ప", "q": "క", "r": "ర", "s": "స", "t": "ట",
    "v": "వ", "w": "వ", "x": "క్స్", "y": "య", "z": "జ్", "sh": "ష", "ch": "చ్", "th": "త్", "ph": "ఫ"
}
_PHONETIC_VOWELS = {
    "a": "ా", "e": "ె", "i": "ి", "o": "ొ", "u": "ు", "ee": "ీ", "oo": "ూ", "ai": "ై", "au": "ౌ"
}


def fallback_english_to_telugu(word: str) -> str:
    """Heuristic phonetic transliteration for English words not in loanword dictionary."""
    w = word.lower()
    if w in LOANWORD_TO_TELUGU:
        return LOANWORD_TO_TELUGU[w]
    out = []
    i, n = 0, len(w)
    while i < n:
        if i + 1 < n and w[i:i+2] in _PHONETIC_CONSONANTS:
            out.append(_PHONETIC_CONSONANTS[w[i:i+2]])
            i += 2
        elif i + 1 < n and w[i:i+2] in _PHONETIC_VOWELS:
            out.append(_PHONETIC_VOWELS[w[i:i+2]])
            i += 2
        elif w[i] in _PHONETIC_CONSONANTS:
            c = _PHONETIC_CONSONANTS[w[i]]
            if i + 1 < n and w[i+1] in _PHONETIC_VOWELS:
                out.append(c + _PHONETIC_VOWELS[w[i+1]])
                i += 2
            else:
                out.append(c + "్" if i + 1 < n else c)
                i += 1
        else:
            i += 1
    res = "".join(out)
    return res if res else word


def fold_phonetics(s: str) -> str:
    """Folds informal phonetic variants of romanized Telugu into standardized stems."""
    w = s.lower().strip()
    if has_telugu(w):
        w = romanize_text(w).lower()
    w = re.sub(r"[^\w\s]", " ", w)
    tokens = w.split()
    folded = []
    for tok in tokens:
        if (
            tok in COMMON_ENGLISH_WORDS
            or tok in LOANWORD_TO_TELUGU
            or tok in {
                "class", "meeting", "office", "college", "school", "report", "notes",
                "interview", "important", "good", "morning", "evening", "night", "food",
                "water", "time", "help", "need", "urgent", "thanks", "please", "sorry"
            }
        ):
            folded.append(tok)
            continue
        # Convert sh -> s for Telugu romanized words (e.g. cheshava -> chesava, shubharatri -> subharatri)
        t = tok.replace("sh", "s")
        t = re.sub(r"^v(?=[uo])", "", t)
        t = re.sub(r"^y(?=[ei])", "", t)
        t = re.sub(r"aa+", "a", t)
        t = re.sub(r"ee+", "e", t)
        t = re.sub(r"oo+", "o", t)
        t = re.sub(r"uu+", "u", t)
        t = re.sub(r"ll+", "l", t)
        t = re.sub(r"nn+", "n", t)
        t = re.sub(r"dd+", "d", t)
        t = re.sub(r"tt+", "t", t)
        t = re.sub(r"ss+", "s", t)
        t = re.sub(r"pp+", "p", t)
        # Standardize forms of unnavu/unnaru/unnanu/undi
        if re.search(r"^v?un(r[a-z]*|ar[a-z]*|ara)$", t):
            t = "unnaru"
        elif re.search(r"^v?un(v[a-z]*|av[a-z]*|vu|va)?$", t):
            t = "unnav"
        elif re.search(r"^v?un(n[a-z]*|an[a-z]*)$", t):
            t = "unnanu"
        elif re.search(r"^v?un(d[a-z]*|di)$", t):
            t = "undi"
        else:
            t = re.sub(r"([aeiou])v[ou]$", r"\1v", t)
            t = re.sub(r"^unav[ou]*$", "unav", t)
            t = re.sub(r"^artham$", "ardam", t)
        folded.append(t)
    return " ".join(folded)



TELUGU_SCRIPT_TO_ENGLISH_WORDS: dict[str, str] = {
    "కన్వర్ట్": "convert", "కన్వర్షన్": "conversion",
    "ట్రాన్స్లేట్": "translate", "ట్రాన్స్‌లేట్": "translate", "ట్రాన్స్లేషన్": "translation",
    "దిస్": "this", "దట్": "that", "దీస్": "these", "దోస్": "those",
    "ఆన్": "an", "ఎ": "a",
    "సెంట్న్స్": "sentence", "సెంటెన్స్": "sentence", "సెంట్న్సెస్": "sentences", "సెంటెన్సెస్": "sentences",
    "వర్డ్": "word", "వర్డ్స్": "words",
    "ఇన్ టు": "into", "ఇంటు": "into", "ఇన్": "in", "టు": "to",
    "ఇంగ్లీష్": "English", "ఇంగ్లిష్": "English", "ఆంగ్లం": "English", "తెలుగు": "Telugu",
    "మీనింగ్": "meaning", "టెక్స్ట్": "text", "లైన్": "line", "పారాగ్రాఫ్": "paragraph",
    "చేంజ్": "change", "టైప్": "type", "రైట్": "write", "రీడ్": "read", "స్పీక్": "speak", "లిజన్": "listen",
    "సెండ్": "send", "పోస్ట్": "post", "షేర్": "share", "కాపీ": "copy", "పేస్ట్": "paste",
    "సెలెక్ట్": "select", "చెక్": "check", "టెస్ట్": "test", "క్లిక్": "click", "ఓపెన్": "open", "క్లోజ్": "close",
    "స్టార్ట్": "start", "స్టాప్": "stop", "కాల్": "call", "మెసేజ్": "message",
    "ప్లీజ్": "please", "థాంక్స్": "thanks", "సారీ": "sorry", "ఓకే": "ok",
    "కాలేజీ": "college", "స్కూల్": "school", "యూనివర్సిటీ": "university", "క్లాస్": "class",
    "స్టూడెంట్": "student", "టీచర్": "teacher", "ఎగ్జామ్": "exam", "నోట్స్": "notes",
    "ప్రాజెక్ట్": "project", "రిపోర్ట్": "report", "అసైన్‌మెంట్": "assignment",
    "ఆఫీస్": "office", "ఆఫీసు": "office", "మీటింగ్": "meeting", "మేనేజర్": "manager",
    "టీమ్": "team", "వర్క్": "work", "జాబ్": "job", "టాస్క్": "task", "ఇంటర్వ్యూ": "interview",
    "ఫోన్": "phone", "మొబైల్": "mobile", "ల్యాప్‌టాప్": "laptop", "కంప్యూటర్": "computer",
    "వైఫై": "wifi", "ఇంటర్నెట్": "internet", "నెట్‌వర్క్": "network", "పాస్‌వర్డ్": "password",
    "యాప్": "app", "లింక్": "link", "ఫైల్": "file", "స్క్రీన్": "screen", "బ్యాటరీ": "battery",
    "చార్జర్": "charger", "ఛార్జర్": "charger", "డౌన్‌లోడ్": "download", "అప్‌డేట్": "update",
    "బస్సు": "bus", "ట్రైన్": "train", "కారు": "car", "బైక్": "bike", "టికెట్": "ticket", "స్టేషన్": "station",
    "హాస్పిటల్": "hospital", "డాక్టర్": "doctor", "హోటల్": "hotel", "రెస్టారెంట్": "restaurant",
    "కాఫీ": "coffee", "టీ": "tea", "ఫుడ్": "food", "వాటర్": "water", "మనీ": "money", "టైమ్": "time",
    "ప్రాబ్లెమ్": "problem", "హెల్ప్": "help", "ఐడియా": "idea", "ప్లాన్": "plan"
}


def detransliterate_telugu_loanwords(text: str) -> str:
    """Replaces English loanwords written in Telugu script with their Latin English equivalents."""
    t = text
    for k in sorted(TELUGU_SCRIPT_TO_ENGLISH_WORDS.keys(), key=len, reverse=True):
        t = re.sub(rf'(?<![^\s.,?!;:]){re.escape(k)}(?![^\s.,?!;:])', TELUGU_SCRIPT_TO_ENGLISH_WORDS[k], t)
    return t


CONVERSATIONAL_PATTERNS: list[tuple[str, Any]] = [
    # Hello / How are you / Praise
    (r"^(helo|hello|hi|namaskaram)\s+(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav|vunnavu)[\?\.]*$", "Hello, how are you?"),
    (r"^(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav|vunnavu)[\?\.]*$", "How are you?"),
    (r"^(nuv|nuvu|nuvvu)\s+(ela|yela|ella)\s+(unav|unnav|vunnav|vunnavu)[\?\.]*$", "How are you?"),
    (r"^(mer|meru|meeru)\s+(ela|yela|ella)\s+(unaru|unnaru|vunnaru|vunnavu)[\?\.]*$", "How are you?"),
    (r"^(helo|hello|hi)\s+(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "Hello, are you doing well?"),
    (r"^(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "Are you doing well? / How are you?"),
    (r"^(nenu|nen)\s+(bagunanu|bagunnanu|baga\s+unanu|baga\s+unnanu)[\.]*$", "I am doing well."),
    (r"^(bagunanu|bagunnanu)[\.]*$", "I am doing well."),
    (r"^(mer|meru|meeru)\s+(chala|chaala)\s+(bagunaru|bagunnaru|bagunaaru|bagunnaaru)[\.]*$", "You look very good."),
    (r"^(nuv|nuvu|nuvvu)\s+(chala|chaala)\s+(bagunav|bagunnav|bagunaav|bagunnaavu)[\.]*$", "You look great."),
    (r"^(nenu|nen)\s+(chala|chaala)\s+(bagunanu|bagunnanu)[\.]*$", "I am doing very well."),

    # Conversion & Translation Requests
    (r"^(convert|kanvart|translate|traslet)\s+(this|dis)\s+(an\s+|a\s+|aan\s+)?(sentence|sentns)\s+(in\s*to|into|in\s+to|to|in\s+tu)\s+(english|ingles|englis)[\.]*$", "Convert this sentence into English."),
    (r"^(convert|kanvart|translate|traslet)\s+(this|dis)\s+(an\s+|a\s+|aan\s+)?(sentence|sentns)[\.]*$", "Convert this sentence."),

    # Multi-sentence greeting + question
    (r"^(hi|hello|hey)\s+(ela|yela|ella),?\s+(unav|unnav|vunav|vunnav|vunnavu)\.?\s+are\s+you\s+coming\s+to\s+([a-z]+)[?\.]*$",
     lambda m: f"Hello, how are you? Are you coming to {m.group(4)}?"),
    (r"^are\s+you\s+coming\s+to\s+([a-z]+)[?\.]*$", lambda m: f"Are you coming to {m.group(1)}?"),
    (r"^are\s+you\s+going\s+to\s+([a-z]+)[?\.]*$", lambda m: f"Are you going to {m.group(1)}?"),

    # Chit-chat & Daily Conversations
    (r"^(enti|yenti)\s+(sangathulu|visheshalu|visesalu)[\?\.]*$", "What's up? / What's the news?"),
    (r"^(em|yem|emi|enti)\s+(chestunav|chestunnav|chesthunnav)[\?\.]*$", "What are you doing?"),
    (r"^(ippudu|ipudu)\s+(em|yem|emi|enti)\s+(chestunav|chestunnav)[\?\.]*$", "What are you doing now?"),
    (r"^(nuv|nuvu|nuvvu)\s+(em|yem|emi|enti)\s+(chestunav|chestunnav)[\?\.]*$", "What are you doing?"),
    (r"^(ekada|ekkada)\s+(unav|unnav|vunnav)[\?\.]*$", "Where are you?"),
    (r"^(nuv|nuvu|nuvvu)\s+(ekada|ekkada)\s+(unav|unnav|vunnav)[\?\.]*$", "Where are you?"),
    (r"^(epudu|eppudu)\s+(vastav|vastunav|vastunnav)[\?\.]*$", "When are you coming?"),
    (r"^(nuv|nuvu|nuvvu)\s+(epudu|eppudu)\s+(vastav|vastunav|vastunnav)[\?\.]*$", "When will you come?"),
    (r"^(repu|repu\s+morning)\s+(kaludam|kaluddam|kalusukundam)[\.]*$", "Let's meet tomorrow."),
    (r"^(manam|manamu)\s+repu\s+(udayam\s+)?(kaludam|kaluddam)[\.]*$", "Let's meet tomorrow morning."),
    (r"^(sare|okay|ok)\s+repu\s+(matladadam|matladam)[\.]*$", "Okay, let's talk tomorrow."),
    (r"^(chala|chaala)\s+(thanks|dhanyavadalu)[\.]*$", "Thank you very much."),
    (r"^(dhanyavadalu|dhanyavadaalu)[\.]*$", "Thank you."),
    (r"^(namaskaram)[\.]*$", "Greetings / Hello."),
    (r"^(subhodayam|shubhodayam)[\.]*$", "Good morning."),
    (r"^(subharatri|shubharatri)[\.]*$", "Good night."),
    (r"^(bhojanam)\s+(chesava|chesara)[\?\.]*$", "Did you have food?"),
    (r"^(naku|naaku)\s+(ardam|ardham|artham)\s+(kaledu)[\.]*$", "I did not understand."),
    (r"^(malli|malli\s+oka\s+sari)\s+(chepu|cheppu)[\.]*$", "Please say it again."),
    (r"^(naku|naaku)\s+(nidra)\s+(vastondi|vastundi)[\.]*$", "I am feeling sleepy."),
    (r"^(naku|naaku)\s+(akali|akaliga)\s+(undi|vundi)[\.]*$", "I am hungry."),
    (r"^(naku|naaku)\s+(telugu\s+)?(chala\s+)?(istam|istham)[\.]*$", "I like Telugu very much."),
    (r"^amma\s+intlo\s+(undi|vundi)[\.]*$", "Mom is at home."),
    (r"^nanna\s+office\s+(ki|ku)\s+velaru[\.]*$", "Dad went to the office."),
    (r"^(naku|naaku)\s+(urgent\s+ga\s+)?(help|sahayam)\s+(kavali)[\.]*$", "I need help."),
    (r"^(naku|naaku)\s+koncham\s+(help|sahayam)\s+(kavali)[\.]*$", "I need some help."),
    (r"^(ippudu|ipudu)\s+(naku|naaku)\s+call\s+(cheyi|chey|cheyandi)[\.]*$", "Please call me now."),
    (r"^sorry\s+nenu\s+late\s+ga\s+(vastanu|vasta)[\.]*$", "Sorry, I will come late."),
    (r"^(good|gud)\s+morning[,!\.]*$", "Good morning."),
    (r"^(good|gud)\s+night[,!\.]*$", "Good night."),
    (r"^where\s+are\s+you\s+going[?\.]*$", "Where are you going?"),
    (r"^where\s+are\s+you[?\.]*$", "Where are you?"),
    (r"^what\s+are\s+you\s+doing[?\.]*$", "What are you doing?"),
    (r"^what\s+is\s+your\s+name[?\.]*$", "What is your name?"),
    (r"^how\s+much(\s+is\s+this|\s+does\s+it\s+cost)?[?\.]*$", "How much is this?"),
    (r"^thank\s+you\s+(so\s+much|very\s+much)[!\.]*$", "Thank you very much."),
    (r"^i\s+(need|want)\s+help[\.]*$", "I need help."),
]

# Dedicated Pure English (Formal & Elegant Standard English) Conversational Patterns
CONVERSATIONAL_PURE_ENGLISH_PATTERNS: list[tuple[str, Any]] = [
    # Greetings & Health Status
    (r"^(helo|hello|hi|namaskaram)\s+(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav|vunnavu)[\?\.]*$", "Greetings. How do you do? I trust you are well."),
    (r"^(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav|vunnavu)[\?\.]*$", "How do you do? I trust you are well."),
    (r"^(nuv|nuvu|nuvvu)\s+(ela|yela|ella)\s+(unav|unnav|vunnav|vunnavu)[\?\.]*$", "How do you do? I hope you are doing well."),
    (r"^(mer|meru|meeru)\s+(ela|yela|ella)\s+(unaru|unnaru|vunnaru|vunnavu)[\?\.]*$", "How do you do? I trust you are well."),
    (r"^(helo|hello|hi)\s+(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "Greetings. Are you in sound health and spirits?"),
    (r"^(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "Are you in sound health and good spirits?"),
    (r"^(nenu|nen)\s+(bagunanu|bagunnanu|baga\s+unanu|baga\s+unnanu)[\.]*$", "I am in sound health and very good spirits."),
    (r"^(bagunanu|bagunnanu)[\.]*$", "I am doing exceptionally well."),
    (r"^(mer|meru|meeru)\s+(chala|chaala)\s+(bagunaru|bagunnaru|bagunaaru|bagunnaaru)[\.]*$", "You are doing exceptionally well."),
    (r"^(nuv|nuvu|nuvvu)\s+(chala|chaala)\s+(bagunav|bagunnav|bagunaav|bagunnaavu)[\.]*$", "You are in excellent spirits."),
    (r"^(nenu|nen)\s+(chala|chaala)\s+(bagunanu|bagunnanu)[\.]*$", "I am in very good health and spirits."),

    # Commands & Linguistic Requests
    (r"^(convert|kanvart|translate|traslet)\s+(this|dis)\s+(an\s+|a\s+|aan\s+)?(sentence|sentns)\s+(in\s*to|into|in\s+to|to|in\s+tu)\s+(english|ingles|englis)[\.]*$", "Please translate this sentence into English."),
    (r"^(convert|kanvart|translate|traslet)\s+(this|dis)\s+(an\s+|a\s+|aan\s+)?(sentence|sentns)[\.]*$", "Please translate this sentence."),

    # Daily Expressions
    (r"^(chala|chaala)\s+(thanks|dhanyavadalu)[\.]*$", "I express my sincere and heartfelt gratitude."),
    (r"^(dhanyavadalu|dhanyavadaalu)[\.]*$", "Thank you very much. I appreciate your kind assistance."),
    (r"^(namaskaram)[\.]*$", "Greetings. It is an honor to speak with you."),
    (r"^(subhodayam|shubhodayam)[\.]*$", "A very pleasant morning to you."),
    (r"^(subharatri|shubharatri)[\.]*$", "Wishing you a peaceful and restful night."),
    (r"^(bhojanam)\s+(chesava|chesara)[\?\.]*$", "Have you partaken of your meal?"),
    (r"^(naku|naaku)\s+(ardam|ardham|artham)\s+(kaledu)[\.]*$", "I have not comprehended that."),
    (r"^(malli|malli\s+oka\s+sari)\s+(chepu|cheppu)[\.]*$", "Kindly repeat your statement."),
    (r"^(naku|naaku)\s+(nidra)\s+(vastondi|vastundi)[\.]*$", "I am feeling drowsy."),
    (r"^(naku|naaku)\s+(akali|akaliga)\s+(undi|vundi)[\.]*$", "I am experiencing hunger."),
    (r"^(naku|naaku)\s+(telugu\s+)?(chala\s+)?(istam|istham)[\.]*$", "I possess a deep appreciation for the Telugu language."),
    (r"^(repu|repu\s+morning)\s+(kaludam|kaluddam|kalusukundam)[\.]*$", "We look forward to convening tomorrow."),
    (r"^(sare|okay|ok)\s+repu\s+(matladadam|matladam)[\.]*$", "Very well, we shall converse tomorrow."),
    (r"^sorry\s+nenu\s+late\s+ga\s+(vastanu|vasta)[\.]*$", "Please accept my apologies; my arrival will be slightly delayed."),
    (r"^(naku|naaku)\s+(urgent\s+ga\s+)?(help|sahayam)\s+(kavali)[\.]*$", "I require immediate assistance."),
    (r"^(naku|naaku)\s+koncham\s+(help|sahayam)\s+(kavali)[\.]*$", "I would be grateful for your assistance."),
    (r"^(ippudu|ipudu)\s+(naku|naaku)\s+call\s+(cheyi|chey|cheyandi)[\.]*$", "Kindly contact me by telephone at this moment."),
    (r"^(hi|hello|hey|hai)[\.!,]*$", "Greetings."),
]


def _translate_motion_destination_clause(folded: str, pure: bool = False) -> str | None:
    """Translates motion and destination clauses like 'eroju college ki vastunava' cleanly."""
    m = re.search(
        r"^(.+?)\s+(ki|ku)\s+(vastunava|vastunnava|vastava|vastara|vastunara|vastunnara|velthunava|velthunnava|velthava|velthara|veltanu|velthanu|velta|veltha|vastanu|vasta)[\?\.]*$",
        folded
    )
    if not m:
        return None
    pre = m.group(1).strip()
    verb_tok = m.group(3)
    subj = "you"
    if re.search(r"\b(nenu|nen)\b", pre):
        subj = "I"
        pre = re.sub(r"\b(nenu|nen)\b", "", pre)
    elif re.search(r"\b(nuvvu|nuv|nuvu)\b", pre):
        subj = "you"
        pre = re.sub(r"\b(nuvvu|nuv|nuvu)\b", "", pre)
    elif re.search(r"\b(meeru|meru)\b", pre):
        subj = "you"
        pre = re.sub(r"\b(meeru|meru)\b", "", pre)
    elif re.search(r"\b(manam|manamu)\b", pre):
        subj = "we"
        pre = re.sub(r"\b(manam|manamu)\b", "", pre)

    time_str = ""
    if re.search(r"\b(repu\s+(morning|udayam))\b", pre):
        time_str = "tomorrow morning"
        pre = re.sub(r"\b(repu\s+(morning|udayam))\b", "", pre)
    elif re.search(r"\b(repu\s+(sayantram|evening))\b", pre):
        time_str = "tomorrow evening"
        pre = re.sub(r"\b(repu\s+(sayantram|evening))\b", "", pre)
    elif re.search(r"\b(eroju|eeroju|ivala|e\s+roju)\s+(morning|udayam)\b", pre):
        time_str = "this morning"
        pre = re.sub(r"\b(eroju|eeroju|ivala|e\s+roju)\s+(morning|udayam)\b", "", pre)
    elif re.search(r"\b(eroju|eeroju|ivala|e\s+roju)\s+(sayantram|evening)\b", pre):
        time_str = "this evening"
        pre = re.sub(r"\b(eroju|eeroju|ivala|e\s+roju)\s+(sayantram|evening)\b", "", pre)
    elif re.search(r"\b(eroju|eeroju|ivala|e\s+roju)\b", pre):
        time_str = "today"
        pre = re.sub(r"\b(eroju|eeroju|ivala|e\s+roju)\b", "", pre)
    elif re.search(r"\b(repu)\b", pre):
        time_str = "tomorrow"
        pre = re.sub(r"\b(repu)\b", "", pre)
    elif re.search(r"\b(ninna)\b", pre):
        time_str = "yesterday"
        pre = re.sub(r"\b(ninna)\b", "", pre)
    elif re.search(r"\b(ippudu|ipudu)\b", pre):
        time_str = "now"
        pre = re.sub(r"\b(ippudu|ipudu)\b", "", pre)

    dest = pre.strip()
    dest_map = {
        "intiki": "home", "illu": "home", "pani": "work",
        "kalasala": "college", "patasala": "school",
        "karyalayam": "office", "samavesam": "meeting",
        "mukhamukhi": "interview", "asupatri": "hospital"
    }
    dest = dest_map.get(dest, dest)

    if dest == "home":
        prep_dest = "home"
    elif dest in {"office", "meeting", "hospital", "station", "airport", "gym", "market", "bank", "library", "lab", "interview", "party"}:
        prep_dest = f"to the {dest}"
    elif dest:
        prep_dest = f"to {dest}"
    else:
        prep_dest = ""

    t_sfx = f" {time_str}" if time_str else ""

    if pure:
        if verb_tok in {"vastunava", "vastunnava", "vastava", "vastara", "vastunara", "vastunnara"}:
            if dest in {"college", "school", "class", "university", "office", "meeting", "interview"}:
                art = "the " if dest in {"office", "meeting", "interview"} else ""
                return f"Will you be attending {art}{dest}{t_sfx}?"
            elif dest == "home":
                return f"Will you be returning home{t_sfx}?"
            elif dest:
                return f"Will you be arriving at {dest}{t_sfx}?"
            else:
                return f"Will you be arriving{t_sfx}?"
        elif verb_tok in {"velthunava", "velthunnava", "velthava", "velthara", "velthunara", "velthunnara"}:
            if dest in {"college", "school", "class", "university", "office", "meeting", "interview"}:
                art = "the " if dest in {"office", "meeting", "interview"} else ""
                return f"Will you be attending {art}{dest}{t_sfx}?"
            elif dest:
                return f"Will you be proceeding to {dest}{t_sfx}?"
            else:
                return f"Will you be departing{t_sfx}?"
        elif verb_tok in {"veltanu", "velthanu", "velta", "veltha"}:
            if dest in {"college", "school", "class", "university", "office", "meeting", "interview"}:
                art = "the " if dest in {"office", "meeting", "interview"} else ""
                return f"{subj} shall attend {art}{dest}{t_sfx}."
            else:
                return f"{subj} shall proceed to {dest}{t_sfx}."
        elif verb_tok in {"vastanu", "vasta"}:
            return f"{subj} shall arrive at {dest}{t_sfx}."
    else:
        if verb_tok in {"vastunava", "vastunnava", "vastava", "vastara", "vastunara", "vastunnara"}:
            return f"Are you coming {prep_dest}{t_sfx}?" if prep_dest else f"Are you coming{t_sfx}?"
        elif verb_tok in {"velthunava", "velthunnava", "velthava", "velthara", "velthunara", "velthunnara"}:
            return f"Are you going {prep_dest}{t_sfx}?" if prep_dest else f"Are you going{t_sfx}?"
        elif verb_tok in {"veltanu", "velthanu", "velta", "veltha"}:
            return f"{subj} will go {prep_dest}{t_sfx}."
        elif verb_tok in {"vastanu", "vasta"}:
            return f"{subj} will come {prep_dest}{t_sfx}."

    return None


def _translate_single_clause_to_english(raw: str, normalized: str = "") -> str:
    """Translates a single Telugu, Tanglish, or Code-Mixed clause into fluent English."""
    raw_clean = raw.strip()
    if not raw_clean:
        return ""

    # Check if already pure English
    te_chars = sum(1 for c in raw_clean if "\u0C00" <= c <= "\u0C7F")
    words = raw_clean.split()
    low_words = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    if te_chars == 0 and all(
        w in LOANWORD_TO_TELUGU
        or w in COMMON_ENGLISH_WORDS
        or w in {
            "the", "a", "an", "is", "are", "i", "you", "we", "he", "she", "it", "they",
            "in", "to", "for", "with", "from", "on", "at", "what", "where", "when", "why",
            "how", "hello", "hi", "good", "morning", "night", "thanks", "thank", "please",
            "not", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
            "coming", "going", "come", "go", "came", "went", "doing", "help", "need", "want",
            "and", "or", "but", "so", "am", "my", "your", "his", "her", "their", "our"
        }
        or not w
        for w in low_words
    ):
        res = raw_clean
        if res and res[0].islower():
            res = res[0].upper() + res[1:]
        return res

    # 1. Phonetic folding & exact conversational pattern match
    detrans = detransliterate_telugu_loanwords(raw_clean)
    folded = fold_phonetics(detrans)
    for pat, eng in CONVERSATIONAL_PATTERNS:
        m = re.search(pat, folded)
        if m:
            return eng(m) if callable(eng) else eng

    if detrans != raw_clean:
        orig_folded = fold_phonetics(raw_clean)
        for pat, eng in CONVERSATIONAL_PATTERNS:
            m = re.search(pat, orig_folded)
            if m:
                return eng(m) if callable(eng) else eng

    if normalized:
        norm_detrans = detransliterate_telugu_loanwords(normalized)
        norm_folded = fold_phonetics(norm_detrans)
        for pat, eng in CONVERSATIONAL_PATTERNS:
            m = re.search(pat, norm_folded)
            if m:
                return eng(m) if callable(eng) else eng

    # Check if this clause contains subclauses separated by comma or semicolon
    if "," in raw_clean or ";" in raw_clean:
        subclauses = [c.strip() for c in re.split(r'[,;]\s*', raw_clean) if c.strip()]
        if len(subclauses) > 1:
            parts = []
            has_q = False
            for i, c in enumerate(subclauses):
                res = _translate_single_clause_to_english(c)
                if res:
                    if res.endswith("?"):
                        has_q = True
                    clean_res = res.rstrip(".!?")
                    if i > 0 and clean_res and not clean_res.startswith("I "):
                        clean_res = clean_res[0].lower() + clean_res[1:]
                    parts.append(clean_res)
            if parts:
                term = "?" if (has_q or raw_clean.endswith("?")) else "."
                return ", ".join(parts) + term

    # 2. Template matching for common syntactic patterns
    m = re.search(r"^(.+?)\s+(ready|charge|miss)\s+ayindi[\.]*$", folded)
    if m:
        item = m.group(1).strip()
        action = m.group(2)
        m_prep = re.match(r"^(\w+)\s+(?:lo|in)\s+(.+)$", item)
        if m_prep:
            place, thing = m_prep.group(1), m_prep.group(2)
            article = "the " if place in {"office", "hospital", "bank", "library", "lab"} else ""
            item_eng = f"{thing} in {article}{place}"
        else:
            item_eng = item
        if action == "ready":
            return f"The {item_eng} is ready."
        elif action == "charge":
            return f"The {item_eng} is charged."
        elif action == "miss":
            return f"The {item_eng} was missed."

    m = re.search(r"^(.+?)\s+send\s+(cheyi|chey|cheyandi)[\.]*$", folded)
    if m:
        item = m.group(1).strip()
        return f"Please send the {item}."

    m = re.search(r"^(nenu\s+)?(.+?)\s+submit\s+chesanu[\.]*$", folded)
    if m:
        item = m.group(2).strip()
        return f"I submitted the {item}."

    m = re.search(r"^(nuvvu\s+|nuv\s+)?(.+?)\s+complete\s+chesava[\?\.]*$", folded)
    if m:
        item = m.group(2).strip()
        return f"Did you complete the {item}?"

    res_motion = _translate_motion_destination_clause(folded, pure=False)
    if res_motion:
        return res_motion

    # 3. Structured Telugu / Tanglish Clause Parser (SOV -> SVO alignment)
    s = folded
    is_question = bool(
        re.search(r"\b(unava|unnava|chesava|nachinda|telusa|vastava|epudu|eppudu|ekada|ekkada|enti|yenti|em|yem|ela|yela|enduku)\b", s)
        or "?" in raw_clean
    )

    subj = ""
    subj_role = "nominative"
    if re.search(r"\b(naku|naaku)\b", s):
        subj = "I"
        subj_role = "dative"
        s = re.sub(r"\b(naku|naaku)\b", "", s)
    elif re.search(r"\b(nenu|nen)\b", s):
        subj = "I"
        s = re.sub(r"\b(nenu|nen)\b", "", s)
    elif re.search(r"\b(nuv|nuvu|nuvvu)\b", s):
        subj = "you"
        s = re.sub(r"\b(nuv|nuvu|nuvvu)\b", "", s)
    elif re.search(r"\b(mer|meru|meeru)\b", s):
        subj = "you"
        s = re.sub(r"\b(mer|meru|meeru)\b", "", s)
    elif re.search(r"\b(manam|manamu)\b", s):
        subj = "we"
        s = re.sub(r"\b(manam|manamu)\b", "", s)
    elif re.search(r"\b(vadu|atanu)\b", s):
        subj = "he"
        s = re.sub(r"\b(vadu|atanu)\b", "", s)
    elif re.search(r"\b(aame)\b", s):
        subj = "she"
        s = re.sub(r"\b(aame)\b", "", s)
    elif re.search(r"\b(vallu|vaallu)\b", s):
        subj = "they"
        s = re.sub(r"\b(vallu|vaallu)\b", "", s)

    time_phrase = ""
    if re.search(r"\b(ivala|e\s+roju|eroju)\b", s):
        time_phrase = "today"
        s = re.sub(r"\b(ivala|e\s+roju|eroju)\b", "", s)
    elif re.search(r"\b(repu\s+morning|repu\s+udayam)\b", s):
        time_phrase = "tomorrow morning"
        s = re.sub(r"\b(repu\s+morning|repu\s+udayam)\b", "", s)
    elif re.search(r"\b(repu)\b", s):
        time_phrase = "tomorrow"
        s = re.sub(r"\b(repu)\b", "", s)
    elif re.search(r"\b(ninna)\b", s):
        time_phrase = "yesterday"
        s = re.sub(r"\b(ninna)\b", "", s)
    elif re.search(r"\b(ippudu|ipudu)\b", s):
        time_phrase = "now"
        s = re.sub(r"\b(ippudu|ipudu)\b", "", s)
    elif re.search(r"\b(sayantram)\b", s):
        time_phrase = "in the evening"
        s = re.sub(r"\b(sayantram)\b", "", s)

    prep_phrases = []
    for m in re.finditer(r"\b(\w+)\s+lo\b", s):
        place = m.group(1)
        article = "the " if place in {"office", "hospital", "bank", "library", "lab"} else ""
        prep_phrases.append(f"in {article}{place}")
        s = re.sub(r"\b" + place + r"\s+lo\b", "", s)

    for m in re.finditer(r"\b(\w+)\s+(ki|ku)\b", s):
        place = m.group(1)
        article = "the " if place in {"office", "hospital", "bank", "library", "lab", "party"} else ""
        prep_phrases.append(f"to {article}{place}")
        s = re.sub(r"\b" + place + r"\s+(ki|ku)\b", "", s)

    for m in re.finditer(r"\b(\w+)\s+(nunchi|nundi)\b", s):
        place = m.group(1)
        prep_phrases.append(f"from {place}")
        s = re.sub(r"\b" + place + r"\s+(nunchi|nundi)\b", "", s)

    verb_phrase = ""
    if re.search(r"\b(velali|vellali)\b", s):
        verb_phrase = "need to go"
        s = re.sub(r"\b(velali|vellali)\b", "", s)
    elif re.search(r"\b(veltanu|velthanu)\b", s):
        verb_phrase = "will go"
        s = re.sub(r"\b(veltanu|velthanu)\b", "", s)
    elif re.search(r"\b(veldam|veldaam)\b", s):
        verb_phrase = "let's go"
        s = re.sub(r"\b(veldam|veldaam)\b", "", s)
    elif re.search(r"\b(vachanu|vachaanu)\b", s):
        verb_phrase = "just came" if ("ippude" in s or "ipude" in s) else "came"
        s = re.sub(r"\b(vachanu|vachaanu|ippude|ipude)\b", "", s)
    elif re.search(r"\b(vastanu|vasta)\b", s):
        verb_phrase = "will come"
        s = re.sub(r"\b(vastanu|vasta)\b", "", s)
    elif re.search(r"\b(vastunnanu|vastunanu)\b", s):
        verb_phrase = "am coming"
        s = re.sub(r"\b(vastunnanu|vastunanu)\b", "", s)
    elif re.search(r"\b(vastundi|vastundhi)\b", s):
        verb_phrase = "arrives"
        s = re.sub(r"\b(vastundi|vastundhi)\b", "", s)
    elif re.search(r"\b(cheyali|cheyyali)\b", s):
        verb_phrase = "must do"
        s = re.sub(r"\b(cheyali|cheyyali)\b", "", s)
    elif re.search(r"\b(chala\s+bagundi|bagundi)\b", s):
        verb_phrase = "is very good"
        s = re.sub(r"\b(chala\s+bagundi|bagundi)\b", "", s)
    elif re.search(r"\b(chala\s+istam|istam)\b", s):
        verb_phrase = "like very much"
        s = re.sub(r"\b(chala\s+istam|istam)\b", "", s)
    elif re.search(r"\b(kavali)\b", s):
        verb_phrase = "need"
        s = re.sub(r"\b(kavali)\b", "", s)
    elif re.search(r"\b(unava|unnava)\b", s):
        verb_phrase = "are you"
        s = re.sub(r"\b(unava|unnava)\b", "", s)
    elif re.search(r"\b(undi|vundi)\b", s):
        if subj == "I" and subj_role == "dative":
            verb_phrase = "have"
        else:
            verb_phrase = "there is"
        s = re.sub(r"\b(undi|vundi)\b", "", s)
    elif re.search(r"\b(ledu)\b", s):
        if subj == "I" and subj_role == "dative":
            verb_phrase = "do not have"
        else:
            verb_phrase = "there is no"
        s = re.sub(r"\b(ledu)\b", "", s)

    rem_tokens = s.strip().split()
    obj_words = []
    for tok in rem_tokens:
        if tok in LOANWORD_TO_TELUGU or tok in {
            "interview", "important", "meeting", "problem", "notes", "report", "exam", "class",
            "phone", "laptop", "wifi", "bus", "train", "ticket"
        }:
            obj_words.append(tok)
        elif tok in TELUGU_TO_ENGLISH_GLOSS:
            obj_words.append(TELUGU_TO_ENGLISH_GLOSS[tok])
        elif tok:
            obj_words.append(tok)

    obj_phrase = " ".join(obj_words).strip()
    if obj_phrase:
        if re.search(r"^(interview|important interview|meeting|problem|report|class|exam)$", obj_phrase):
            article = "an" if (obj_phrase.startswith(("a", "e", "i", "o", "u")) or obj_phrase.startswith("important")) else "a"
            obj_phrase = f"{article} {obj_phrase}"

    parts = []
    if verb_phrase in {"there is", "there is no"}:
        parts.append(verb_phrase)
        if obj_phrase:
            parts.append(obj_phrase)
        if prep_phrases:
            parts.extend(prep_phrases)
        if time_phrase:
            parts.append(time_phrase)
    elif verb_phrase in {"have", "do not have", "need"}:
        parts.append(subj or "I")
        parts.append(verb_phrase)
        if obj_phrase:
            parts.append(obj_phrase)
        if prep_phrases:
            parts.extend(prep_phrases)
        if time_phrase:
            parts.append(time_phrase)
    elif verb_phrase == "are you":
        parts.append("Are you")
        if prep_phrases:
            parts.extend(prep_phrases)
        if time_phrase:
            parts.append(time_phrase)
    elif verb_phrase == "let's go":
        parts.append("Let's go")
        if prep_phrases:
            parts.extend(prep_phrases)
        if time_phrase:
            parts.append(time_phrase)
    else:
        if subj:
            parts.append(subj)
        if verb_phrase:
            parts.append(verb_phrase)
        if obj_phrase:
            parts.append(obj_phrase)
        if prep_phrases:
            parts.extend(prep_phrases)
        if time_phrase:
            parts.append(time_phrase)

    assembled = " ".join(parts).strip()
    if not assembled or len(assembled) < 3:
        glosses = []
        for tok in raw_clean.split():
            _, core, _ = split_edge_punct(tok)
            low = core.lower()
            if low in TELUGU_TO_ENGLISH_GLOSS:
                glosses.append(TELUGU_TO_ENGLISH_GLOSS[low])
            elif low in LOANWORD_TO_TELUGU:
                glosses.append(low)
            else:
                glosses.append(core)
        assembled = " ".join(glosses)

    if assembled:
        assembled = assembled[0].upper() + assembled[1:]
        if is_question and not assembled.endswith("?"):
            assembled += "?"
        elif not assembled.endswith((".", "!", "?")):
            assembled += "."

    return assembled


def elevate_to_pure_english(text: str) -> str:
    """Elevates conversational English text to pristine, formal Pure English."""
    if not text:
        return ""
    t = text.strip()

    elevations = [
        (r"\bcan't\b", "cannot"),
        (r"\bdon't\b", "do not"),
        (r"\bdidn't\b", "did not"),
        (r"\bwon't\b", "will not"),
        (r"\bi'm\b", "I am"),
        (r"\byou're\b", "you are"),
        (r"\bit's\b", "it is"),
        (r"\bthere's\b", "there is"),
        (r"\bthanks a lot\b", "I express my sincere gratitude"),
        (r"\bthanks\b", "thank you"),
        (r"\bsorry\b", "please accept my apologies"),
        (r"\bhelp\b", "assistance"),
        (r"\ba lot of\b", "a substantial amount of"),
        (r"\bwant\b", "require"),
        (r"\bwill go\b", "shall proceed"),
        (r"\bcall me now\b", "please contact me by telephone at once"),
        (r"\b(hello|hi)[,\s]+how are you[?\.]*", "Greetings. How do you do? I trust you are well."),
        (r"^how are you[?\.]*", "How do you do? I trust you are well."),
        (r"\bhow are you[?\.]*", "how do you do? I trust you are well."),
        (r"\b(and\s+)?good morning\b", "and a very pleasant morning to you"),
        (r"\bgood morning\b", "a very pleasant morning to you"),
        (r"\bgood night\b", "wishing you a peaceful and restful night"),
        (r"\bi am doing well\b", "I am in sound health and good spirits"),
        (r"\byou look very good\b", "you are doing exceptionally well"),
        (r"\bconvert this sentence into english\b", "please translate this sentence into English"),
    ]
    for pat, rep in elevations:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    t = re.sub(r"\.\s+and\b", ", and", t, flags=re.IGNORECASE)

    if t:
        t = t[0].upper() + t[1:]
        if not t.endswith((".", "!", "?")):
            t += "."
    return t


def _translate_single_clause_to_pure_english(raw: str, normalized: str = "") -> str:
    """Translates a single Telugu, Tanglish, or Code-Mixed clause into formal, elegant Pure English."""
    raw_clean = raw.strip()
    if not raw_clean:
        return ""

    detrans = detransliterate_telugu_loanwords(raw_clean)
    folded = fold_phonetics(detrans)

    for pat, eng in CONVERSATIONAL_PURE_ENGLISH_PATTERNS:
        m = re.search(pat, folded)
        if m:
            return eng(m) if callable(eng) else eng

    if detrans != raw_clean:
        orig_folded = fold_phonetics(raw_clean)
        for pat, eng in CONVERSATIONAL_PURE_ENGLISH_PATTERNS:
            m = re.search(pat, orig_folded)
            if m:
                return eng(m) if callable(eng) else eng

    if normalized:
        norm_detrans = detransliterate_telugu_loanwords(normalized)
        norm_folded = fold_phonetics(norm_detrans)
        for pat, eng in CONVERSATIONAL_PURE_ENGLISH_PATTERNS:
            m = re.search(pat, norm_folded)
            if m:
                return eng(m) if callable(eng) else eng

    # Check if this clause contains subclauses separated by comma or semicolon
    if "," in raw_clean or ";" in raw_clean:
        subclauses = [c.strip() for c in re.split(r'[,;]\s*', raw_clean) if c.strip()]
        if len(subclauses) > 1:
            parts = []
            has_q = False
            for i, c in enumerate(subclauses):
                res = _translate_single_clause_to_pure_english(c)
                if res:
                    if res.endswith("?"):
                        has_q = True
                    clean_res = res.rstrip(".!?")
                    parts.append(clean_res)
            if parts:
                term = "?" if (has_q or raw_clean.endswith("?")) else "."
                if parts[0].lower() in {"greetings", "hello", "good morning", "good evening", "good afternoon"}:
                    first = parts[0]
                    if not first.endswith("."):
                        first += "."
                    second = parts[1]
                    if second:
                        second = second[0].upper() + second[1:]
                    rest = parts[2:]
                    return " ".join([first, second] + rest) + term
                return "; ".join(parts) + term

    m = re.search(r"^(.+?)\s+(ready|charge|miss)\s+ayindi[\.]*$", folded)
    if m:
        item = m.group(1).strip()
        action = m.group(2)
        if action == "ready":
            return f"The {item} has been prepared and is ready for use."
        elif action == "charge":
            return f"The {item} has been fully recharged."
        elif action == "miss":
            return f"The scheduled {item} was unfortunately missed."

    m = re.search(r"^(.+?)\s+send\s+(cheyi|chey|cheyandi)[\.]*$", folded)
    if m:
        item = m.group(1).strip()
        return f"Please transmit the {item} at your earliest convenience."

    m = re.search(r"^(nenu\s+)?(.+?)\s+submit\s+chesanu[\.]*$", folded)
    if m:
        item = m.group(2).strip()
        return f"I have successfully submitted the {item}."

    m = re.search(r"^(nuvvu\s+|nuv\s+)?(.+?)\s+complete\s+chesava[\?\.]*$", folded)
    if m:
        item = m.group(2).strip()
        return f"Have you finalized the {item}?"

    res_motion_pure = _translate_motion_destination_clause(folded, pure=True)
    if res_motion_pure:
        return res_motion_pure

    m = re.search(r"^(nenu\s+)?repu\s+(.+?)\s+(ki|ku)\s+(veltanu|velthanu)[\.]*$", folded)
    if m:
        item = m.group(2).strip()
        return f"I shall proceed to {item} tomorrow."

    base_en = _translate_single_clause_to_english(raw_clean, normalized)
    return elevate_to_pure_english(base_en)


def translate_to_english(raw: str, normalized: str = "") -> str:
    """Translates Telugu, Tanglish, or Code-Mixed text into natural, fluent English with multi-sentence support."""
    raw_clean = raw.strip()
    if not raw_clean:
        return ""

    # Multi-sentence split
    sentences = re.split(r'(?<=[.!?])\s+', raw_clean)
    norm_sentences = re.split(r'(?<=[.!?])\s+', normalized.strip()) if normalized else []
    if len(sentences) > 1:
        parts = []
        for i, s in enumerate(sentences):
            s_clean = s.strip()
            if not s_clean:
                continue
            norm_s = norm_sentences[i].strip() if i < len(norm_sentences) else ""
            res = _translate_single_clause_to_english(s_clean, norm_s)
            if res:
                parts.append(res)
        if parts:
            return " ".join(parts)

    return _translate_single_clause_to_english(raw_clean, normalized)


def translate_to_pure_english(raw: str, normalized: str = "") -> str:
    """Translates Telugu, Tanglish, or Code-Mixed text into formal, pristine Standard English (శుద్ధ ఆంగ్లం)."""
    raw_clean = raw.strip()
    if not raw_clean:
        return ""

    # Multi-sentence split
    sentences = re.split(r'(?<=[.!?])\s+', raw_clean)
    norm_sentences = re.split(r'(?<=[.!?])\s+', normalized.strip()) if normalized else []
    if len(sentences) > 1:
        parts = []
        for i, s in enumerate(sentences):
            s_clean = s.strip()
            if not s_clean:
                continue
            norm_s = norm_sentences[i].strip() if i < len(norm_sentences) else ""
            res = _translate_single_clause_to_pure_english(s_clean, norm_s)
            if res:
                parts.append(res)
        if parts:
            return " ".join(parts)

    return _translate_single_clause_to_pure_english(raw_clean, normalized)


# Destination, Subject, and Time mappings for natural Pure Telugu syntax generation
_PURE_DEST_MAP: dict[str, str] = {
    "college": "కళాశాలకు",
    "school": "పాఠశాలకు",
    "office": "కార్యాలయానికి",
    "university": "విశ్వవిద్యాలయానికి",
    "hospital": "వైద్యశాలకు",
    "library": "గ్రంథాలయానికి",
    "home": "ఇంటికి",
    "house": "ఇంటికి",
    "room": "గదికి",
    "hostel": "వసతి గృహానికి",
    "market": "సంతకు",
    "city": "నగరానికి",
    "village": "గ్రామానికి",
    "theatre": "చిత్ర మందిరానికి",
    "hotel": "భోజనశాలకు",
    "restaurant": "భోజనశాలకు",
}

_PURE_SUB_MAP: dict[str, tuple[str, str, str, str, str]] = {
    # (Telugu pronoun, going_prog, coming_prog, go_future, come_future)
    "he": ("అతను", "వెళ్తున్నాడు", "వస్తున్నాడు", "వెళ్తాడు", "వస్తాడు"),
    "she": ("ఆమె", "వెళ్తున్నది", "వస్తున్నది", "వెళ్తుంది", "వస్తుంది"),
    "they": ("వారు", "వెళ్తున్నారు", "వస్తున్నారు", "వెళ్తారు", "వస్తారు"),
    "we": ("మనం", "వెళ్తున్నాం", "వస్తున్నాం", "వెళ్తాం", "వస్తాం"),
    "i": ("నేను", "వెళ్తున్నాను", "వస్తున్నాను", "వెళ్తాను", "వస్తాను"),
    "you": ("మీరు", "వెళ్తున్నారు", "వస్తున్నారు", "వెళ్తారు", "వస్తారు"),
}

_PURE_Q_MAP: dict[str, tuple[str, str, str, str, str]] = {
    # (Telugu pronoun, going_prog_q, coming_prog_q, go_future_q, come_future_q)
    "he": ("అతను", "వెళ్తున్నాడా", "వస్తున్నాడా", "వెళ్తాడా", "వస్తాడా"),
    "she": ("ఆమె", "వెళ్తున్నదా", "వస్తున్నదా", "వెళ్తుందా", "వస్తుందా"),
    "they": ("వారు", "వెళ్తున్నారా", "వస్తున్నారా", "వెళ్తారా", "వస్తారా"),
    "you": ("మీరు", "వెళ్తున్నారా", "వస్తున్నారా", "వెళ్తారా", "వస్తారా"),
    "we": ("మనం", "వెళ్తున్నామా", "వస్తున్నామా", "వెళ్తామా", "వస్తామా"),
}

_PURE_TIME_MAP: dict[str, str] = {
    "today": "ఈ రోజు",
    "tomorrow": "రేపు",
    "yesterday": "నిన్న",
    "now": "ఇప్పుడు",
    "morning": "ఉదయం",
    "evening": "సాయంత్రం",
    "night": "రాత్రి",
}


def _translate_movement_pattern(m: re.Match) -> str:
    sub = m.group(1).lower()
    verb = m.group(2).lower()
    dest = m.group(3).lower()
    time = m.group(4).lower() if len(m.groups()) >= 4 and m.group(4) else None

    sub_te, go_prog, come_prog, go_fut, come_fut = _PURE_SUB_MAP.get(
        sub, (sub, "వెళ్తున్నారు", "వస్తున్నారు", "వెళ్తారు", "వస్తారు")
    )
    if "go" in verb:
        verb_te = go_prog if "ing" in verb else go_fut
    else:
        verb_te = come_prog if "ing" in verb else come_fut

    dest_te = _PURE_DEST_MAP.get(dest, dest + "కు")
    time_te = _PURE_TIME_MAP.get(time, "") if time else ""

    parts = [sub_te]
    if time_te:
        parts.append(time_te)
    parts.append(dest_te)
    parts.append(verb_te + ".")
    return " ".join(parts)


def _translate_q_movement(m: re.Match) -> str:
    aux = m.group(1).lower()
    sub = m.group(2).lower()
    verb = m.group(3).lower()
    dest = m.group(4).lower()
    time = m.group(5).lower() if len(m.groups()) >= 5 and m.group(5) else None

    sub_te, go_prog_q, come_prog_q, go_fut_q, come_fut_q = _PURE_Q_MAP.get(
        sub, (sub, "వెళ్తున్నారా", "వస్తున్నారా", "వెళ్తారా", "వస్తారా")
    )
    if aux == "will" or verb in ("come", "go"):
        verb_te = go_fut_q if "go" in verb else come_fut_q
    else:
        verb_te = go_prog_q if "go" in verb else come_prog_q

    dest_te = _PURE_DEST_MAP.get(dest, dest + "కు")
    time_te = _PURE_TIME_MAP.get(time, "") if time else ""

    parts = [sub_te]
    if time_te:
        parts.append(time_te)
    parts.append(dest_te)
    parts.append(verb_te + "?")
    return " ".join(parts)


# Authentic Pure Telugu (అచ్చ తెలుగు / శుద్ధ తెలుగు) Conversational Idioms
PURE_TELUGU_IDIOMS: list[tuple[str, Any]] = [
    (r"^(helo|hello|hi|namaskaram)\s+(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav)[\?\.]*$", "నమస్కారం, ఎలా ఉన్నారు?"),
    (r"^(ela|yela|ella)\s+(unav|unnav|unaru|unnaru|vunnav)[\?\.]*$", "ఎలా ఉన్నారు?"),
    (r"^(nuv|nuvu|nuvvu)\s+(ela|yela|ella)\s+(unav|unnav|vunnav)[\?\.]*$", "నువ్వు ఎలా ఉన్నావు?"),
    (r"^(mer|meru|meeru)\s+(ela|yela|ella)\s+(unaru|unnaru|vunnaru)[\?\.]*$", "మీరు ఎలా ఉన్నారు?"),
    (r"^(helo|hello|hi)\s+(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "నమస్కారం, బాగున్నారా?"),
    (r"^(bagunava|bagunnava|bagunara|bagunnara)[\?\.]*$", "బాగున్నారా?"),
    (r"^(nenu|nen)\s+(bagunanu|bagunnanu|baga\s+unanu|baga\s+unnanu)[\.]*$", "నేను చాలా బాగున్నాను."),
    (r"^(bagunanu|bagunnanu)[\.]*$", "చాలా బాగున్నాను."),
    (r"^(enti|yenti)\s+(sangathulu|visheshalu|visesalu)[\?\.]*$", "ఏంటి విశేషాలు?"),
    (r"^(em|yem|emi|enti)\s+(chestunav|chestunnav|chesthunnav)[\?\.]*$", "ఏమి చేస్తున్నారు?"),
    (r"^(ippudu|ipudu)\s+(em|yem|emi|enti)\s+(chestunav|chestunnav)[\?\.]*$", "ఇప్పుడు ఏమి చేస్తున్నారు?"),
    (r"^(nuv|nuvu|nuvvu)\s+(em|yem|emi|enti)\s+(chestunav|chestunnav)[\?\.]*$", "నువ్వు ఏమి చేస్తున్నావు?"),
    (r"^(ekada|ekkada)\s+(unav|unnav|vunnav)[\?\.]*$", "ఎక్కడ ఉన్నారు?"),
    (r"^(nuv|nuvu|nuvvu)\s+(ekada|ekkada)\s+(unav|unnav|vunnav)[\?\.]*$", "నువ్వు ఎక్కడ ఉన్నావు?"),
    (r"^(epudu|eppudu)\s+(vastav|vastunav|vastunnav)[\?\.]*$", "ఎప్పుడు వస్తారు?"),
    (r"^(nuv|nuvu|nuvvu)\s+(epudu|eppudu)\s+(vastav|vastunav|vastunnav)[\?\.]*$", "నువ్వు ఎప్పుడు వస్తావు?"),
    (r"^(repu|repu\s+morning)\s+(kaludam|kaluddam|kalusukundam)[\.]*$", "రేపు కలుద్దాం."),
    (r"^(manam|manamu)\s+repu\s+(udayam\s+)?(kaludam|kaluddam)[\.]*$", "మనం రేపు ఉదయం కలుద్దాం."),
    (r"^(sare|okay|ok)\s+repu\s+(matladadam|matladam)[\.]*$", "సరే, రేపు మాట్లాడదాం."),
    (r"^(chala|chaala)\s+(thanks|dhanyavadalu)[\.]*$", "చాలా ధన్యవాదాలు."),
    (r"^(dhanyavadalu|dhanyavadaalu)[\.]*$", "ధన్యవాదాలు."),
    (r"^(namaskaram)[\.]*$", "నమస్కారం."),
    (r"^(subhodayam|shubhodayam)[\.]*$", "శుభోదయం."),
    (r"^(subharatri|shubharatri)[\.]*$", "శుభరాత్రి."),
    (r"^(bhojanam)\s+(chesava|chesara)[\?\.]*$", "భోజనం చేశారా?"),
    (r"^(naku|naaku)\s+(ardam|ardham|artham)\s+(kaledu)[\.]*$", "నాకు అర్థం కాలేదు."),
    (r"^(malli|malli\s+oka\s+sari)\s+(chepu|cheppu)[\.]*$", "దయచేసి మళ్లీ చెప్పండి."),
    (r"^(naku|naaku)\s+(nidra)\s+(vastondi|vastundi)[\.]*$", "నాకు నిద్ర వస్తోంది."),
    (r"^(naku|naaku)\s+(akali|akaliga)\s+(undi|vundi)[\.]*$", "నాకు ఆకలిగా ఉంది."),
    (r"^(naku|naaku)\s+(telugu\s+)?(chala\s+)?(istam|istham)[\.]*$", "నాకు తెలుగు భాష అంటే చాలా ఇష్టం."),
    (r"^amma\s+intlo\s+(undi|vundi)[\.]*$", "అమ్మ ఇంట్లో ఉంది."),
    (r"^nanna\s+office\s+(ki|ku)\s+velaru[\.]*$", "నాన్నగారు కార్యాలయానికి వెళ్లారు."),
    (r"^(naku|naaku)\s+(urgent\s+ga\s+)?(help|sahayam)\s+(kavali)[\.]*$", "నాకు అత్యవసరంగా సహాయం కావాలి."),
    (r"^(naku|naaku)\s+koncham\s+(help|sahayam)\s+(kavali)[\.]*$", "నాకు కొంచెం సహాయం కావాలి."),
    (r"^(ippudu|ipudu)\s+(naku|naaku)\s+call\s+(cheyi|chey|cheyandi)[\.]*$", "ఇప్పుడు నాకు ఫోన్ చేయండి."),
    (r"^sorry\s+nenu\s+late\s+ga\s+(vastanu|vasta)[\.]*$", "క్షమించండి, నేను ఆలస్యంగా వస్తాను."),

    # --- English sentence idiom patterns ---
    # Greetings with "how are you" variants
    (r"^(hi|hello|hey),?\s+(how\s+are\s+you|how\s+r\s+u|how\s+r\s+you)[?\.]*$", "నమస్కారం, ఎలా ఉన్నారు?"),
    (r"^(how\s+are\s+you|how\s+r\s+u)[?\.]*$", "ఎలా ఉన్నారు?"),
    (r"^(hi|hello|hey)[,!\.]*$", "నమస్కారం."),
    (r"^good\s+morning[,!\.]*$", "శుభోదయం."),
    (r"^good\s+evening[,!\.]*$", "శుభ సాయంత్రం."),
    (r"^good\s+night[,!\.]*$", "శుభరాత్రి."),
    (r"^(thank\s+you|thanks)[,!\.]*$", "ధన్యవాదాలు."),
    (r"^(i\s+am\s+)?sorry[,!\.]*$", "క్షమించండి."),
    (r"^(bye|goodbye|see\s+you)[,!\.]*$", "వెళ్ళొస్తాను."),
    (r"^(ok|okay|alright)[,!\.]*$", "సరే."),
    (r"^(i\s+am\s+)?(fine|good|well)[,!\.]*$", "బాగున్నాను."),

    # "Hi Ella vunnavu" / "Hi how are you" mixed patterns
    (r"^(hi|hello|hey)\s+(ela|yela|ella),?\s+(unav|unnav|vunav|vunnav|vunnavu)[?\.]*$", "నమస్కారం, ఎలా ఉన్నారు?"),

    # "Are you coming to X" patterns
    (r"^are\s+you\s+coming\s+to\s+(college|school|university|office|class|meeting|hospital|library|hostel)[?\.]*$",
     lambda m: "మీరు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "university": "విశ్వవిద్యాలయానికి", "office": "కార్యాలయానికి", "class": "తరగతికి", "meeting": "సమావేశానికి", "hospital": "వైద్యశాలకు", "library": "గ్రంథాలయానికి", "hostel": "వసతి గృహానికి"}[m.group(1).lower()] + " వస్తున్నారా?"),
    (r"^are\s+you\s+coming\s+to\s+(\w+)[?\.]*$", "మీరు వస్తున్నారా?"),
    (r"^are\s+you\s+coming[?\.]*$", "మీరు వస్తున్నారా?"),

    # "Are you going to X" patterns
    (r"^are\s+you\s+going\s+to\s+(college|school|university|office|class|meeting|hospital|library|hostel)[?\.]*$",
     lambda m: "మీరు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "university": "విశ్వవిద్యాలయానికి", "office": "కార్యాలయానికి", "class": "తరగతికి", "meeting": "సమావేశానికి", "hospital": "వైద్యశాలకు", "library": "గ్రంథాలయానికి", "hostel": "వసతి గృహానికి"}[m.group(1).lower()] + " వెళ్తున్నారా?"),
    (r"^are\s+you\s+going[?\.]*$", "మీరు వెళ్తున్నారా?"),

    # "I am coming / going to X"
    (r"^i\s+am\s+coming\s+to\s+(college|school|university|office|class|meeting|hospital|library|hostel)[\.]*$",
     lambda m: "నేను " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "university": "విశ్వవిద్యాలయానికి", "office": "కార్యాలయానికి", "class": "తరగతికి", "meeting": "సమావేశానికి", "hospital": "వైద్యశాలకు", "library": "గ్రంథాలయానికి", "hostel": "వసతి గృహానికి"}[m.group(1).lower()] + " వస్తున్నాను."),
    (r"^i\s+am\s+coming[\.]*$", "నేను వస్తున్నాను."),
    (r"^i\s+am\s+going\s+to\s+(college|school|university|office|class|meeting|hospital|library|hostel)[\.]*$",
     lambda m: "నేను " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "university": "విశ్వవిద్యాలయానికి", "office": "కార్యాలయానికి", "class": "తరగతికి", "meeting": "సమావేశానికి", "hospital": "వైద్యశాలకు", "library": "గ్రంథాలయానికి", "hostel": "వసతి గృహానికి"}[m.group(1).lower()] + " వెళ్తున్నాను."),
    (r"^i\s+am\s+going[\.]*$", "నేను వెళ్తున్నాను."),

    # "Where are you" / "Where are you going"
    (r"^where\s+are\s+you\s+going(?:\s+(today|now|tomorrow))?[?\.]*$",
     lambda m: "మీరు " + ({"today": "ఈ రోజు ", "tomorrow": "రేపు ", "now": "ఇప్పుడు "}.get(m.group(1).lower() if m.group(1) else "", "")) + "ఎక్కడికి వెళ్తున్నారు?"),
    (r"^where\s+are\s+you\s+coming\s+from[?\.]*$", "మీరు ఎక్కడి నుంచి వస్తున్నారు?"),
    (r"^where\s+are\s+you[?\.]*$", "మీరు ఎక్కడ ఉన్నారు?"),
    (r"^where\s+is\s+(he|she)[?\.]*$", "ఎక్కడ ఉన్నారు?"),

    # "When are you coming"
    (r"^when\s+are\s+you\s+coming(?:\s+(today|tomorrow))?[?\.]*$",
     lambda m: "మీరు " + ({"today": "ఈ రోజు ", "tomorrow": "రేపు "}.get(m.group(1).lower() if m.group(1) else "", "")) + "ఎప్పుడు వస్తారు?"),
    (r"^when\s+are\s+you\s+going[?\.]*$", "మీరు ఎప్పుడు వెళ్తారు?"),

    # "What are you doing"
    (r"^what\s+are\s+you\s+doing(?:\s+(today|now))?[?\.]*$",
     lambda m: "మీరు " + ({"today": "ఈ రోజు ", "now": "ఇప్పుడు "}.get(m.group(1).lower() if m.group(1) else "", "")) + "ఏమి చేస్తున్నారు?"),
    (r"^what\s+happened[?\.]*$", "ఏమయింది?"),
    (r"^what\s+is\s+this[?\.]*$", "ఇది ఏమిటి?"),
    (r"^what\s+is\s+that[?\.]*$", "అది ఏమిటి?"),

    # "I need / want" patterns
    (r"^i\s+need\s+help[\.]*$", "నాకు సహాయం కావాలి."),
    (r"^i\s+need\s+water[\.]*$", "నాకు మంచినీరు కావాలి."),
    (r"^i\s+need\s+food[\.]*$", "నాకు ఆహారం కావాలి."),
    (r"^i\s+am\s+hungry[\.]*$", "నాకు ఆకలిగా ఉంది."),
    (r"^i\s+am\s+sleepy[\.]*$", "నాకు నిద్ర వస్తోంది."),
    (r"^i\s+am\s+tired[\.]*$", "నాకు అలసటగా ఉంది."),
    (r"^i\s+don'?t\s+know[\.]*$", "నాకు తెలియదు."),
    (r"^i\s+don'?t\s+understand[\.]*$", "నాకు అర్థం కాలేదు."),
    (r"^i\s+understand[\.]*$", "నాకు అర్థమైంది."),

    # "Please" patterns
    (r"^please\s+come[\.]*$", "దయచేసి రండి."),
    (r"^please\s+wait[\.]*$", "దయచేసి ఆగండి."),
    (r"^please\s+help(\s+me)?[\.]*$", "దయచేసి సహాయం చేయండి."),
    (r"^please\s+sit(\s+down)?[\.]*$", "దయచేసి కూర్చోండి."),
    (r"^please\s+send\s+(me\s+)?the\s+(\w+)[\.]*$", "దయచేసి పంపించండి."),
    (r"^please\s+call\s+me[\.]*$", "దయచేసి నాకు ఫోన్ చేయండి."),
    (r"^please\s+tell\s+me[\.]*$", "దయచేసి నాకు చెప్పండి."),

    # "Can you / Could you" patterns
    (r"^can\s+you\s+come[?\.]*$", "మీరు రాగలరా?"),
    (r"^can\s+you\s+help(\s+me)?[?\.]*$", "మీరు సహాయం చేయగలరా?"),
    (r"^can\s+you\s+call\s+me[?\.]*$", "మీరు నాకు ఫోన్ చేయగలరా?"),
    (r"^can\s+you\s+send[?\.]*$", "మీరు పంపగలరా?"),

    # "I have" patterns
    (r"^i\s+have\s+work[\.]*$", "నాకు పని ఉంది."),
    (r"^i\s+have\s+a\s+(meeting|exam|class|interview)[\.]*$",
     lambda m: "నాకు " + {"meeting": "సమావేశం", "exam": "పరీక్ష", "class": "తరగతి", "interview": "ముఖాముఖి"}[m.group(1).lower()] + " ఉంది."),

    # "Did you" patterns
    (r"^did\s+you\s+eat[?\.]*$", "భోజనం చేశారా?"),
    (r"^did\s+you\s+come[?\.]*$", "మీరు వచ్చారా?"),
    (r"^did\s+you\s+go[?\.]*$", "మీరు వెళ్ళారా?"),
    (r"^did\s+you\s+see[?\.]*$", "మీరు చూశారా?"),
    (r"^did\s+you\s+finish[?\.]*$", "మీరు పూర్తి చేశారా?"),

    # Time expressions
    (r"^(come|see\s+you)\s+tomorrow[\.]*$", "రేపు కలుద్దాం."),
    (r"^(let'?s\s+)?meet\s+tomorrow[\.]*$", "రేపు కలుద్దాం."),
    (r"^i\s+will\s+come\s+tomorrow[\.]*$", "నేను రేపు వస్తాను."),
    (r"^i\s+will\s+come\s+now[\.]*$", "నేను ఇప్పుడు వస్తాను."),

    # Multi-sentence & movement patterns
    (r"^(hi|hello|hey|hai)\s*,?\s*(nuvvu|nuv|meeru|meru)?\s*(eroju|eeroju|ivala)\s+(college|school|office|class|university|meeting)\s*(ki|ku)\s*(vastunava|vastunnava|vastava|vastara)[?\.]*$",
     lambda m: "నమస్కారం, మీరు ఈరోజు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "office": "కార్యాలయానికి", "class": "తరగతికి", "university": "విశ్వవిద్యాలయానికి", "meeting": "సమావేశానికి"}.get(m.group(4).lower(), "కళాశాలకు") + " వస్తున్నారా?"),
    (r"^(nuvvu|nuv|meeru|meru)?\s*(eroju|eeroju|ivala)\s+(college|school|office|class|university|meeting)\s*(ki|ku)\s*(vastunava|vastunnava|vastava|vastara)[?\.]*$",
     lambda m: "మీరు ఈరోజు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "office": "కార్యాలయానికి", "class": "తరగతికి", "university": "విశ్వవిద్యాలయానికి", "meeting": "సమావేశానికి"}.get(m.group(3).lower(), "కళాశాలకు") + " వస్తున్నారా?"),
    (r"^(hi|hello|hey|hai)\s*,?\s*(nuvvu|nuv|meeru|meru)?\s*repu\s+(college|school|office|class|university|meeting)\s*(ki|ku)\s*(vastunava|vastunnava|vastava|vastara)[?\.]*$",
     lambda m: "నమస్కారం, మీరు రేపు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "office": "కార్యాలయానికి", "class": "తరగతికి", "university": "విశ్వవిద్యాలయానికి", "meeting": "సమావేశానికి"}.get(m.group(3).lower(), "కళాశాలకు") + " వస్తున్నారా?"),
    (r"^(nuvvu|nuv|meeru|meru)?\s*repu\s+(college|school|office|class|university|meeting)\s*(ki|ku)\s*(vastunava|vastunnava|vastava|vastara)[?\.]*$",
     lambda m: "మీరు రేపు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "office": "కార్యాలయానికి", "class": "తరగతికి", "university": "విశ్వవిద్యాలయానికి", "meeting": "సమావేశానికి"}.get(m.group(2).lower(), "కళాశాలకు") + " వస్తున్నారా?"),
    (r"^(hi|hello|hey)\s+(ela|yela|ella),?\s+(unav|unnav|vunav|vunnav|vunnavu)\.?\s+are\s+you\s+coming\s+to\s+(college|school|university|office)[?\.]*$",
     lambda m: "నమస్కారం, ఎలా ఉన్నారు? మీరు " + {"college": "కళాశాలకు", "school": "పాఠశాలకు", "university": "విశ్వవిద్యాలయానికి", "office": "కార్యాలయానికి"}[m.group(4).lower()] + " వస్తున్నారా?"),

    # Full movement sentence patterns (Subject + is/am/are/will + going/coming/go/come + to Destination + Time)
    (r"^(he|she|they|we|i|you)\s+(?:is|am|are|will)\s+(going|coming|go|come)\s+to\s+([a-z]+)(?:\s+(today|tomorrow|yesterday|now|morning|evening|night))?[\.\?!]*$",
     _translate_movement_pattern),

    # Movement questions (Is/Are/Will + Subject + coming/going + to Destination + Time)
    (r"^(is|are|will)\s+(he|she|they|you|we)\s+(coming|going|come|go)\s+to\s+([a-z]+)(?:\s+(today|tomorrow|yesterday|now|morning|evening|night))?[\.\?!]*$",
     _translate_q_movement),

    # Common inquiries and statements
    (r"^what\s+is\s+your\s+name[?\.]*$", "మీ పేరు ఏమిటి?"),
    (r"^how\s+much(\s+is\s+this|\s+does\s+it\s+cost)?[?\.]*$", "ఇది ఎంత?"),
    (r"^thank\s+you\s+(so\s+much|very\s+much)[!\.]*$", "చాలా ధన్యవాదాలు."),
    (r"^where\s+is\s+the\s+(college|school|hospital|library|station|bus\s+stop|hotel|restaurant)[?\.]*$",
     lambda m: {"college": "కళాశాల", "school": "పాఠశాల", "hospital": "వైద్యశాల", "library": "గ్రంథాలయం", "station": "రైల్వే స్టేషన్", "bus stop": "బస్సు ఆగే స్థలం", "hotel": "భోజనశాల", "restaurant": "భోజనశాల"}.get(m.group(1).lower(), m.group(1)) + " ఎక్కడ ఉంది?"),
    (r"^i\s+(want|need)\s+(water|food|help|money|leave|rest|medicine)[\.]*$",
     lambda m: "నాకు " + {"water": "మంచినీరు", "food": "ఆహారం", "help": "సహాయం", "money": "డబ్బులు", "leave": "సెలవు", "rest": "విశ్రాంతి", "medicine": "మందులు"}.get(m.group(2).lower(), m.group(2)) + " కావాలి."),
]

PURE_TELUGU_POSTPOSITIONS: list[tuple[str, str]] = [
    (r'(?i)(?:\b|^)(college|కాలేజీ|కళాశాల)\s*(లో|lo)(?:\s|$|[.,?!])', 'కళాశాలలో '),
    (r'(?i)(?:\b|^)(college|కాలేజీ|కళాశాల)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'కళాశాలకు '),
    (r'(?i)(?:\b|^)(college|కాలేజీ|కళాశాల)\s*(నుంచి|నుండి|nunchi|nundi)(?:\s|$|[.,?!])', 'కళాశాల నుంచి '),
    (r'(?i)(?:\b|^)(office|ఆఫీస్|ఆఫీసు|ఆఫీసుకి|కార్యాలయం)\s*(లో|lo)(?:\s|$|[.,?!])', 'కార్యాలయంలో '),
    (r'(?i)(?:\b|^)(office|ఆఫీస్|ఆఫీసు|కార్యాలయం)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'కార్యాలయానికి '),
    (r'(?i)(?:\b|^)(office|ఆఫీస్|ఆఫీసు|కార్యాలయం)\s*(నుంచి|నుండి|nunchi|nundi)(?:\s|$|[.,?!])', 'కార్యాలయం నుంచి '),
    (r'(?i)(?:\b|^)(library|లైబ్రరీ|గ్రంథాలయం)\s*(లో|lo)(?:\s|$|[.,?!])', 'గ్రంథాలయంలో '),
    (r'(?i)(?:\b|^)(library|లైబ్రరీ|గ్రంథాలయం)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'గ్రంథాలయానికి '),
    (r'(?i)(?:\b|^)(library|లైబ్రరీ|గ్రంథాలయం)\s*(నుంచి|నుండి|nunchi|nundi)(?:\s|$|[.,?!])', 'గ్రంథాలయం నుంచి '),
    (r'(?i)(?:\b|^)(hostel|హాస్టల్|వసతి గృహం)\s*(లో|lo)(?:\s|$|[.,?!])', 'వసతి గృహంలో '),
    (r'(?i)(?:\b|^)(hostel|హాస్టల్|వసతి గృహం)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'వసతి గృహానికి '),
    (r'(?i)(?:\b|^)(hospital|హాస్పిటల్|వైద్యశాల|ఆసుపత్రి)\s*(లో|lo)(?:\s|$|[.,?!])', 'వైద్యశాలలో '),
    (r'(?i)(?:\b|^)(hospital|హాస్పిటల్|వైద్యశాల|ఆసుపత్రి)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'వైద్యశాలకు '),
    (r'(?i)(?:\b|^)(meeting|మీటింగ్|సమావేశం)\s*(లో|lo)(?:\s|$|[.,?!])', 'సమావేశంలో '),
    (r'(?i)(?:\b|^)(meeting|మీటింగ్|సమావేశం)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'సమావేశానికి '),
    (r'(?i)(?:\b|^)(class|క్లాస్|తరగతి)\s*(లో|lo)(?:\s|$|[.,?!])', 'తరగతిలో '),
    (r'(?i)(?:\b|^)(class|క్లాస్|తరగతి)\s*(కి|కు|ki|ku)(?:\s|$|[.,?!])', 'తరగతికి '),
    (r'(?i)(?:\b|^)(bus|బస్సు)\s*(లో|lo)(?:\s|$|[.,?!])', 'బస్సులో '),
    (r'(?i)(?:\b|^)(train|ట్రైన్|రైలు)\s*(లో|lo)(?:\s|$|[.,?!])', 'రైలులో '),
    (r'(?i)(?:\b|^)(phone|ఫోన్|మొబైల్|చరవాణి)\s*(లో|lo)(?:\s|$|[.,?!])', 'చరవాణిలో '),
]

PURE_TELUGU_PHRASES: list[tuple[str, str]] = [
    (r'(?i)(?<![^\s.,?!;:])(important\s+interview|ఇంపార్టెంట్\s+ఇంటర్వ్యూ|ఇంపోర్టంట్\s+ఇంటర్వ్|important\s+ముఖాముఖి)(?![^\s.,?!;:])', 'ముఖ్యమైన ముఖాముఖి'),
    (r'(?i)(?<![^\s.,?!;:])(project\s+report|ప్రాజెక్ట్\s+రిపోర్ట్|కార్య\s+సాధన\s+/\s+ప్రాజెక్ట్\s+కార్యం\s+నివేదిక)(?![^\s.,?!;:])', 'కార్య నివేదిక'),
    (r'(?i)(?<![^\s.,?!;:])(class\s+notes|క్లాస్\s+నోట్స్|తరగతి\s+నోట్స్)(?![^\s.,?!;:])', 'తరగతి ముఖ్యాంశాలు'),
    (r'(?i)(?<![^\s.,?!;:])(exam\s+results|ఎగ్జామ్\s+రిజల్ట్స్)(?![^\s.,?!;:])', 'పరీక్ష ఫలితాలు'),
    (r'(?i)(?<![^\s.,?!;:])(ready\s+ayindi|రెడీ\s+అయింది|ready\s+అయింది)(?![^\s.,?!;:])', 'సిద్ధమైంది'),
    (r'(?i)(?<![^\s.,?!;:])(charge\s+ayindi|ఛార్జ్\s+అయింది|ఛార్జింగ్\s+అయింది)(?![^\s.,?!;:])', 'పూర్తిగా ఛార్జ్ అయింది'),
    (r'(?i)(?<![^\s.,?!;:])(miss\s+ayindi|మిస్\s+అయింది)(?![^\s.,?!;:])', 'తప్పిపోయింది'),
    (r'(?i)(?<![^\s.,?!;:])(send\s+cheyi|సెండ్\s+చెయ్యి|send\s+చేయి)(?![^\s.,?!;:])', 'దయచేసి పంపించండి'),
    (r'(?i)(?<![^\s.,?!;:])(submit\s+chesanu|సబ్మిట్\s+చేశాను|submit\s+చేశాను)(?![^\s.,?!;:])', 'సమర్పించాను'),
    (r'(?i)(?<![^\s.,?!;:])(complete\s+chesava|కంప్లీట్\s+చేశావా|complete\s+చేశావా)(?![^\s.,?!;:])', 'పూర్తి చేశావా'),
    (r'(?i)(?<![^\s.,?!;:])(check\s+chestanu|చెక్\s+చేస్తాను|check\s+చేస్తాను)(?![^\s.,?!;:])', 'పరిశీలిస్తాను'),
    (r'(?i)(?<![^\s.,?!;:])(call\s+cheyi|కాల్\s+చెయ్యి|call\s+చేయి)(?![^\s.,?!;:])', 'ఫోన్ చేయండి'),
    (r'(?i)(?<![^\s.,?!;:])(urgent\s+ga|అర్జెంట్\s+గా)(?![^\s.,?!;:])', 'అత్యవసరంగా'),
    (r'(?i)(?<![^\s.,?!;:])(late\s+ga|లేట్\s+గా)(?![^\s.,?!;:])', 'ఆలస్యంగా'),
    (r'(?i)(?<![^\s.,?!;:])(ivala|ఇవాళ)(?![^\s.,?!;:])', 'ఈ రోజు'),
    (r'(?i)(?<![^\s.,?!;:])(koncham\s+help|కొంచెం\s+హెల్ప్)(?![^\s.,?!;:])', 'కొంచెం సహాయం'),
    (r'(?i)(?<![^\s.,?!;:])(help\s+kavali|హెల్ప్\s+కావాలి)(?![^\s.,?!;:])', 'సహాయం కావాలి'),
]

PURE_TELUGU_WORDS: dict[str, str] = {
    # Education
    "college": "కళాశాల", "కాలేజీ": "కళాశాల",
    "school": "పాఠశాల", "స్కూల్": "పాఠశాల",
    "university": "విశ్వవిద్యాలయం", "యూనివర్సిటీ": "విశ్వవిద్యాలయం",
    "library": "గ్రంథాలయం", "లైబ్రరీ": "గ్రంథాలయం",
    "class": "తరగతి", "క్లాస్": "తరగతి",
    "hostel": "వసతి గృహం", "హాస్టల్": "వసతి గృహం",
    "exam": "పరీక్ష", "ఎగ్జామ్": "పరీక్ష",
    "results": "ఫలితాలు", "రిజల్ట్స్": "ఫలితాలు",
    "student": "విద్యార్థి", "స్టూడెంట్": "విద్యార్థి",
    "teacher": "ఉపాధ్యాయుడు", "టీచర్": "ఉపాధ్యాయుడు",
    "professor": "ఆచార్యుడు", "ప్రొఫెసర్": "ఆచార్యుడు",
    "presentation": "సమర్పణ", "ప్రెజెంటేషన్": "సమర్పణ",
    "report": "నివేదిక", "రిపోర్ట్": "నివేదిక",
    "interview": "ముఖాముఖి", "ఇంటర్వ్యూ": "ముఖాముఖి",
    "important": "ముఖ్యమైన", "ఇంపార్టెంట్": "ముఖ్యమైన",
    "project": "కార్యం", "ప్రాజెక్ట్": "కార్యం",
    "book": "పుస్తకం", "books": "పుస్తకాలు",
    "homework": "ఇంటి పని", "assignment": "కార్యకలాపం",

    # Workplace
    "office": "కార్యాలయం", "ఆఫీస్": "కార్యాలయం", "ఆఫీసు": "కార్యాలయం",
    "meeting": "సమావేశం", "మీటింగ్": "సమావేశం",
    "salary": "జీతం", "శాలరీ": "జీతం",
    "leave": "సెలవు", "లీవ్": "సెలవు",
    "manager": "నిర్వాహకుడు", "మేనేజర్": "నిర్వాహకుడు",
    "team": "బృందం", "టీమ్": "బృందం",
    "company": "సంస్థ", "కంపెనీ": "సంస్థ",
    "job": "ఉద్యోగం", "జాబ్": "ఉద్యోగం",
    "work": "పని", "వర్క్": "పని",

    # Tech
    "phone": "చరవాణి", "ఫోన్": "చరవాణి",
    "mobile": "చరవాణి", "మొబైల్": "చరవాణి",
    "computer": "సంగణకం", "కంప్యూటర్": "సంగణకం",
    "internet": "అంతర్జాలం", "ఇంటర్నెట్": "అంతర్జాలం",
    "wifi": "వైర్‌లెస్ అంతర్జాలం", "వైఫై": "వైర్‌లెస్ అంతర్జాలం",
    "message": "సందేశం", "మెసేజ్": "సందేశం",
    "password": "రహస్య సంకేతం", "పాస్‌వర్డ్": "రహస్య సంకేతం",
    "call": "ఫోన్",
    "email": "విద్యుత్ సందేశం",

    # Transit & City
    "train": "రైలు", "ట్రైన్": "రైలు",
    "bus": "బస్సు",
    "ticket": "ప్రయాణ చీటీ", "టికెట్": "ప్రయాణ చీటీ",
    "hospital": "వైద్యశాల", "హాస్పిటల్": "వైద్యశాల",
    "doctor": "వైద్యుడు", "డాక్టర్": "వైద్యుడు",
    "hotel": "భోజనశాల", "హోటల్": "భోజనశాల",
    "restaurant": "భోజనశాల", "రెస్టారెంట్": "భోజనశాల",
    "movie": "చలనచిత్రం", "మూవీ": "చలనచిత్రం",
    "theatre": "చిత్ర మందిరం", "థియేటర్": "చిత్ర మందిరం",
    "shop": "దుకాణం", "market": "సంత",
    "city": "నగరం", "village": "గ్రామం",
    "road": "రహదారి", "house": "ఇల్లు", "home": "ఇల్లు",

    # Conversation & Etiquette
    "hello": "నమస్కారం", "హెల్లో": "నమస్కారం",
    "hi": "నమస్కారం",
    "hey": "నమస్కారం",
    "thanks": "ధన్యవాదాలు", "థాంక్స్": "ధన్యవాదాలు",
    "sorry": "క్షమించండి", "సారీ": "క్షమించండి",
    "please": "దయచేసి", "ప్లీజ్": "దయచేసి",
    "help": "సహాయం", "హెల్ప్": "సహాయం",
    "problem": "సమస్య", "ప్రాబ్లెమ్": "సమస్య",
    "time": "సమయం", "టైమ్": "సమయం",
    "food": "ఆహారం", "ఫుడ్": "ఆహారం",
    "water": "మంచినీరు", "వాటర్": "మంచినీరు",
    "money": "డబ్బులు", "మనీ": "డబ్బులు",
    "urgent": "అత్యవసరం", "అర్జెంట్": "అత్యవసరం",
    "ok": "సరే", "okay": "సరే", "alright": "సరే",
    "yes": "అవును", "no": "లేదు",
    "friend": "మిత్రుడు", "friends": "మిత్రులు",
    "brother": "సోదరుడు", "sister": "సోదరి",
    "father": "తండ్రి", "mother": "తల్లి",
    "sir": "అయ్యా", "madam": "అమ్మా",
    "good": "మంచి", "bad": "చెడు",
    "big": "పెద్ద", "small": "చిన్న",
    "new": "కొత్త", "old": "పాత",
    "happy": "సంతోషం", "sad": "బాధ",
    "ready": "సిద్ధం", "రెడీ": "సిద్ధం",
    "late": "ఆలస్యం", "లేట్": "ఆలస్యం",
    "fast": "వేగం", "slow": "నెమ్మది",
    "right": "సరైన", "wrong": "తప్పు",
    "true": "నిజం", "false": "అబద్ధం",
    "easy": "సులభం", "difficult": "కష్టం",
    "nice": "బాగుంది", "beautiful": "అందమైన",

    # Common English Pronouns
    "i": "నేను", "me": "నాకు", "my": "నా", "mine": "నాది",
    "you": "మీరు", "your": "మీ", "yours": "మీది",
    "he": "అతను", "him": "అతనికి", "his": "అతని",
    "she": "ఆమె", "her": "ఆమెకి",
    "we": "మనం", "us": "మాకు", "our": "మన",
    "they": "వారు", "them": "వారికి", "their": "వారి",
    "it": "అది", "its": "దాని",
    "this": "ఇది", "that": "అది",
    "these": "ఇవి", "those": "అవి",

    # Common English Verbs & Verb Forms
    "coming": "వస్తున్న", "going": "వెళ్తున్న",
    "come": "రా", "go": "వెళ్ళు",
    "came": "వచ్చాను", "went": "వెళ్ళాను",
    "doing": "చేస్తున్న", "done": "పూర్తయింది",
    "eating": "తింటున్న", "eat": "తిను",
    "sleeping": "నిద్రపోతున్న", "sleep": "నిద్రపో",
    "reading": "చదువుతున్న", "read": "చదువు",
    "writing": "రాస్తున్న", "write": "రాయి",
    "working": "పని చేస్తున్న",
    "sitting": "కూర్చున్న", "sit": "కూర్చో",
    "standing": "నిలబడ్డ", "stand": "నిలబడు",
    "running": "పరుగెత్తున్న", "run": "పరుగెత్తు",
    "walking": "నడుస్తున్న", "walk": "నడువు",
    "talking": "మాట్లాడుతున్న", "talk": "మాట్లాడు",
    "waiting": "ఎదురుచూస్తున్న", "wait": "ఆగు",
    "give": "ఇవ్వు", "gave": "ఇచ్చాను",
    "take": "తీసుకో", "took": "తీసుకున్నాను",
    "see": "చూడు", "saw": "చూశాను",
    "know": "తెలుసు", "knew": "తెలిసింది",
    "want": "కావాలి", "need": "అవసరం",
    "like": "ఇష్టం", "love": "ప్రేమ",
    "think": "అనుకుంటున్న", "said": "చెప్పాను",
    "tell": "చెప్పు", "told": "చెప్పాను",
    "ask": "అడుగు", "asked": "అడిగాను",
    "send": "పంపు", "sent": "పంపాను",
    "buy": "కొను", "bought": "కొన్నాను",
    "finish": "పూర్తి చేయు", "finished": "పూర్తయింది",
    "start": "మొదలు పెట్టు", "started": "మొదలైంది",
    "stop": "ఆపు", "stopped": "ఆపాను",
    "open": "తెరువు", "close": "మూయు",
    "try": "ప్రయత్నించు", "tried": "ప్రయత్నించాను",
    "learn": "నేర్చుకో", "teach": "నేర్పించు",
    "play": "ఆడు", "playing": "ఆడుతున్న",
    "bring": "తీసుకురా", "keep": "ఉంచు",
    "make": "చేయు", "made": "చేశాను",
    "get": "పొందు", "got": "పొందాను",
    "find": "కనుగొను", "found": "కనుగొన్నాను",
    "live": "నివసించు", "living": "నివసిస్తున్న",
    "feel": "అనిపిస్తోంది", "listen": "విను",
    "remember": "గుర్తుంచుకో", "forget": "మర్చిపో",
    "forgot": "మర్చిపోయాను",
    "understand": "అర్థమైంది",
    "prepare": "సిద్ధం చేయు", "cancel": "రద్దు చేయు",
    "confirm": "నిర్ధారించు", "submit": "సమర్పించు",
    "check": "పరిశీలించు", "complete": "పూర్తి చేయు",
    "use": "ఉపయోగించు", "change": "మార్చు",

    # Common English Prepositions & Conjunctions
    "to": "కి", "from": "నుంచి", "in": "లో", "at": "వద్ద",
    "with": "తో", "for": "కోసం", "on": "మీద", "of": "యొక్క",
    "by": "ద్వారా", "about": "గురించి", "after": "తర్వాత",
    "before": "ముందు", "between": "మధ్య",
    "and": "మరియు", "or": "లేదా", "but": "కానీ",
    "because": "ఎందుకంటే", "so": "కాబట్టి",
    "also": "కూడా", "only": "మాత్రమే",
    "if": "ఒకవేళ", "then": "అప్పుడు",

    # Common English Question Words
    "what": "ఏమిటి", "where": "ఎక్కడ", "when": "ఎప్పుడు",
    "why": "ఎందుకు", "how": "ఎలా", "who": "ఎవరు",
    "which": "ఏది",

    # Auxiliaries & Common English Words
    "am": "ఉన్నాను", "is": "ఉంది", "are": "ఉన్నారు",
    "was": "ఉన్నాను", "were": "ఉన్నారు",
    "have": "ఉంది", "has": "ఉంది", "had": "ఉన్నది",
    "will": "చేస్తాను", "can": "గలరు", "could": "గలరు",
    "should": "చేయాలి", "would": "చేస్తారు",
    "must": "తప్పక", "may": "కావచ్చు",
    "not": "కాదు", "don't": "వద్దు", "didn't": "లేదు",
    "won't": "చేయను", "can't": "లేదు",
    "very": "చాలా", "much": "చాలా", "many": "చాలా",
    "more": "ఇంకా", "less": "తక్కువ",
    "all": "అన్నీ", "some": "కొన్ని", "any": "ఏదైనా",
    "every": "ప్రతి", "each": "ప్రతి ఒక్కటి",
    "here": "ఇక్కడ", "there": "అక్కడ",
    "now": "ఇప్పుడు", "today": "ఈ రోజు",
    "tomorrow": "రేపు", "yesterday": "నిన్న",
    "morning": "ఉదయం", "evening": "సాయంత్రం",
    "night": "రాత్రి", "afternoon": "మధ్యాహ్నం",
    "always": "ఎల్లప్పుడూ", "never": "ఎప్పుడూ",
    "sometimes": "కొన్నిసార్లు", "often": "తరచుగా",
    "already": "ఇప్పటికే", "still": "ఇంకా",
    "again": "మళ్ళీ", "just": "ఇప్పుడే",
    "really": "నిజంగా", "actually": "వాస్తవంగా",
    "too": "కూడా", "enough": "సరిపడా",

    # Time & Calendar
    "day": "రోజు", "week": "వారం", "month": "నెల", "year": "సంవత్సరం",
    "hour": "గంట", "minute": "నిమిషం",

    # Articles & Determiners (suppress or translate contextually)
    "the": "", "a": "ఒక", "an": "ఒక",

    # People
    "person": "వ్యక్తి", "people": "ప్రజలు",
    "man": "మనిషి", "woman": "స్త్రీ",
    "boy": "అబ్బాయి", "girl": "అమ్మాయి",
    "child": "పిల్లవాడు", "children": "పిల్లలు",
    "baby": "శిశువు",
    "name": "పేరు",
}

PURE_TELUGU_ROMAN_MAP: dict[str, str] = {
    "hi": "హలో", "hai": "హలో", "hello": "హలో", "hey": "హలో",
    "naku": "నాకు", "naaku": "నాకు", "nenu": "నేను", "nuvvu": "నువ్వు", "nuvu": "నువ్వు", "meeru": "మీరు",
    "atanu": "అతను", "aame": "ఆమె", "idi": "ఇది", "adi": "అది", "manamu": "మనం", "manam": "మనం",
    "ivala": "ఈ రోజు", "eroju": "ఈరోజు", "eeroju": "ఈరోజు", "repu": "రేపు", "ninna": "నిన్న", "ippudu": "ఇప్పుడు", "ipudu": "ఇప్పుడు",
    "appudu": "అప్పుడు", "eppudu": "ఎప్పుడు", "ekkada": "ఎక్కడ", "ikkada": "ఇక్కడ", "akkada": "అక్కడ",
    "ela": "ఎలా", "ella": "ఎలా", "yela": "ఎలా", "enti": "ఏంటి", "enduku": "ఎందుకు",
    "undi": "ఉంది", "ledu": "లేదు", "unnav": "ఉన్నావు", "unnava": "ఉన్నావా", "unnanu": "ఉన్నాను", "unnaru": "ఉన్నారు",
    "unav": "ఉన్నావు", "vunnav": "ఉన్నావు", "vunnavu": "ఉన్నావు", "vunav": "ఉన్నావు", "vunnvu": "ఉన్నావు", "unnvu": "ఉన్నావు", "unvu": "ఉన్నావు", "vunvu": "ఉన్నావు", "vunnu": "ఉన్నావు",
    "vellali": "వెళ్లాలి", "velthanu": "వెళ్తాను", "vachanu": "వచ్చాను", "vastanu": "వస్తాను", "vastunnanu": "వస్తున్నాను",
    "vastunnava": "వస్తున్నావా", "vastunava": "వస్తున్నావా", "vellanu": "వెళ్లాను", "vastunnaru": "వస్తున్నారు", "velthunnaru": "వెళ్తున్నారు",
    "vastava": "వస్తావా", "vastara": "వస్తారా", "vastunara": "వస్తున్నారు", "velthava": "వెళ్తావా",
    "chala": "చాలా", "baga": "బాగా", "pani": "పని", "help": "సహాయం",
    "amma": "అమ్మ", "ammaki": "అమ్మకి", "nanna": "నాన్న", "nannagaru": "నాన్నగారు",
    "chesanu": "చేశాను", "chey": "చేయి", "cheyali": "చేయాలి", "cheyandi": "చేయండి", "chestanu": "చేస్తాను", "chesaru": "చేశారు",
    "chestunnav": "చేస్తున్నావు", "chestunnaru": "చేస్తున్నారు",
    "bagundi": "బాగుంది", "kavali": "కావాలి", "istam": "ఇష్టం", "istham": "ఇష్టం", "ardham": "అర్థం", "ardam": "అర్థం",
    "nidra": "నిద్ర", "akali": "ఆకలి", "sahayam": "సహాయం", "intlo": "ఇంట్లో", "intiki": "ఇంటికి",
    "bhojanam": "భోజనం", "namaskaram": "నమస్కారం", "dhanyavadalu": "ధన్యవాదాలు",
    "ra": "రా", "randi": "రండి", "velu": "వెళ్ళు", "vellu": "వెళ్ళు", "velandi": "వెళ్ళండి",
    "tinu": "తిను", "tinnav": "తిన్నావా", "tinnara": "తిన్నారా",
    "chudu": "చూడు", "chusav": "చూశావా", "chusara": "చూశారా",
    "paduko": "పడుకో", "levu": "లేవు", "levandi": "లేవండి",
    "telugu": "తెలుగు", "english": "ఆంగ్లం",
    "lo": "లో", "ki": "కి", "ku": "కు", "tho": "తో", "kosam": "కోసం",
    "nundi": "నుండి", "nunchi": "నుంచి", "mida": "మీద",
    "mariyu": "మరియు", "leda": "లేదా", "kani": "కానీ", "kuda": "కూడా",
    "bayata": "బయట", "lopala": "లోపల",
    "ayna": "అయినా", "ayindi": "అయింది", "avthundi": "అవుతుంది",
    "sare": "సరే", "okay": "సరే",
    "emiti": "ఏమిటి", "evaru": "ఎవరు", "edi": "ఏది",
    "udayam": "ఉదయం", "sayantram": "సాయంత్రం", "ratri": "రాత్రి",
    "koncham": "కొంచెం", "chinna": "చిన్న", "pedda": "పెద్ద",
    "oka": "ఒక", "rendu": "రెండు", "moodu": "మూడు", "nalugu": "నాలుగు", "aidu": "ఐదు",
}


def _has_latin(text: str) -> bool:
    """Return True if text contains any ASCII Latin letters."""
    return bool(re.search(r'[a-zA-Z]', text))


def _try_idiom_match(text: str) -> str | None:
    """Try to match text against PURE_TELUGU_IDIOMS across raw, normalized, and folded variants."""
    raw_s = text.strip()
    norm_s = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    norm_s = re.sub(r"\s+", " ", norm_s)
    folded = fold_phonetics(text)

    candidates = [raw_s, norm_s, folded]
    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        for pat, pure in PURE_TELUGU_IDIOMS:
            m = re.search(pat, c, re.IGNORECASE)
            if m:
                if callable(pure):
                    return pure(m)
                return pure
    return None


def _translate_segment_words(text: str) -> str:
    """Word-level translation pipeline for a single text segment."""
    # Postposition fusions
    for pat, rep in PURE_TELUGU_POSTPOSITIONS:
        text = re.sub(pat, rep, text)

    # Compound phrase fusions
    for pat, rep in PURE_TELUGU_PHRASES:
        text = re.sub(pat, rep, text)

    # Word-by-word substitution of loanwords
    for loan, pure in PURE_TELUGU_WORDS.items():
        text = re.sub(rf'(?i)(?<![^\s.,?!;:]){re.escape(loan)}(?![^\s.,?!;:])', pure, text)

    # Romanized Telugu roots
    words = text.split()
    converted_words = []
    for w in words:
        clean_w = re.sub(r'[^\w]', '', w.lower())
        if clean_w in PURE_TELUGU_ROMAN_MAP:
            te_term = PURE_TELUGU_ROMAN_MAP[clean_w]
            # Preserve trailing punctuation
            trail = ''
            stripped = w
            while stripped and stripped[-1] in '.,?!;:':
                trail = stripped[-1] + trail
                stripped = stripped[:-1]
            converted_words.append(te_term + trail)
        else:
            converted_words.append(w)
    text = ' '.join(converted_words)
    return text


def _transliterate_remaining_latin(text: str) -> str:
    """Transliterate any remaining Latin-script words to Telugu script as a final fallback."""
    if not _has_latin(text):
        return text
    words = text.split()
    result = []
    for w in words:
        # Preserve punctuation at word edges
        trail = ''
        lead = ''
        stripped = w
        while stripped and stripped[0] in '.,?!;:()[]{}':
            lead += stripped[0]
            stripped = stripped[1:]
        while stripped and stripped[-1] in '.,?!;:()[]{}':
            trail = stripped[-1] + trail
            stripped = stripped[:-1]

        if stripped and re.search(r'[a-zA-Z]', stripped):
            # Check if it's a pure Latin word (not already partially Telugu)
            translit = fallback_english_to_telugu(stripped)
            result.append(lead + translit + trail)
        else:
            result.append(w)
    return ' '.join(result)


def translate_to_pure_telugu(raw: str, normalized: str = "") -> str:
    """Translates Telugu, Tanglish, or Code-Mixed text into authentic, literary Pure Telugu (అచ్చ తెలుగు).

    Strategy:
    1. Try full-text idiom match (including multi-sentence patterns).
    2. Split into sentences and translate each independently:
       a. Try sentence-level idiom match
       b. Word-level substitution (postpositions, phrases, loanwords, roots)
       c. Transliterate remaining Latin words into Telugu script
    """
    raw_clean = raw.strip()
    if not raw_clean:
        return ""

    # 1. Try full-text idiom match first (catches multi-sentence patterns)
    full_match = _try_idiom_match(raw_clean)
    if full_match:
        return full_match

    if normalized:
        full_match = _try_idiom_match(normalized)
        if full_match:
            return full_match

    # 2. Split into sentences and translate each independently
    # Split on sentence-ending punctuation while preserving it
    text = normalized if normalized else raw_clean
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        sentences = [text]

    translated_parts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 2a. Try sentence-level idiom match
        sent_match = _try_idiom_match(sentence)
        if sent_match:
            translated_parts.append(sent_match)
            continue

        # 2b. Word-level translation pipeline
        result = _translate_segment_words(sentence)

        # 2c. Transliterate remaining Latin words into Telugu script
        result = _transliterate_remaining_latin(result)

        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        # Remove empty tokens from suppressed articles
        result = re.sub(r'\s{2,}', ' ', result).strip()

        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        if result:
            translated_parts.append(result)

    final = ' '.join(translated_parts)
    final = re.sub(r'\s+', ' ', final).strip()
    return final


# Mappings from literary Pure Telugu words to standard conversational loanwords in Telugu script
PURE_TO_STANDARD_TE: dict[str, str] = {
    "కళాశాలలో": "కాలేజీలో",
    "కళాశాలకు": "కాలేజీకి",
    "కళాశాల నుంచి": "కాలేజీ నుంచి",
    "కళాశాల": "కాలేజీ",
    "కార్యాలయంలో": "ఆఫీస్‌లో",
    "కార్యాలయానికి": "ఆఫీస్‌కి",
    "కార్యాలయం": "ఆఫీస్",
    "కార్య నివేదిక": "ప్రాజెక్ట్ రిపోర్ట్",
    "కార్యం": "ప్రాజెక్ట్",
    "సిద్ధమైంది": "రెడీ అయింది",
    "సిద్ధం": "రెడీ",
    "ముఖాముఖి": "ఇంటర్వ్యూ",
    "చరవాణి": "ఫోన్",
    "సంగణకం": "కంప్యూటర్",
    "చిత్ర మందిరానికి": "థియేటర్‌కి",
    "చిత్ర మందిరం": "థియేటర్",
    "భోజనశాలకు": "హోటల్‌కి",
    "భోజనశాల": "హోటల్",
    "వైద్యశాలకు": "హాస్పిటల్‌కి",
    "వైద్యశాల": "హాస్పిటల్",
    "గ్రంథాలయానికి": "లైబ్రరీకి",
    "గ్రంథాలయంలో": "లైబ్రరీలో",
    "గ్రంథాలయం": "లైబ్రరీ",
    "వసతి గృహానికి": "హాస్టల్‌కి",
    "వసతి గృహం": "హాస్టల్",
    "నమస్కారం,": "హలో,",
    "నమస్కారం.": "హలో.",
    "నమస్కారం": "హలో",
}

TE_TO_CONV_TANGLISH: dict[str, str] = {
    "హలో,": "Hello,",
    "హలో.": "Hello.",
    "హలో": "Hello",
    "నమస్కారం,": "Namaskaram,",
    "నమస్కారం.": "Namaskaram.",
    "ఎలా ఉన్నారు?": "ela unnaru?",
    "ఎలా ఉన్నావు?": "ela unnavu?",
    "కాలేజీకి": "college ki",
    "కాలేజీలో": "college lo",
    "కాలేజీ": "college",
    "ఆఫీస్‌కి": "office ki",
    "ఆఫీస్‌లో": "office lo",
    "ఆఫీస్": "office",
    "వస్తున్నారా?": "vastunnara?",
    "వస్తున్నావా?": "vastunnava?",
    "వెళ్తున్నారా?": "velthunnara?",
    "వెళ్తున్నాడు.": "velthunnadu.",
    "వస్తున్నది.": "vastunnadi.",
    "చేస్తున్నారు?": "chestunnaru?",
    "ఎక్కడికి": "ekkadiki",
    "ఎక్కడ": "ekkada",
    "ఉన్నారు?": "unnaru?",
    "ఉంది?": "undi?",
    "ఉంది.": "undi.",
    "ప్రాజెక్ట్ రిపోర్ట్": "project report",
    "రెడీ అయింది.": "ready ayindi.",
    "ముఖ్యమైన": "important",
    "ఇంటర్వ్యూ": "interview",
    "శుభోదయం.": "Good morning.",
    "శుభరాత్రి.": "Good night.",
    "నాకు": "naku",
    "సహాయం కావాలి.": "help kavali.",
    "మంచినీరు కావాలి.": "water kavali.",
    "మీరు": "meeru",
    "అతను": "atanu",
    "ఆమె": "aame",
    "రేపు": "repu",
    "ఈ రోజు": "e roju",
    "ఇప్పుడు": "ippudu",
}


def translate_to_standard_telugu_script(pure_te_text: str, raw: str = "", normalized: str = "") -> str:
    """Renders natural, conversational Telugu entirely in Telugu script with standard loanwords."""
    if not pure_te_text:
        return ""
    text = pure_te_text
    for pure, std in PURE_TO_STANDARD_TE.items():
        text = text.replace(pure, std)
    return text


def translate_to_standard_tanglish(std_te_text: str, raw: str = "") -> str:
    """Renders conversational Romanized Tanglish from standard conversational Telugu."""
    if not std_te_text:
        return ""
    s = std_te_text
    for te, tang in TE_TO_CONV_TANGLISH.items():
        s = s.replace(te, tang)
    if has_telugu(s):
        s = romanize_text(s)
    return s



def detect_modality(text: str, english_vocab: set[str] | None = None, telugu_lexicon: dict[str, str] | None = None) -> dict[str, Any]:
    """Inspects text and determines script composition, token language tags, and input modality."""
    clean = clean_text(text)
    if not clean:
        return {
            "modality": "empty",
            "label": "Empty Input",
            "confidence": 0.0,
            "telugu_char_pct": 0.0,
            "latin_char_pct": 0.0,
            "token_count": 0,
            "tokens": []
        }

    tokens = clean.split()
    te_chars = sum(1 for c in clean if "\u0C00" <= c <= "\u0C7F")
    latin_chars = sum(1 for c in clean if c.isascii() and c.isalpha())
    total_alpha = max(te_chars + latin_chars, 1)

    te_char_pct = round((te_chars / total_alpha) * 100, 1)
    latin_char_pct = round((latin_chars / total_alpha) * 100, 1)

    token_analyses = []
    te_script_tokens = 0
    en_tokens = 0
    te_roman_tokens = 0
    univ_tokens = 0

    eng_set = english_vocab or set(LOANWORD_TO_TELUGU.keys())
    tel_lex = telugu_lexicon or {}

    for tok in tokens:
        pre, core, post = split_edge_punct(tok)
        if not core:
            univ_tokens += 1
            token_analyses.append({"token": tok, "script": "punct", "lang": "UNIV"})
            continue

        low = core.lower()
        if has_telugu(core):
            te_script_tokens += 1
            token_analyses.append({"token": tok, "script": "telugu", "lang": "TE", "is_script": True})
        elif core.isascii() and core.isalpha():
            is_common_en = low in COMMON_ENGLISH_WORDS
            is_loan = is_common_en or (eng_set and low in eng_set) or low in LOANWORD_TO_TELUGU
            # Only count as Telugu Roman if it actually maps to Telugu script or is in Telugu gloss
            maps_to_telugu = bool(tel_lex and low in tel_lex and has_telugu(tel_lex[low]))
            is_te_roman = (maps_to_telugu or low in TELUGU_TO_ENGLISH_GLOSS) and not is_common_en
            if is_loan and not is_te_roman:
                en_tokens += 1
                token_analyses.append({"token": tok, "script": "latin", "lang": "EN", "is_loanword": True})
            elif is_te_roman and not is_loan:
                te_roman_tokens += 1
                token_analyses.append({"token": tok, "script": "latin", "lang": "TE_ROMAN", "is_tanglish": True})
            else:
                if is_common_en or low in {"college", "office", "interview", "class", "wifi", "router", "project", "presentation", "train", "ticket"}:
                    en_tokens += 1
                    token_analyses.append({"token": tok, "script": "latin", "lang": "EN"})
                elif is_te_roman:
                    te_roman_tokens += 1
                    token_analyses.append({"token": tok, "script": "latin", "lang": "TE_ROMAN"})
                else:
                    # Default latin
                    token_analyses.append({"token": tok, "script": "latin", "lang": "EN" if is_loan else "TE_ROMAN"})
                    if is_loan:
                        en_tokens += 1
                    else:
                        te_roman_tokens += 1
        else:
            univ_tokens += 1
            token_analyses.append({"token": tok, "script": "symbol", "lang": "UNIV"})

    # Classify overall input modality
    if te_script_tokens > 0 and en_tokens == 0 and te_roman_tokens == 0:
        modality = "telugu_script_pure"
        label = "Native Telugu Script"
        conf = 0.98
    elif te_script_tokens > 0 and (en_tokens > 0 or te_roman_tokens > 0):
        modality = "telugu_english_biscriptal"
        label = "Bi-scriptal Code-Mixed (Telugu Script + English)"
        conf = 0.95
    elif te_script_tokens == 0 and en_tokens > 0 and te_roman_tokens == 0 and te_chars == 0:
        modality = "pure_english"
        label = "Pure English"
        conf = 0.92
    elif te_roman_tokens > 0 and en_tokens > 0:
        total_words = en_tokens + te_roman_tokens
        if te_chars == 0 and en_tokens / max(1, total_words) >= 0.75 and te_roman_tokens <= 1:
            modality = "pure_english"
            label = "Pure English"
            conf = 0.92
        else:
            modality = "romanized_tanglish"
            label = "Romanized Tanglish (Telugu-English Code-Mixed)"
            conf = 0.96
    elif te_roman_tokens > 0 and en_tokens == 0:
        modality = "romanized_telugu_pure"
        label = "Romanized Telugu (Tanglish)"
        conf = 0.94
    else:
        modality = "mixed_general"
        label = "Multi-Modal Code-Mixed"
        conf = 0.85

    return {
        "modality": modality,
        "label": label,
        "confidence": conf,
        "telugu_char_pct": te_char_pct,
        "latin_char_pct": latin_char_pct,
        "token_count": len(tokens),
        "tokens": token_analyses,
    }


class OmniProcessor:
    """Universal auto-detecting processor serving all possible transformations."""

    def __init__(self, registry=None):
        self.registry = registry

    def process(self, text: str, engine: str = "auto", beam_size: int = 4, n_best: int = 1) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw = text.strip()
        if not raw:
            return {"error": "Empty input text"}

        # 1. Step 1: Auto-Detect Input Modality & Token Characteristics
        eng_set = self.registry.rule_baseline.english if (self.registry and self.registry.rule_baseline) else None
        tel_lex = self.registry.rule_baseline.lex if (self.registry and self.registry.rule_baseline) else None
        detected = detect_modality(raw, english_vocab=eng_set, telugu_lexicon=tel_lex)

        # 2. Step 2: Compute Option 1 - Standard Normalized Representation (SOTA Neural)
        normalized_text = ""
        norm_conf = 0.90
        engine_used = "auto"
        model_used = "production_sota"

        if detected["modality"] == "pure_english":
            normalized_text = raw
            engine_used = "identity_preserve"
            model_used = "english_passthrough"
            norm_conf = 1.0
        elif detected["modality"] == "telugu_script_pure":
            normalized_text = raw
            engine_used = "identity_preserve"
            model_used = "telugu_passthrough"
            norm_conf = 1.0
        elif self.registry is not None:
            try:
                hyps, engine_used, model_used = self.registry.normalize([raw], None, engine=engine, beam_size=beam_size, n_best=n_best)
                if hyps and hyps[0]:
                    normalized_text = hyps[0][0]["text"]
                    norm_conf = hyps[0][0]["confidence"]
            except Exception as e:
                # Safe fallback
                if self.registry.rule_baseline:
                    normalized_text = self.registry.rule_baseline.normalize(raw)
                    engine_used = "rule_fallback"
                    model_used = "rule_baseline"
                else:
                    normalized_text = raw
        else:
            normalized_text = raw


        # 3. Step 3: Compute English Translation, Pure English & Prescribed Meaning
        english_translation_text = translate_to_english(raw, normalized=normalized_text)
        pure_english_text = translate_to_pure_english(raw, normalized=normalized_text)

        # If neural GPU translation is available and requested:
        if engine in ("neural", "auto") and self.registry and detected["modality"] != "pure_english":
            neural_preds = self.registry.translate([raw], model_name="translation_sota")
            if neural_preds and neural_preds[0] and len(neural_preds[0].strip()) > 2:
                neural_en = neural_preds[0].strip()
                if not any(c in neural_en for c in ["<unk>", "<pad>"]):
                    english_translation_text = neural_en
                    pure_english_text = elevate_to_pure_english(neural_en)

        prescribed_meaning = english_translation_text

        # 4. Step 4: Compute Authentic Pure Telugu Translation (అచ్చ తెలుగు)
        pure_telugu_text = translate_to_pure_telugu(raw, normalized=normalized_text)

        # 5. Step 5: Compute Full Meaningful Telugu Script Rendering (Standard Conversational Telugu)
        all_telugu_text = translate_to_standard_telugu_script(pure_telugu_text, raw=raw, normalized=normalized_text)

        # 6. Step 6: Compute Full Romanized / Tanglish Rendering
        all_roman_text = translate_to_standard_tanglish(all_telugu_text, raw=raw)

        # 7. Step 7: Compute Option 6 - English Semantic Gloss / Vocabulary Meaning
        gloss_tokens = []
        for tok in raw.split():
            pre, core, post = split_edge_punct(tok)
            low = core.lower()
            if low in TELUGU_TO_ENGLISH_GLOSS:
                gloss_tokens.append(pre + f"[{TELUGU_TO_ENGLISH_GLOSS[low]}]" + post)
            elif core in TELUGU_TO_ENGLISH_GLOSS:
                gloss_tokens.append(pre + f"[{TELUGU_TO_ENGLISH_GLOSS[core]}]" + post)
            elif low in LOANWORD_TO_TELUGU:
                gloss_tokens.append(pre + core + post)
            elif tel_lex and low in tel_lex:
                # Telugu root found, look up transliterated word
                te_word = tel_lex[low]
                if te_word in TELUGU_TO_ENGLISH_GLOSS:
                    gloss_tokens.append(pre + f"[{TELUGU_TO_ENGLISH_GLOSS[te_word]}]" + post)
                else:
                    gloss_tokens.append(pre + core + post)
            else:
                gloss_tokens.append(tok)
        english_gloss_text = " ".join(gloss_tokens)

        # 8. Step 8: Detailed Token Diagnostics
        detailed_tokens = []
        for orig_tok, norm_tok in zip(raw.split(), normalized_text.split()):
            is_te = has_telugu(norm_tok)
            is_en = norm_tok.isascii() and any(c.isalpha() for c in norm_tok)
            tag = "TE" if is_te else ("EN" if is_en else "SYM")
            gloss = TELUGU_TO_ENGLISH_GLOSS.get(orig_tok.lower(), TELUGU_TO_ENGLISH_GLOSS.get(norm_tok, ""))
            detailed_tokens.append({
                "source": orig_tok,
                "normalized": norm_tok,
                "telugu_script": LOANWORD_TO_TELUGU.get(orig_tok.lower(), norm_tok),
                "romanized": romanize_word(norm_tok),
                "tag": tag,
                "gloss": gloss
            })

        latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "input": raw,
            "detected": detected,
            "recommended": {
                "text": normalized_text,
                "confidence": norm_conf,
                "engine": engine_used,
                "model": model_used
            },
            "options": {
                "normalized_code_mixed": normalized_text,
                "english_translation": english_translation_text,
                "pure_english": pure_english_text,
                "pure_telugu": pure_telugu_text,
                "all_telugu_script": all_telugu_text,
                "all_romanized_tanglish": all_roman_text,
                "english_gloss": english_gloss_text,
                "prescribed_meaning": prescribed_meaning,
            },
            "tokens": detailed_tokens,
            "latency_ms": latency
        }
