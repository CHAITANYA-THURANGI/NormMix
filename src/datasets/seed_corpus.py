"""Hand-written seed corpus: clean Telugu-English code-mixed targets from templates + free sentences.

IMPORTANT (honesty note): this is a small, template-driven TOY corpus. It exists so that the whole
pipeline (data -> train -> evaluate -> API) runs end-to-end on a laptop. Scores obtained on data
generated from it are NOT evidence of real-world performance. Replace/extend it with real
data as described in docs/dataset.md before drawing research conclusions.

Target convention (documented design decision): Telugu words in Telugu script, English words in
lower-case Latin, code-mixing preserved, and a Telugu case marker attached to an English word is
written as a separate token ("college కి"), so language boundaries fall on whitespace.
"""
from __future__ import annotations

import hashlib
import unicodedata

PLACE = ["college", "hostel", "office", "library", "canteen", "mall", "hospital", "bank", "lab", "hotel", "class"]
EVENT = ["exam", "meeting", "class", "party", "interview", "movie"]
EVENT_LIKED = ["movie", "party", "class"]
OBJ = ["phone", "laptop", "charger", "bike", "ticket", "notes"]
OBJ_DEV = ["phone", "laptop", "charger", "bike"]
WORK = ["project", "assignment", "report", "form"]
TECH = ["internet", "wifi"]
VEH = ["bus", "train"]

TEMPLATES: list[tuple[str, str, list[str]]] = [
    ("P1", "నేను రేపు {X} కి వెళ్తాను", PLACE),
    ("P2", "నువ్వు ఇప్పుడు {X} లో ఉన్నావా?", PLACE),
    ("P3", "మనం సాయంత్రం {X} కి వెళ్దాం", PLACE),
    ("P4", "నాకు ఇవాళ {X} కి వెళ్ళాలి", PLACE),
    ("P5", "వాడు {X} లో ఉన్నాడు", PLACE),
    ("P6", "{X} నుంచి ఇప్పుడే వచ్చాను", PLACE),
    ("P7", "మీరు రేపు {X} కి రండి", PLACE),
    ("P8", "{X} ఎక్కడ ఉంది?", PLACE),
    ("E1", "రేపు {X} ఉంది", EVENT),
    ("E2", "{X} ఎప్పుడు మొదలవుతుంది?", EVENT),
    ("E3", "ఈ రోజు {X} లేదు", EVENT),
    ("E4", "నీకు {X} నచ్చిందా?", EVENT_LIKED),
    ("O1", "నా {X} ఎక్కడ పెట్టావు?", OBJ),
    ("O2", "నాకు {X} కావాలి", OBJ),
    ("O3", "నా {X} లో problem ఉంది", OBJ_DEV),
    ("O4", "ఈ {X} చాలా బాగుంది", OBJ_DEV),
    ("W1", "నేను {X} submit చేశాను", WORK),
    ("W2", "మీరు {X} send చేయండి", WORK),
    ("W3", "నేను రేపు {X} check చేస్తాను", WORK),
    ("W4", "నువ్వు {X} complete చేశావా?", WORK),
    ("W5", "{X} ఇంకా complete కాలేదు", WORK),
    ("T1", "{X} లో problem ఉంది", TECH),
    ("T2", "ఇక్కడ {X} రావడం లేదు", TECH),
    ("V1", "{X} ఎప్పుడు వస్తుంది?", VEH),
    ("V2", "నేను {X} లో వస్తున్నాను", VEH),
    ("V3", "{X} miss అయింది", VEH),
]

FREE_SENTENCES = [
    "నాకు తెలుగు చాలా ఇష్టం", "నువ్వు ఎక్కడ ఉన్నావు?", "class కి వస్తున్నావా?", "నువ్వు ఎప్పుడు వస్తావు?",
    "ఇప్పుడు ఏమి చేస్తున్నావు?", "భోజనం చేశావా?", "నేను ఇప్పుడు ఇంటికి వెళ్తున్నాను", "రేపు కలుద్దాం",
    "నాకు అర్థం కాలేదు", "మళ్ళీ చెప్పు", "నాకు నిద్ర వస్తోంది", "నువ్వు బాగున్నావా?", "నేను బాగున్నాను",
    "నాకు ఆకలిగా ఉంది", "ఇక్కడ చాలా వేడిగా ఉంది", "నేను ఇంకా ఇంట్లో ఉన్నాను", "నాకు చాలా పని ఉంది",
    "ఈ రోజు నాకు time లేదు", "నీకు ఈ విషయం తెలుసా?", "నేను చెప్పింది విన్నావా?", "మనం రేపు ఉదయం కలుద్దాం",
    "ఆ పని ఇంకా అవ్వలేదు", "నాకు కొంచెం help కావాలి", "ఇప్పుడు నాకు call చెయ్యి", "అమ్మ ఇంట్లో ఉంది",
    "నాన్న ఆఫీసుకి వెళ్ళారు", "చాలా thanks", "సరే, రేపు మాట్లాడదాం", "sorry, నేను late గా వస్తాను",
    "మా friends అందరూ party కి వస్తున్నారు", "meeting కి late గా వస్తాను", "project report ready అయింది",
    "class notes send చెయ్యి", "exam results ఎప్పుడు వస్తాయి?", "phone charge అయింది", "office లో meeting ఉంది",
    "college bus miss అయింది", "lab record complete చేశావా?",
]


def group_key(tgt: str) -> str:
    """Stable id of a clean target. All noisy variants of one target share it (leakage guard)."""
    norm = unicodedata.normalize("NFC", " ".join(tgt.split())).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def build_clean_corpus() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for tid, tpl, pool in TEMPLATES:
        for x in pool:
            tgt = tpl.replace("{X}", x)
            if tgt in seen:
                continue
            seen.add(tgt)
            rows.append({"tgt": tgt, "template_id": tid, "slot": x})
    for i, s in enumerate(FREE_SENTENCES):
        if s not in seen:
            seen.add(s)
            rows.append({"tgt": s, "template_id": f"F{i + 1}", "slot": None})
    for r in rows:
        r["group"] = group_key(r["tgt"])
    return rows
