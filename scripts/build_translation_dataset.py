#!/usr/bin/env python3
"""Build training and validation datasets for Telugu/Tanglish -> English translation on GPU.

Generates paired examples:
  src: Telugu script, Romanized Tanglish, or Code-mixed text
  tgt: Natural English translation or Pure English translation
  meta: information about kind, style (conversational vs pure), etc.

Outputs:
  data/processed/train_translation.jsonl
  data/processed/valid_translation.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Base seed templates with English & Pure English mappings
SEED_TRANSLATION_TEMPLATES = [
    # Places
    ("నేను రేపు {X} కి వెళ్తాను", "I will go to {X} tomorrow", "I shall proceed to {X} tomorrow"),
    ("నువ్వు ఇప్పుడు {X} లో ఉన్నావా?", "Are you in {X} right now?", "Are you presently located at {X}?"),
    ("మనం సాయంత్రం {X} కి వెళ్దాం", "Let's go to {X} this evening", "We shall visit {X} this evening"),
    ("నాకు ఇవాళ {X} కి వెళ్ళాలి", "I need to go to {X} today", "I must travel to {X} today"),
    ("వాడు {X} లో ఉన్నాడు", "He is in {X}", "He is presently situated in {X}"),
    ("{X} నుంచి ఇప్పుడే వచ్చాను", "I just arrived from {X}", "I have just returned from {X}"),
    ("మీరు రేపు {X} కి రండి", "Please come to {X} tomorrow", "You are cordially invited to visit {X} tomorrow"),
    ("{X} ఎక్కడ ఉంది?", "Where is {X}?", "Could you please tell me where {X} is located?"),
    
    # Events
    ("రేపు {X} ఉంది", "There is an {X} tomorrow", "An {X} has been scheduled for tomorrow"),
    ("{X} ఎప్పుడు మొదలవుతుంది?", "When does the {X} start?", "At what time is the {X} scheduled to commence?"),
    ("ఈ రోజు {X} లేదు", "There is no {X} today", "No {X} is scheduled for today"),
    ("నీకు {X} నచ్చిందా?", "Did you like the {X}?", "Did you find the {X} enjoyable?"),

    # Objects
    ("నా {X} ఎక్కడ పెట్టావు?", "Where did you put my {X}?", "Where have you placed my {X}?"),
    ("నాకు {X} కావాలి", "I need my {X}", "I require my {X}"),
    ("నా {X} లో problem ఉంది", "There is a problem with my {X}", "An issue has occurred with my {X}"),
    ("ఈ {X} చాలా బాగుంది", "This {X} is very good", "This {X} is of superior quality"),

    # Work & Tasks
    ("నేను {X} submit చేశాను", "I submitted the {X}", "I have successfully submitted the {X}"),
    ("మీరు {X} send చేయండి", "Please send the {X}", "Please transmit the {X} at your earliest convenience"),
    ("నేను రేపు {X} check చేస్తాను", "I will check the {X} tomorrow", "I shall review the {X} tomorrow"),
    ("నువ్వు {X} complete చేశావా?", "Did you complete the {X}?", "Have you finalized the {X}?"),
    ("{X} ఇంకా complete కాలేదు", "The {X} is not completed yet", "The {X} remains incomplete at this stage"),

    # Tech & Transport
    ("{X} లో problem ఉంది", "There is a problem with {X}", "A technical difficulty has arisen with {X}"),
    ("ఇక్కడ {X} రావడం లేదు", "The {X} is not working here", "The {X} connectivity is unavailable here"),
    ("{X} ఎప్పుడు వస్తుంది?", "When will the {X} arrive?", "At what time is the {X} scheduled to arrive?"),
    ("నేను {X} లో వస్తున్నాను", "I am coming by {X}", "I am traveling via {X}"),
    ("{X} miss అయింది", "I missed the {X}", "I unfortunately missed the scheduled {X}"),
]

PLACE_ITEMS = ["college", "hostel", "office", "library", "canteen", "hospital", "bank", "lab", "hotel", "class"]
EVENT_ITEMS = ["exam", "meeting", "class", "party", "interview", "presentation"]
OBJ_ITEMS = ["phone", "laptop", "charger", "bike", "ticket", "notes"]
WORK_ITEMS = ["project", "assignment", "report", "file", "document"]
TECH_ITEMS = ["internet", "wifi"]
VEH_ITEMS = ["bus", "train", "flight", "cab"]

FREE_TRANSLATION_PAIRS = [
    ("మీరు చాలా బాగున్నారు, కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్",
     "You look very good, convert this sentence into English.",
     "You are doing exceptionally well; please translate this sentence into English."),
    ("కన్వర్ట్ దిస్ ఆన్ సెంట్న్స్ ఇన్ టు ఇంగ్లీష్",
     "Convert this sentence into English.",
     "Please translate this sentence into English."),
    ("మీరు చాలా బాగున్నారు",
     "You look very good.",
     "You are doing exceptionally well."),
    ("నువ్వు చాలా బాగున్నావు",
     "You look great.",
     "You are in excellent spirits."),
    ("నేను చాలా బాగున్నాను",
     "I am doing very well.",
     "I am in very good health and spirits."),
    ("నాకు తెలుగు చాలా ఇష్టం",
     "I like Telugu very much.",
     "I possess a deep appreciation for the Telugu language."),
    ("నువ్వు ఎక్కడ ఉన్నావు?",
     "Where are you?",
     "Where are you currently located?"),
    ("మీరు ఎక్కడ ఉన్నారు?",
     "Where are you?",
     "Could you kindly disclose your present location?"),
    ("class కి వస్తున్నావా?",
     "Are you coming to class?",
     "Will you be attending the class?"),
    ("నువ్వు ఎప్పుడు వస్తావు?",
     "When will you come?",
     "At what time may we anticipate your arrival?"),
    ("ఇప్పుడు ఏమి చేస్తున్నావు?",
     "What are you doing now?",
     "What are you engaged in at the present moment?"),
    ("భోజనం చేశావా?",
     "Did you have your meal?",
     "Have you partaken of your meal?"),
    ("నేను ఇప్పుడు ఇంటికి వెళ్తున్నాను",
     "I am going home now.",
     "I am presently en route to my residence."),
    ("రేపు కలుద్దాం",
     "See you tomorrow.",
     "We look forward to convening tomorrow."),
    ("నాకు అర్థం కాలేదు",
     "I did not understand.",
     "I was unable to comprehend that."),
    ("మళ్ళీ చెప్పు",
     "Tell me again.",
     "Kindly repeat what you said."),
    ("నాకు నిద్ర వస్తోంది",
     "I feel sleepy.",
     "I am experiencing drowsiness."),
    ("నువ్వు బాగున్నావా?",
     "Are you doing well?",
     "Are you in sound health?"),
    ("నాకు ఆకలిగా ఉంది",
     "I am hungry.",
     "I am experiencing an appetite."),
    ("ఇక్కడ చాలా వేడిగా ఉంది",
     "It is very hot here.",
     "The temperature is considerably warm here."),
    ("నాకు చాలా పని ఉంది",
     "I have a lot of work.",
     "I have a substantial amount of work to attend to."),
    ("ఈ రోజు నాకు time లేదు",
     "I don't have time today.",
     "My schedule does not permit any spare time today."),
    ("నీకు ఈ విషయం తెలుసా?",
     "Do you know about this?",
     "Are you aware of this matter?"),
    ("మనం రేపు ఉదయం కలుద్దాం",
     "Let's meet tomorrow morning.",
     "We shall assemble tomorrow morning."),
    ("నాకు కొంచెం help కావాలి",
     "I need some help.",
     "I would be grateful for your assistance."),
    ("ఇప్పుడు నాకు call చెయ్యి",
     "Call me now.",
     "Please contact me by telephone at once."),
    ("చాలా thanks",
     "Thanks a lot.",
     "I express my sincere and heartfelt gratitude."),
    ("ధన్యవాదాలు",
     "Thank you.",
     "Thank you very much. I appreciate your kind assistance."),
    ("నమస్కారం",
     "Hello.",
     "Greetings. It is a pleasure to connect with you."),
    ("సరే, రేపు మాట్లాడదాం",
     "Okay, let's talk tomorrow.",
     "Very well, we shall converse tomorrow."),
    ("sorry, నేను late గా వస్తాను",
     "Sorry, I will be late.",
     "Please accept my apologies; my arrival will be slightly delayed."),
    ("మా friends అందరూ party కి వస్తున్నారు",
     "All my friends are coming to the party.",
     "All of our companions shall be attending the celebration."),
    ("project report ready అయింది",
     "The project report is ready.",
     "The project documentation has been fully prepared."),
    ("phone charge అయింది",
     "The phone is charged.",
     "The mobile device has been fully recharged."),
    ("office లో meeting ఉంది",
     "There is a meeting in the office.",
     "A formal meeting has been convened at the office."),
    ("college bus miss అయింది",
     "I missed the college bus.",
     "I inadvertently missed the college transport bus."),
    ("lab record complete చేశావా?",
     "Did you complete the lab record?",
     "Have you finalized the laboratory record?"),
    ("నాకు కాఫీ కావాలి",
     "I want coffee.",
     "I would like to have a cup of coffee."),
    ("దయచేసి సహాయం చేయండి",
     "Please help me.",
     "Kindly extend your valued assistance."),
    ("మీ పేరు ఏమిటి?",
     "What is your name?",
     "May I inquire what your name is?"),
    ("ఇది చాలా ముఖ్యమైనది",
     "This is very important.",
     "This matter is of paramount importance."),
    ("త్వరగా రండి",
     "Come quickly.",
     "Please arrive promptly without delay."),
]


def romanize_tanglish_simple(te_text: str) -> str:
    """Generate typical romanized Tanglish version of a Telugu string."""
    from src.preprocessing.translit_te import romanize_text
    return romanize_text(te_text)


def build_datasets():
    random.seed(42)
    pairs = []

    # 1. Expand templates with pool items
    for tpl, eng_tpl, pure_tpl in SEED_TRANSLATION_TEMPLATES:
        if "{X}" in tpl:
            # Determine appropriate category
            pool = PLACE_ITEMS
            if any(w in tpl for w in [" మొదలవుతుంది", " నచ్చిందా"]):
                pool = EVENT_ITEMS
            elif any(w in tpl for w in [" పెట్టావు", " కావాలి"]):
                pool = OBJ_ITEMS
            elif any(w in tpl for w in ["submit", "send", "check", "complete"]):
                pool = WORK_ITEMS
            elif any(w in tpl for w in ["రావడం లేదు"]):
                pool = TECH_ITEMS
            elif any(w in tpl for w in ["వస్తుంది", "miss", "లో వస్తున్నాను"]):
                pool = VEH_ITEMS
            
            for item in pool:
                te_src = tpl.replace("{X}", item)
                en_tgt = eng_tpl.replace("{X}", item)
                pure_tgt = pure_tpl.replace("{X}", item)
                pairs.append((te_src, en_tgt, pure_tgt))
        else:
            pairs.append((tpl, eng_tpl, pure_tpl))

    # 2. Add all free translation pairs
    for te_src, en_tgt, pure_tgt in FREE_TRANSLATION_PAIRS:
        pairs.append((te_src, en_tgt, pure_tgt))

    # 3. Build diverse training rows:
    # - Native Telugu script -> English
    # - Native Telugu script -> Pure English
    # - Romanized Tanglish -> English
    # - Romanized Tanglish -> Pure English
    # - Multi-variant noisy Tanglish -> English & Pure English
    dataset_rows = []

    for te_src, en_tgt, pure_tgt in pairs:
        # A. Telugu script -> English
        dataset_rows.append({"src": te_src, "tgt": en_tgt, "source": "seed_te", "style": "conversational"})
        # B. Telugu script -> Pure English
        dataset_rows.append({"src": te_src, "tgt": pure_tgt, "source": "seed_te", "style": "pure_english"})

        # C. Romanized Tanglish variants
        rom_src = romanize_tanglish_simple(te_src)
        dataset_rows.append({"src": rom_src, "tgt": en_tgt, "source": "tanglish_canonical", "style": "conversational"})
        dataset_rows.append({"src": rom_src, "tgt": pure_tgt, "source": "tanglish_canonical", "style": "pure_english"})

        # D. Common conversational variants (capitalization, colloquial spelling)
        rom_lower = rom_src.lower()
        if rom_lower != rom_src:
            dataset_rows.append({"src": rom_lower, "tgt": en_tgt, "source": "tanglish_lower", "style": "conversational"})
        
        # Subtle variations (aa -> a, ee -> i, oo -> u)
        subtle = rom_lower.replace("aa", "a").replace("ee", "i").replace("oo", "u")
        if subtle != rom_lower:
            dataset_rows.append({"src": subtle, "tgt": en_tgt, "source": "tanglish_colloquial", "style": "conversational"})
            dataset_rows.append({"src": subtle, "tgt": pure_tgt, "source": "tanglish_colloquial", "style": "pure_english"})

    # Shuffle
    random.shuffle(dataset_rows)

    # Split into train (90%) and valid (10%)
    split_idx = int(len(dataset_rows) * 0.90)
    train_rows = dataset_rows[:split_idx]
    valid_rows = dataset_rows[split_idx:]

    out_train = ROOT / "data" / "processed" / "train_translation.jsonl"
    out_valid = ROOT / "data" / "processed" / "valid_translation.jsonl"

    out_train.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_valid, "w", encoding="utf-8") as f:
        for r in valid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_rows)} translation training pairs in {out_train}")
    print(f"Generated {len(valid_rows)} translation validation pairs in {out_valid}")


if __name__ == "__main__":
    build_datasets()
