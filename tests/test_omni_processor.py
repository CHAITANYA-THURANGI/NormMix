"""Tests for the OmniProcessor multi-modal auto-detecting pipeline."""
from __future__ import annotations

import pytest
from src.pipeline.omni_processor import detect_modality, OmniProcessor, LOANWORD_TO_TELUGU, TELUGU_TO_ENGLISH_GLOSS


def test_detect_modality_tanglish():
    res = detect_modality("naku ivala college lo important interview undi")
    assert res["modality"] == "romanized_tanglish"
    assert res["confidence"] >= 0.90
    assert res["latin_char_pct"] > 80.0


def test_detect_modality_pure_telugu():
    res = detect_modality("నాకు ఇవాళ ముఖ్యమైన పని ఉంది")
    assert res["modality"] == "telugu_script_pure"
    assert res["confidence"] >= 0.95
    assert res["telugu_char_pct"] > 80.0


def test_detect_modality_pure_english():
    res = detect_modality("Where are you going today? Please send me the project report.")
    assert res["modality"] == "pure_english"
    assert res["confidence"] >= 0.90


def test_detect_modality_biscriptal():
    res = detect_modality("నాకు ivala college లో interview ఉంది")
    assert res["modality"] == "telugu_english_biscriptal"
    assert res["confidence"] >= 0.90


def test_omni_processor_generates_all_options():
    op = OmniProcessor(registry=None)
    out = op.process("naku ivala college lo interview undi")
    assert "options" in out
    opts = out["options"]
    assert "normalized_code_mixed" in opts
    assert "all_telugu_script" in opts
    assert "all_romanized_tanglish" in opts
    assert "english_gloss" in opts

    # Verify loanword is transliterated in all_telugu_script
    assert "కాలేజీ" in opts["all_telugu_script"]

    # Verify English gloss contains translations
    assert "[to me]" in opts["english_gloss"]
    assert "[today]" in opts["english_gloss"]

    # Verify English translation is generated
    assert "english_translation" in opts
    assert len(opts["english_translation"]) > 0

    # Verify Pure Telugu translation is generated
    assert "pure_telugu" in opts
    assert len(opts["pure_telugu"]) > 0

    # Verify Prescribed Meaning is generated
    assert "prescribed_meaning" in opts
    assert len(opts["prescribed_meaning"]) > 0


def test_english_translation_conversational():
    op = OmniProcessor(registry=None)
    
    # 1. User test case: conversational Tanglish greeting
    res_tanglish = op.process("hello ella vunnavuu")
    assert res_tanglish["options"]["english_translation"] == "Hello, how are you?"

    # 2. Telugu script equivalent
    res_script = op.process("హెల్లో ఎల్లా ఉన్నావూ")
    assert res_script["options"]["english_translation"] == "Hello, how are you?"

    # 3. Conversational check-ins
    res_how = op.process("ela unnav")
    assert res_how["options"]["english_translation"] == "How are you?"

    # 4. Complex sentence with preposition and time
    res_clause = op.process("naku ivala college lo important interview undi")
    assert res_clause["options"]["english_translation"] == "I have an important interview in college today."

    # 5. Pure English preservation
    res_en = op.process("Where are you going today?")
    assert res_en["options"]["english_translation"] == "Where are you going today?"


def test_pure_telugu_translation():
    op = OmniProcessor(registry=None)

    # 1. Conversational Tanglish greeting -> Pure literary Telugu
    res_greet = op.process("hello ella vunnavuu")
    assert res_greet["options"]["pure_telugu"] == "నమస్కారం, ఎలా ఉన్నారు?"

    # 2. English loanword postposition fusion: college lo -> కళాశాలలో, project report -> కార్య నివేదిక, ready ayindi -> సిద్ధమైంది
    res_proj = op.process("college lo project report ready ayindi")
    assert "కళాశాలలో" in res_proj["options"]["pure_telugu"]
    assert "కార్య నివేదిక" in res_proj["options"]["pure_telugu"]
    assert "సిద్ధమైంది" in res_proj["options"]["pure_telugu"]

    # 3. Telugu script input with loanwords
    res_te = op.process("కాలేజీలో ప్రాజెక్ట్ రిపోర్ట్ రెడీ అయింది")
    assert "కళాశాలలో" in res_te["options"]["pure_telugu"]
    assert "కార్య నివేదిక" in res_te["options"]["pure_telugu"]
    assert "సిద్ధమైంది" in res_te["options"]["pure_telugu"]

    # 4. Office interview: office ki -> కార్యాలయానికి, important interview -> ముఖ్యమైన ముఖాముఖి
    res_work = op.process("office ki important interview undi")
    assert "కార్యాలయానికి" in res_work["options"]["pure_telugu"]
    assert "ముఖాముఖి" in res_work["options"]["pure_telugu"]

    # 5. Multi-sentence mixed conversational input
    res_multi = op.process("Hi Ella, vunnavu. Are you coming to college?")
    assert res_multi["options"]["pure_telugu"] == "నమస్కారం, ఎలా ఉన్నారు? మీరు కళాశాలకు వస్తున్నారా?"

    # 6. Direct English greetings and requests
    res_gm = op.process("Good morning")
    assert res_gm["options"]["pure_telugu"] == "శుభోదయం."
    res_help = op.process("I need help")
    assert res_help["options"]["pure_telugu"] == "నాకు సహాయం కావాలి."

    # 7. Movement patterns with SOV grammar
    res_mov = op.process("He is going to college tomorrow")
    assert res_mov["options"]["pure_telugu"] == "అతను రేపు కళాశాలకు వెళ్తున్నాడు."
    res_q = op.process("Are you going to office?")
    assert res_q["options"]["pure_telugu"] == "మీరు కార్యాలయానికి వెళ్తున్నారా?"

    # 8. Meaningful Telugu script and prescribed meaning verification
    assert res_multi["options"]["all_telugu_script"] == "హలో, ఎలా ఉన్నారు? మీరు కాలేజీకి వస్తున్నారా?"
    assert res_multi["options"]["prescribed_meaning"] == "Hello, how are you? Are you coming to college?"
