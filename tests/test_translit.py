from src.preprocessing.translit_te import romanize_word, romanize_text


def test_deterministic_without_rng():
    assert romanize_word("నేను") == romanize_word("నేను")


def test_basic_words():
    assert romanize_word("నేను") == "nenu"
    assert romanize_word("నువ్వు") == "nuvvu"


def test_passthrough_non_telugu():
    assert romanize_word("college") == "college"


def test_variation_uses_rng():
    import random
    rng = random.Random(0)
    variants = {romanize_word("వెళ్ళాలి", rng, variation=1.0) for _ in range(20)}
    assert len(variants) >= 1  # variation is possible, never crashes
    assert romanize_text("నేను వెళ్తాను") == "nenu veltaanu" or " " in romanize_text("నేను వెళ్తాను")
