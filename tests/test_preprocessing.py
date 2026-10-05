from src.preprocessing.text_cleaning import clean_text, has_telugu, split_edge_punct, split_protected, clamp_repeats


def test_clamp_repeats():
    assert clamp_repeats("sooooo good") == "sooo good"
    assert clamp_repeats("aaa", max_run=3) == "aaa"


def test_zero_width_and_whitespace():
    assert clean_text("hi\u200b  there\u00a0!") == "hi there !"


def test_has_telugu():
    assert has_telugu("నేను") and not has_telugu("nenu")


def test_split_edge_punct():
    assert split_edge_punct("college?!") == ("", "college", "?!")
    assert split_edge_punct("...hi...") == ("...", "hi", "...")
    assert split_edge_punct("నేను,") == ("", "నేను", ",")


def test_split_protected_keeps_url_and_mention():
    parts = split_protected("check https://x.com/a and @bob now")
    protected = [p for p, is_p in parts if is_p]
    assert "https://x.com/a" in protected and "@bob" in protected
