from src.evaluation.metrics import cer, wer, exact_match, chrf, bleu, edit_distance, token_accuracy


def test_edit_distance_basic():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0


def test_cer_and_wer_perfect():
    assert cer(["hi there"], ["hi there"]) == 0.0
    assert wer(["a b c"], ["a b c"]) == 0.0


def test_exact_match():
    assert exact_match(["a", "b"], ["a", "c"]) == 0.5


def test_chrf_and_bleu_range():
    p, r = ["నాకు ఇవాళ college కి వెళ్ళాలి"], ["నాకు ఇవాళ college కి వెళ్ళాలి"]
    assert chrf(p, r) > 99.0
    assert bleu(p, r) > 99.0
    assert 0.0 <= bleu(["completely different"], ["నాకు ఇవాళ college కి వెళ్ళాలి"]) < 5.0


def test_token_accuracy():
    assert token_accuracy(["a b c"], ["a b c"]) == 1.0
    assert 0.0 < token_accuracy(["a x c"], ["a b c"]) < 1.0
