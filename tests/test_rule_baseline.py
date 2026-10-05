from src.models.rule_baseline import RuleBaseline


def test_fit_and_normalize_known_pair():
    rows = [{"src": "college ki vellali", "tgt": "college కి వెళ్ళాలి"},
            {"src": "college ki vellali", "tgt": "college కి వెళ్ళాలి"}]
    rb = RuleBaseline().fit(rows)
    assert rb.normalize("college ki vellali") == "college కి వెళ్ళాలి"


def test_save_and_load_roundtrip(tmp_path):
    rows = [{"src": "hi", "tgt": "హాయ్"}]
    rb = RuleBaseline().fit(rows)
    p = tmp_path / "rb.json"
    rb.save(p)
    rb2 = RuleBaseline.load(p)
    assert rb2.normalize("hi") == "హాయ్"
