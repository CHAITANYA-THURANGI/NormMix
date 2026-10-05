from src.datasets.splits import assign_split, split_rows, dedupe_pairs, leakage_report, filter_noisy


def test_assign_split_is_deterministic():
    assert assign_split("g1") == assign_split("g1")


def test_split_rows_no_leakage_and_group_integrity():
    rows = [{"src": f"s{i}", "tgt": f"t{i % 20}", "group": f"g{i % 20}"} for i in range(200)]
    splits = split_rows(rows, valid_frac=0.1, test_frac=0.1)
    rep = leakage_report(splits)
    assert rep["ok"]
    seen = {}
    for name, rs in splits.items():
        for r in rs:
            seen.setdefault(r["group"], set()).add(name)
    assert all(len(v) == 1 for v in seen.values())


def test_heldout_words_routes_to_unseen_split():
    rows = [{"src": "a", "tgt": "canteen ki vellali", "group": "g1"}, {"src": "b", "tgt": "hi", "group": "g2"}]
    splits = split_rows(rows, heldout_words=["canteen"])
    assert len(splits["test_unseen_words"]) == 1
    assert all("canteen" not in r["tgt"] for r in splits["train"] + splits["valid"] + splits["test"])


def test_dedupe_removes_exact_duplicates():
    rows = [{"src": "a", "tgt": "x"}, {"src": "a", "tgt": "x"}, {"src": "a", "tgt": "y"}]
    kept, info = dedupe_pairs(rows)
    assert len(kept) == 2 and info["removed_duplicates"] == 1


def test_filter_noisy_drops_empty_and_extreme_ratio():
    rows = [{"src": "", "tgt": "x"}, {"src": "hi", "tgt": "hi"}, {"src": "a" * 2, "tgt": "b" * 40}]
    kept, n = filter_noisy(rows, min_len=1)
    assert n == 2 and kept == [{"src": "hi", "tgt": "hi"}]
