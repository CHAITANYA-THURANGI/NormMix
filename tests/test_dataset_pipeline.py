from src.config import load_config
from src.datasets.pipeline import build_dataset, load_split


def test_build_dataset_no_leakage_and_nonempty(tmp_path):
    cfg = load_config()
    cfg["data"]["variants_per_sentence"] = 3
    stats = build_dataset(cfg, out_dir=tmp_path)
    assert stats["leakage"]["ok"]
    for name in ["train", "valid", "test"]:
        assert len(load_split(tmp_path, name)) > 0


def test_gold_demo_not_in_train(tmp_path):
    cfg = load_config()
    cfg["data"]["variants_per_sentence"] = 3
    build_dataset(cfg, out_dir=tmp_path)
    train_srcs = {r["src"] for r in load_split(tmp_path, "train")}
    gold_srcs = {r["src"] for r in load_split(tmp_path, "gold_demo")}
    assert not (train_srcs & gold_srcs)
