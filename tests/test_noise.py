import random

from src.augmentation.noise import NoiseConfig, SourceNoiser


def test_noiser_is_deterministic_given_seed():
    noiser = SourceNoiser(NoiseConfig())
    a = noiser.make_source("నాకు ఇవాళ college కి వెళ్ళాలి", random.Random(7))
    b = noiser.make_source("నాకు ఇవాళ college కి వెళ్ళాలి", random.Random(7))
    assert a == b


def test_noiser_changes_text_most_of_the_time():
    noiser = SourceNoiser(NoiseConfig(strength=2.0))
    tgt = "నాకు ఇవాళ college కి వెళ్ళాలి"
    changed = sum(noiser.make_source(tgt, random.Random(i)) != tgt for i in range(30))
    assert changed >= 20


def test_no_crash_on_short_text():
    noiser = SourceNoiser()
    assert isinstance(noiser.make_source("హాయ్", random.Random(1)), str)
