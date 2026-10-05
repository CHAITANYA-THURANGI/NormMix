from src.tokenization.char_tokenizer import CharTokenizer
from src.tokenization.base import EOS_ID, SOS_ID


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.build(["hello నేను", "world వెళ్ళాలి"])
    ids = tok.encode("hello నేను", add_sos=True, add_eos=True)
    assert ids[0] == SOS_ID and ids[-1] == EOS_ID
    assert tok.decode(ids) == "hello నేను"


def test_unk_for_unseen_char():
    tok = CharTokenizer.build(["abc"])
    ids = tok.encode("abz")
    assert tok.decode(ids) != "abz"  # 'z' unseen -> unk, dropped on decode


def test_to_from_dict():
    tok = CharTokenizer.build(["hi"])
    tok2 = CharTokenizer.from_dict(tok.to_dict())
    assert tok2.itos == tok.itos
