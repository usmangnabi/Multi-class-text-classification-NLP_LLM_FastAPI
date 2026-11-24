# Tests for the NLP logic (no model files needed)
import numpy as np  # numpy arrays

from app.services.preprocessing import clean_for_model, clean_text  # cleaning functions
from app.services.similarity import cosine_similarity  # similarity function
from app.services.tokenization import BPETokenizer, WordPieceTokenizer  # tokenizers

TEXTS = ["football match football team", "cooking recipe pasta recipe", "football team wins match"]  # tiny corpus


def test_clean_text_removes_noise():
    assert clean_text("Hello, WORLD! 123 <b>Hi</b>") == "hello world hi"  # lowercase, no symbols


def test_clean_for_model_removes_stopwords():
    assert clean_for_model("The team is great") == "team great"  # stopwords removed


def test_cosine_similarity_values():
    a = np.array([1.0, 0.0])  # vector A
    assert cosine_similarity(a, a) == 1.0  # same vector = 1
    assert cosine_similarity(a, np.array([0.0, 1.0])) == 0.0  # 90 degrees = 0
    assert cosine_similarity(a, np.zeros(2)) == 0.0  # zero vector is safe


def test_bpe_learns_and_handles_unknown():
    tokenizer = BPETokenizer(num_merges=10)  # small tokenizer
    tokenizer.train(TEXTS)  # learn
    assert "<unk>" in tokenizer.tokenize("xyz")  # unseen letters become <unk>
    assert "".join(tokenizer.tokenize("football")).replace("</w>", "") == "football"  # nothing is lost


def test_wordpiece_learns_and_handles_unknown():
    tokenizer = WordPieceTokenizer(num_merges=10)  # small tokenizer
    tokenizer.train(TEXTS)  # learn
    assert "[UNK]" in tokenizer.tokenize("xyz")  # unseen letters make the word unknown
    assert tokenizer.tokenize("football")[0] != "[UNK]"  # known word works


def test_tokenizer_save_and_load(tmp_path):
    tokenizer = BPETokenizer(num_merges=5)  # small tokenizer
    tokenizer.train(TEXTS)  # learn
    tokenizer.save(tmp_path / "bpe.json")  # save to a temporary file
    loaded = BPETokenizer.load(tmp_path / "bpe.json")  # load it back
    assert loaded.tokenize("football") == tokenizer.tokenize("football")  # same result
