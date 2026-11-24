# Tests for the API endpoints (run with `pytest`)
import pytest  # pytest tools
from fastapi.testclient import TestClient  # fake client to call our API

from app import config  # file locations
from app.main import app  # our application
from app.services import classification  # to change the key during a test

client = TestClient(app)  # create the test client

# Some tests need files created by the notebooks; skip them if the files are missing
needs_models = pytest.mark.skipif(not config.CLASSIFIER_PATH.exists(), reason="run the Colab notebooks first")
needs_tokenizers = pytest.mark.skipif(not config.BPE_PATH.exists(), reason="run notebook 02 first")


def test_health():
    response = client.get("/health")  # call /health
    assert response.status_code == 200  # request worked
    assert response.json()["status"] == "ok"  # correct answer


def test_preprocess():
    response = client.post("/preprocess", json={"text": "Hello, WORLD! 123 <b>Hi</b>"})  # call API
    assert response.status_code == 200  # request worked
    assert response.json()["cleaned"] == "hello world hi"  # text is cleaned


def test_empty_text_is_rejected():
    response = client.post("/preprocess", json={"text": ""})  # empty input
    assert response.status_code == 422  # validation error


def test_wrong_method_is_rejected():
    response = client.post("/vectorize", json={"text": "hello", "method": "wrong"})  # bad method
    assert response.status_code == 422  # validation error


def test_llm_without_key_gives_503(monkeypatch):
    monkeypatch.setattr(classification, "GROQ_API_KEY", "")  # pretend the key is missing
    response = client.post("/classify/llm", json={"text": "The team won the match"})  # call API
    assert response.status_code == 503  # clear "not configured" error


@needs_tokenizers
def test_tokenize_bpe_and_wordpiece():
    for method in ["bpe", "wordpiece"]:  # test both tokenizers
        response = client.post("/tokenize", json={"text": "football match", "method": method})  # call API
        assert response.status_code == 200  # request worked
        assert response.json()["token_count"] > 0  # tokens were created


@needs_models
def test_vectorize_tfidf():
    response = client.post("/vectorize", json={"text": "government election vote", "method": "tfidf"})  # call API
    assert response.status_code == 200  # request worked
    assert response.json()["dimension"] > 0  # vector has numbers


@needs_models
def test_similarity_same_text_is_one():
    body = {"text_a": "football match result", "text_b": "football match result", "method": "tfidf"}  # same texts
    assert client.post("/similarity", json=body).json()["cosine_similarity"] > 0.99  # almost 1.0


@needs_models
def test_similarity_search_returns_five():
    response = client.post("/similarity/search", json={"query": "football match result", "method": "tfidf"})  # call API
    assert len(response.json()["results"]) == 5  # top 5 results


@needs_models
def test_classify_ml():
    response = client.post("/classify/ml", json={"text": "The football team won the championship match"})  # call API
    assert response.status_code == 200  # request worked
    assert response.json()["label"] == "sport"  # BBC class name for sports news
