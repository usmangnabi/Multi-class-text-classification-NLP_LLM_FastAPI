# DAY 2 - Step 1: Turn text into numbers (TF-IDF, Word2Vec, BERT)
# The models are TRAINED on Google Colab and only LOADED here.
from functools import lru_cache  # remember results so models load only once

import json  # reads the word list

import joblib  # loads .pkl files
import numpy as np  # numpy handles number arrays

from app.config import BERT_MAX_LENGTH, BERT_MODEL_NAME, BERT_VECTORS_PATH, TFIDF_PATH, WORD2VEC_VECTORS_PATH, WORD2VEC_VOCAB_PATH  # settings
from app.services.artifacts import require_file  # file check
from app.services.data_loader import load_dataset  # our dataset
from app.services.preprocessing import clean_for_model  # cleaning function


@lru_cache  # load only once
def get_tfidf():
    return joblib.load(require_file(TFIDF_PATH))  # TF-IDF vectorizer trained on Colab


@lru_cache  # load only once
def get_word2vec() -> dict:
    with open(require_file(WORD2VEC_VOCAB_PATH)) as file:  # list of words
        words = json.load(file)  # read the word list
    vectors = np.load(require_file(WORD2VEC_VECTORS_PATH))  # one row per word (trained on Colab)
    return {word: vectors[i] for i, word in enumerate(words)}  # dictionary: word -> vector


@lru_cache  # load only once
def get_bert():
    from transformers import AutoModel, AutoTokenizer  # imported here because it is heavy
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)  # BERT's own tokenizer
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)  # pretrained BERT model
    model.eval()  # we only use it, not train it
    return tokenizer, model  # return both


def tfidf_vector(text: str) -> np.ndarray:
    cleaned = clean_for_model(text)  # clean the same way as in training
    return get_tfidf().transform([cleaned]).toarray()[0]  # sparse -> normal array, first row


def word2vec_vector(text: str) -> np.ndarray:
    word_vectors = get_word2vec()  # dictionary word -> vector
    words = clean_for_model(text).split()  # cleaned words
    vectors = [word_vectors[w] for w in words if w in word_vectors]  # vectors of known words
    if not vectors:  # no known words
        return np.zeros(len(next(iter(word_vectors.values()))))  # return a zero vector
    return np.mean(vectors, axis=0)  # sentence vector = average of word vectors


def bert_vector(text: str) -> np.ndarray:
    import torch  # imported here because it is heavy
    tokenizer, model = get_bert()  # load BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=BERT_MAX_LENGTH)  # text -> ids
    with torch.no_grad():  # no training, saves memory
        outputs = model(**inputs)  # run BERT
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()  # average all token vectors


def vectorize(text: str, method: str) -> np.ndarray:
    if method == "tfidf":  # choose the method
        return tfidf_vector(text)  # TF-IDF vector
    if method == "word2vec":
        return word2vec_vector(text)  # Word2Vec vector
    if method == "bert":
        return bert_vector(text)  # BERT vector
    raise ValueError(f"Unknown method: {method}")  # wrong method name


@lru_cache  # compute dataset vectors only once per method
def dataset_matrix(method: str) -> np.ndarray:
    df = load_dataset()  # all documents
    if method == "bert":  # BERT vectors were already computed on Colab (slow on a normal PC)
        return np.load(require_file(BERT_VECTORS_PATH))  # one row per document
    if method == "tfidf":  # TF-IDF: transform all cleaned documents at once
        return get_tfidf().transform(df["clean_text"]).toarray()  # one row per document
    return np.array([word2vec_vector(t) for t in df["text"]])  # Word2Vec: one row per document
