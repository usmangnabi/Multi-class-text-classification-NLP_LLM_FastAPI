# Central settings file: all paths and settings live here (industry standard)
import os  # os lets us read environment variables
from pathlib import Path  # Path helps us build file paths safely

from dotenv import load_dotenv  # reads the .env file

load_dotenv()  # load secret values from .env into environment variables

BASE_DIR = Path(__file__).resolve().parent.parent  # project root folder
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # folder with preprocessed data
MODELS_DIR = BASE_DIR / "models"  # folder with models trained on Colab

DATA_PATH = PROCESSED_DIR / "bbc_clean.csv"  # all documents (text, clean_text, label)
TRAIN_PATH = PROCESSED_DIR / "train.csv"  # training part (used for few-shot examples)

BPE_PATH = MODELS_DIR / "bpe_tokenizer.json"  # saved BPE tokenizer
WORDPIECE_PATH = MODELS_DIR / "wordpiece_tokenizer.json"  # saved WordPiece tokenizer
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"  # saved TF-IDF vectorizer
WORD2VEC_VECTORS_PATH = MODELS_DIR / "word2vec_vectors.npy"  # saved Word2Vec word vectors
WORD2VEC_VOCAB_PATH = MODELS_DIR / "word2vec_vocab.json"  # saved Word2Vec word list
BERT_VECTORS_PATH = MODELS_DIR / "bert_doc_vectors.npy"  # saved BERT vectors of all documents
CLASSIFIER_PATH = MODELS_DIR / "tfidf_logreg.pkl"  # saved TF-IDF + Logistic Regression model

BERT_MODEL_NAME = "bert-base-uncased"  # pretrained BERT model name
BERT_MAX_LENGTH = 256  # max tokens given to BERT (same value used in Colab)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # secret key read from environment
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # LLM used for zero/few-shot
