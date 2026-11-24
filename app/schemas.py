# Request shapes (Pydantic checks the input automatically)
from typing import Literal  # Literal limits a field to fixed choices

from pydantic import BaseModel, Field  # BaseModel = base for all schemas

Method = Literal["tfidf", "word2vec", "bert"]  # allowed vectorization methods


class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=1)  # text to clean (cannot be empty)
    remove_stopwords: bool = False  # optional stopword removal


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)  # text to tokenize
    method: Literal["bpe", "wordpiece"] = "bpe"  # tokenizer type


class VectorizeRequest(BaseModel):
    text: str = Field(..., min_length=1)  # text to convert
    method: Method = "tfidf"  # vectorization method


class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1)  # first text
    text_b: str = Field(..., min_length=1)  # second text
    method: Method = "tfidf"  # vectorization method


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)  # search text
    method: Method = "tfidf"  # vectorization method
    top_k: int = Field(5, ge=1, le=10)  # how many results (default 5)


class MLClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)  # text to classify


class LLMClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)  # text to classify
    mode: Literal["zero_shot", "few_shot"] = "zero_shot"  # prompting style
