# DAY 2 - Step 2: Cosine similarity and top-5 search
import numpy as np  # numpy for math on arrays

from app.services.data_loader import load_dataset  # our dataset
from app.services.vectorization import dataset_matrix, vectorize  # vector helpers


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)  # length of a times length of b
    if norm == 0:  # avoid dividing by zero
        return 0.0  # empty vector means no similarity
    return float(np.dot(a, b) / norm)  # formula: (a . b) / (|a| * |b|)


def top_k_similar(query: str, method: str, k: int = 5) -> list:
    query_vec = vectorize(query, method)  # turn the query into a vector
    matrix = dataset_matrix(method)  # vectors of all dataset texts
    scores = [cosine_similarity(query_vec, row) for row in matrix]  # compare with every text
    best = np.argsort(scores)[::-1][:k]  # indexes of the k highest scores
    df = load_dataset()  # dataset table
    return [  # build the result list
        {"index": int(i), "label": df.loc[i, "label"], "score": round(scores[i], 4), "text": df.loc[i, "text"][:200]}
        for i in best  # one item per best match
    ]
