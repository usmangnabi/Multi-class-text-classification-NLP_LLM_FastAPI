# Endpoints: POST /similarity and POST /similarity/search
from fastapi import APIRouter  # router class

from app.schemas import SearchRequest, SimilarityRequest  # input shapes
from app.services.similarity import cosine_similarity, top_k_similar  # similarity functions
from app.services.vectorization import vectorize  # vector function

router = APIRouter(tags=["Similarity"])  # create the router


@router.post("/similarity")  # compare two texts
def similarity(request: SimilarityRequest):
    vec_a = vectorize(request.text_a, request.method)  # vector of text A
    vec_b = vectorize(request.text_b, request.method)  # vector of text B
    score = cosine_similarity(vec_a, vec_b)  # compare them
    return {"method": request.method, "cosine_similarity": round(score, 4)}  # result


@router.post("/similarity/search")  # find the most similar dataset texts
def similarity_search(request: SearchRequest):
    results = top_k_similar(request.query, request.method, request.top_k)  # search
    return {"query": request.query, "method": request.method, "results": results}  # result
