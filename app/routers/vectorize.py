# Endpoint: POST /vectorize
from fastapi import APIRouter  # router class

from app.schemas import VectorizeRequest  # input shape
from app.services.vectorization import vectorize as make_vector  # vector function

router = APIRouter(tags=["Vectorization"])  # create the router


@router.post("/vectorize")  # URL and HTTP method
def vectorize(request: VectorizeRequest):
    vector = make_vector(request.text, request.method)  # service cleans the text itself
    return {  # result
        "method": request.method,
        "dimension": len(vector),  # length of the vector
        "vector_preview": [round(float(v), 4) for v in vector[:10]],  # first 10 numbers only
    }
