# Endpoints: POST /classify/ml and POST /classify/llm
from fastapi import APIRouter, HTTPException  # HTTPException sends error responses

from app.schemas import LLMClassifyRequest, MLClassifyRequest  # input shapes
from app.services.classification import classify_llm, classify_ml  # classifiers
from app.services.errors import LLMNotConfiguredError, ModelNotFoundError  # our error types

router = APIRouter(tags=["Classification"])  # create the router


@router.post("/classify/ml")  # TF-IDF + Logistic Regression
def classify_with_ml(request: MLClassifyRequest):
    return classify_ml(request.text)  # predict (service cleans the text itself)


@router.post("/classify/llm")  # zero-shot or few-shot with Groq
def classify_with_llm(request: LLMClassifyRequest):
    try:
        result = classify_llm(request.text, request.mode)  # ask the LLM
    except ModelNotFoundError:  # data file missing
        raise  # main.py turns this into a 503 message
    except LLMNotConfiguredError as error:  # missing API key
        raise HTTPException(status_code=503, detail=str(error))  # 503 = service unavailable
    except Exception as error:  # any other LLM problem (network, rate limit...)
        raise HTTPException(status_code=502, detail=f"LLM error: {error}")  # 502 = bad gateway
    return {"mode": request.mode, **result}  # result
