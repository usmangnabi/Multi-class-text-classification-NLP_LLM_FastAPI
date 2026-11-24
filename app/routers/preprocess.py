# Endpoint: POST /preprocess
from fastapi import APIRouter  # APIRouter groups related endpoints

from app.schemas import PreprocessRequest  # input shape
from app.services.preprocessing import clean_text  # cleaning function

router = APIRouter(tags=["Preprocessing"])  # create the router


@router.post("/preprocess")  # URL and HTTP method
def preprocess(request: PreprocessRequest):
    cleaned = clean_text(request.text, request.remove_stopwords)  # clean the text
    return {"original": request.text, "cleaned": cleaned, "tokens": cleaned.split()}  # result
