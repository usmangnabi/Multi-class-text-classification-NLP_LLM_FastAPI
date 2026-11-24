# Application entry point: run with `uvicorn app.main:app --reload`
from fastapi import FastAPI, Request  # the web framework
from fastapi.responses import JSONResponse  # to send custom error replies

from app.routers import classify, preprocess, similarity, tokenize, vectorize  # our routers
from app.services.errors import ModelNotFoundError  # our error type

app = FastAPI(  # create the app
    title="NLP and LLM Application",
    description="Preprocessing, tokenization, vectorization, similarity and classification (ML + Groq LLM).",
    version="1.0.0",
)

app.include_router(preprocess.router)  # add /preprocess
app.include_router(tokenize.router)  # add /tokenize
app.include_router(vectorize.router)  # add /vectorize
app.include_router(similarity.router)  # add /similarity and /similarity/search
app.include_router(classify.router)  # add /classify/ml and /classify/llm


@app.exception_handler(ModelNotFoundError)  # runs whenever a model file is missing
def handle_missing_model(request: Request, error: ModelNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(error)})  # clear message, not a crash


@app.get("/health", tags=["Health"])  # simple check that the app is alive
def health():
    return {"status": "ok"}  # response
