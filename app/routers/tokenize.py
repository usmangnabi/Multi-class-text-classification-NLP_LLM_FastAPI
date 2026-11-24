# Endpoint: POST /tokenize
from functools import lru_cache  # load each tokenizer only once

from fastapi import APIRouter  # router class

from app.config import BPE_PATH, WORDPIECE_PATH  # saved tokenizer files
from app.schemas import TokenizeRequest  # input shape
from app.services.artifacts import require_file  # file check
from app.services.preprocessing import clean_text  # cleaning function
from app.services.tokenization import BPETokenizer, WordPieceTokenizer  # tokenizers

router = APIRouter(tags=["Tokenization"])  # create the router


@lru_cache  # remember the loaded tokenizer
def get_tokenizer(method: str):
    if method == "bpe":  # choose type
        return BPETokenizer.load(require_file(BPE_PATH))  # load BPE trained on Colab
    return WordPieceTokenizer.load(require_file(WORDPIECE_PATH))  # load WordPiece trained on Colab


@router.post("/tokenize")  # URL and HTTP method
def tokenize(request: TokenizeRequest):
    tokenizer = get_tokenizer(request.method)  # get the tokenizer
    tokens = tokenizer.tokenize(clean_text(request.text))  # clean first, then tokenize
    unknown = sum(1 for t in tokens if t in ("<unk>", "[UNK]"))  # count unknown tokens
    return {  # result
        "method": request.method,
        "tokens": tokens,
        "token_count": len(tokens),
        "unknown_count": unknown,
        "vocab_size": len(tokenizer.vocab),
    }
