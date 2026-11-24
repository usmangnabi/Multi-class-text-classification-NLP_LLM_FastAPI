# DAY 3: ML classifier (TF-IDF + Logistic Regression) and LLM classifier (Groq zero/few-shot)
from functools import lru_cache  # remember results so the model loads once

import joblib  # loads .pkl files

from app.config import CLASSIFIER_PATH, GROQ_API_KEY, GROQ_MODEL  # settings
from app.services.artifacts import require_file  # file check
from app.services.data_loader import get_labels, load_train  # dataset helpers
from app.services.errors import LLMNotConfiguredError  # our error type
from app.services.preprocessing import clean_for_model  # cleaning function

MAX_TEXT_CHARS = 1500  # long articles are cut to this size before sending to the LLM
EXAMPLE_CHARS = 300  # few-shot examples are cut to this size


@lru_cache  # load only once
def load_ml_model():
    return joblib.load(require_file(CLASSIFIER_PATH))  # model trained on Colab


def classify_ml(text: str) -> dict:
    model = load_ml_model()  # get the model
    cleaned = clean_for_model(text)  # clean the same way as in training
    probabilities = model.predict_proba([cleaned])[0]  # one probability per class
    best = probabilities.argmax()  # position of the highest probability
    return {  # result
        "label": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]), 4),
        "all_probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, probabilities)},
    }


def build_prompt(text: str, mode: str) -> str:
    labels = ", ".join(get_labels())  # all allowed labels
    prompt = f"Classify the news article into one of these labels: {labels}.\nAnswer with the label only.\n"  # instruction
    if mode == "few_shot":  # add examples only for few-shot
        train = load_train()  # training table
        for label in get_labels():  # one example per label
            example = train[train["label"] == label].iloc[0]["text"][:EXAMPLE_CHARS]  # first article of this label
            prompt += f"\nArticle: {example}\nLabel: {label}\n"  # show the example
    prompt += f"\nArticle: {text[:MAX_TEXT_CHARS]}\nLabel:"  # the real question
    return prompt  # final prompt


def classify_llm(text: str, mode: str) -> dict:
    if not GROQ_API_KEY:  # key is missing
        raise LLMNotConfiguredError("GROQ_API_KEY is not set. Add it to your .env file.")  # tell the user
    from groq import Groq  # imported here so the app starts without it
    client = Groq(api_key=GROQ_API_KEY)  # create the Groq client
    extra = {"reasoning_effort": "low"} if "gpt-oss" in GROQ_MODEL else {}  # gpt-oss models "think" first, keep it short
    response = client.chat.completions.create(  # call the LLM
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": build_prompt(text, mode)}],
        temperature=0,  # 0 = most stable answers
        max_tokens=300,  # the answer is just one word
    )
    answer = response.choices[0].message.content.strip().lower()  # read the answer text
    label = next((l for l in get_labels() if l in answer), "unknown")  # match a known label
    return {"label": label, "raw_answer": answer, "model": GROQ_MODEL}  # result
