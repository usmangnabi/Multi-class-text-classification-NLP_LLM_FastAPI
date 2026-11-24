<div align="center">

# 📰 NLP & LLM News Classification API

**Preprocessing · Tokenization · Embeddings · Similarity Search · ML and LLM Classification, served with FastAPI**

![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

📄 **Full project report:** [docs/Text_Preprocessing_and_Classification.md](docs/Text_Preprocessing_and_Classification.md)

</div>

---

## Overview

A proof-of-concept NLP application built on the Kaggle **BBC News** dataset (5 topics: business, entertainment, politics, sport, tech). Heavy training runs once on **Google Colab**; a lightweight **FastAPI** service loads the saved models and serves predictions.

| Area | Implemented |
|---|---|
| Preprocessing | Lowercasing, HTML / link / number removal, stopwords |
| Tokenization | BPE and WordPiece, written from scratch |
| Vectorization | TF-IDF, Word2Vec, BERT |
| Similarity | Cosine similarity and top-k search |
| Classification | TF-IDF + Logistic Regression, Groq LLM (zero-shot / few-shot) |
| API | 8 FastAPI endpoints with Swagger docs |

**Headline result:** TF-IDF + Logistic Regression reaches **97.4% accuracy** on the 426-article test set. See the [report](docs/Text_Preprocessing_and_Classification.md) for all results and explanations.

## Endpoints

| Method | URL | Purpose |
|---|---|---|
| `GET` | `/health` | Service status |
| `POST` | `/preprocess` | Clean text |
| `POST` | `/tokenize` | BPE or WordPiece tokens |
| `POST` | `/vectorize` | TF-IDF, Word2Vec or BERT vector |
| `POST` | `/similarity` | Cosine similarity of two texts |
| `POST` | `/similarity/search` | Top-k similar articles |
| `POST` | `/classify/ml` | Logistic Regression prediction |
| `POST` | `/classify/llm` | Groq zero-shot / few-shot prediction |

## Quick Start

**1. Install**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure** (get a free key at [console.groq.com/keys](https://console.groq.com/keys))

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

```env
GROQ_API_KEY=your-groq-key
GROQ_MODEL=openai/gpt-oss-20b
```

**3. Prepare data and models** (details in [GUIDE.md](GUIDE.md))

| Notebook | Where | Output |
|---|---|---|
| `01_data_preprocessing.ipynb` | Local | `data/processed/*.csv` |
| `02_tokenization_and_vectorization_colab.ipynb` | Colab | tokenizers, TF-IDF, Word2Vec, BERT vectors |
| `03_classification_and_evaluation_colab.ipynb` | Colab | `tfidf_logreg.pkl`, comparison table |

Copy the Colab downloads into `models/`.

**4. Run**

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>.

## Example

```bash
curl -X POST http://127.0.0.1:8000/classify/llm \
  -H "Content-Type: application/json" \
  -d '{"text": "The football team won the championship match", "mode": "few_shot"}'
```

```json
{ "mode": "few_shot", "label": "sport", "raw_answer": "sport", "model": "openai/gpt-oss-20b" }
```

## Project Structure

```text
.
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # paths and settings
│   ├── schemas.py         # request validation
│   ├── routers/           # HTTP endpoints
│   └── services/          # NLP logic
├── notebooks/             # preprocessing (local), training (Colab)
├── data/                  # raw (not committed) and processed CSV files
├── models/                # files trained on Colab
├── tests/                 # pytest suites
├── docs/                  # project report and images
├── GUIDE.md               # step-by-step instructions
└── requirements.txt
```

## Testing

```bash
pytest
```

Tests that need trained model files are skipped automatically until those files exist.

## Author

**Your Name** · [GitHub](https://github.com/<your-username>) · [LinkedIn](https://www.linkedin.com/in/<your-profile>)
