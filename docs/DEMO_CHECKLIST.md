# Demo Checklist (5-10 minutes)

Before the meeting:
- [ ] `.env` has a valid `GROQ_API_KEY`
- [ ] `models/` contains all files from the Colab notebooks
- [ ] `pytest` passes
- [ ] Run once: `uvicorn app.main:app --reload`, and call a `bert` endpoint once so BERT is already downloaded

Demo order (matches the Task):
1. Notebook 01 - show cleaning and the saved CSV files
2. `/docs` -> `GET /health`
3. `POST /preprocess` - paste a messy sentence
4. `POST /tokenize` - BPE vs WordPiece with a rare word
5. `POST /vectorize` - TF-IDF, Word2Vec, BERT (show dimensions)
6. `POST /similarity` - two similar texts and two different texts
7. `POST /similarity/search` - top 5 results for a news sentence
8. `POST /classify/ml` and `POST /classify/llm` (zero_shot, then few_shot)
9.Show `docs/REPORT.md` tables and the GitHub repository
