---
title: KnowledgeBaseAI
emoji: 📚
sdk: streamlit
sdk_version: 1.63.0
app_file: app/ui_app.py
pinned: false
---

# KnowledgeBaseAI - Ask questions to your PDFs

Live demo: https://knowledgebaseai-pzi7.streamlit.app/

A very simple project using ONLY LangChain. You give PDFs, you ask questions, it gives answers with page numbers.

## How it works (in simple words)

1. **Read PDF** -> `PyMuPDFLoader` gets text page by page
2. **Cut text** -> `RecursiveCharacterTextSplitter` makes parent (1000) + child (300)
3. **Make index** -> `HuggingFaceEmbeddings + FAISS` (meaning) + `BM25Retriever` (words)
4. **Search** -> hybrid + RRF mix + `HuggingFaceCrossEncoder` rerank
5. **Answer** -> `ChatGroq` rewrites question, writes answer with [Page X], checks pages

## Run it (3 steps)

1. Put PDFs in `data/pdfs/` or use Upload button in UI. Add keys to `.env` (copy from `.env.example`):
   - `GROQ_API_KEY` for answers (model: `openai/gpt-oss-120b`)
   - `HUGGINGFACE_API_KEY` for embeddings (optional, model is public)
2. Build: `./aienv/bin/python -m app.indexing`
3. Ask:
   - UI local: `./aienv/bin/python -m streamlit run app/ui_app.py`
   - UI live: https://knowledgebaseai-pzi7.streamlit.app/
   - API: `./aienv/bin/python -m uvicorn app.main_api:app --reload` -> POST `/ask` with `{"question": "..."}`


## Files (all very simple)

- `app/config.py` - all settings
- `app/ingestion.py` - read PDFs
- `app/chunking.py` - parent/child cut
- `app/indexing.py` - dense + BM25
- `app/retrieval.py` - hybrid + RRF + rerank
- `app/generation.py` - rewrite + answer + citations
- `app/main_api.py` - FastAPI
- `app/ui_app.py` - Streamlit
- `evals/golden.json` - 60 test questions
- `tests/test_all.py` - end-to-end test


