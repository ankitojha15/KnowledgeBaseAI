# KnowledgeBaseAI - Ask questions to your PDFs

A very simple project using ONLY LangChain. You give PDFs, you ask questions, it gives answers with page numbers.

## How it works (in simple words)

1. **Read PDF** -> `PyMuPDFLoader` gets text page by page
2. **Cut text** -> `RecursiveCharacterTextSplitter` makes parent (1000) + child (300)
3. **Make index** -> `HuggingFaceEmbeddings + FAISS` (meaning) + `BM25Retriever` (words)
4. **Search** -> hybrid + RRF mix + `HuggingFaceCrossEncoder` rerank
5. **Answer** -> `ChatGroq` rewrites question, writes answer with [Page X], checks pages

## Run it (3 steps)

1. Put PDFs in `data/pdfs/`, add keys to `.env` (copy from `.env.example`):
   - `GROQ_API_KEY` for answers
   - `HUGGINGFACE_API_KEY` for embeddings (optional, model is public)
2. Build: `python -m app.indexing`
3. Ask:
   - API: `uvicorn app.main_api:app --reload` -> POST `/ask` with `{"question": "..."}`
   - UI: `streamlit run app/ui_app.py`

## Test it

- Eval: `python -m evals.run_eval` -> Hit@5, MRR, NDCG, no-answer score (60 cases, demo book)
- All tests: `python -m tests.test_all` -> checks config, chunking, search, answer, API

## Scores (demo book, 50 normal + 10 adversarial)

- Hit@5: 1.00, MRR: 1.00, NDCG: 1.00, No-answer: 10/10

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

## Resume line (copy this)

Built KnowledgeBaseAI, a LangChain RAG system that answers PDF questions with page-level citations. Used parent-child chunking, hybrid dense (FAISS) + BM25 retrieval with RRF and cross-encoder reranking, Groq LLM with query rewriting and citation verification. Added 60-case eval (Hit/MRR/NDCG), FastAPI + Streamlit, tested end-to-end.
