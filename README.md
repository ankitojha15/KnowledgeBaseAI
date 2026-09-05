# KnowledgeBaseAI - Ask questions to your PDFs

A very simple project. You give PDFs, you ask questions, it gives answers with page numbers.

## How it works (in simple words)

1. **Read PDF** -> open PDF and get text page by page
2. **Cut text** -> cut into small pieces (child) and big pieces (parent)
3. **Make index** -> dense (meaning search) + BM25 (word search)
4. **Search** -> hybrid search + RRF + rerank to find best pieces
5. **Answer** -> LLM writes answer with page citations

## Tech we use (all simple + free)

- PDF reading: `PyMuPDF`
- Meaning search: `sentence-transformers (all-MiniLM-L6-v2)` + `FAISS`
- Word search: `rank-bm25`
- Rerank: `cross-encoder (ms-marco-MiniLM-L-6-v2)`
- Smart answer: `Groq (llama-3.3-70b-versatile)` - free API
- Backend: `FastAPI`
- UI: `Streamlit`

## Folders

- `app/config.py` - all settings
- `app/ingestion.py` - read PDFs (Step 2)
- `app/chunking.py` - cut text (Step 3)
- `app/indexing.py` - make search index (Step 4)
- `app/retrieval.py` - find answers (Step 5)
- `app/generation.py` - write answer (Step 6)
- `data/pdfs/` - put PDFs here
- `data/processed/` - clean text here
- `data/index/` - search files here
- `evals/` - test questions here
