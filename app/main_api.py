# Step 8: API using FastAPI + LangChain.
# Like a waiter: takes question, gives answer.
# Run with: uvicorn app.main_api:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from app.indexing import build_all
from app.generation import answer_question

app = FastAPI(title="KnowledgeBaseAI")


class AskIn(BaseModel):
    question: str  # what user asks


class AskOut(BaseModel):
    answer: str
    citations: list = []
    no_answer: bool = False


@app.get("/")
def home():
    return {"msg": "KnowledgeBaseAI is running. Use /ask to ask."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/index")
def make_index():
    # Read PDFs and make search files.
    build_all()
    return {"done": True}


@app.post("/ask", response_model=AskOut)
def ask(data: AskIn):
    # Find answer with pages.
    out = answer_question(data.question)
    return {
        "answer": out["answer"],
        "citations": out.get("citations", []),
        "no_answer": out.get("no_answer", False),
    }
