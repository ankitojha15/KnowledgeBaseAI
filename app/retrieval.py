# Step 5: Find best pieces using ONLY LangChain.
# Hybrid = dense (meaning) + BM25 (words). RRF = mix both lists.
# Rerank = ask a smart checker to pick the best.
# Like asking 2 friends, mixing answers, then picking best.

import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app import config
from app.indexing import get_embedding_model

load_dotenv()


def load_dense():
    # Load saved FAISS index
    model = get_embedding_model()
    path = os.path.join(config.INDEX_FOLDER, "faiss")
    shop = FAISS.load_local(path, model, allow_dangerous_deserialization=True)
    return shop


def load_bm25():
    # Load saved BM25 index
    path = os.path.join(config.INDEX_FOLDER, "bm25.pkl")
    with open(path, "rb") as f:
        tool = pickle.load(f)
    return tool


def rrf_mix(dense_docs, bm25_docs):
    # Mix 2 lists with RRF. Simple formula: score = 1 / (60 + rank)
    # Higher score = better.
    scores = {}  # chunk_id -> score
    box = {}  # chunk_id -> doc

    for rank, d in enumerate(dense_docs):
        cid = d.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)
        box[cid] = d

    for rank, d in enumerate(bm25_docs):
        cid = d.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)
        box[cid] = d

    # Sort by score, best first
    best_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    mixed = []
    for cid in best_ids:
        mixed.append(box[cid])
    return mixed


_checker = None  # keep model in memory so we load only once


def rerank(query, docs, top_k=5):
    # Ask cross-encoder: how well does each doc match query?
    global _checker
    if _checker is None:
        _checker = HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL)
    pairs = []
    for d in docs:
        pairs.append((query, d.page_content))
    marks = _checker.score(pairs)

    # Join docs with marks and sort
    together = []
    for i in range(len(docs)):
        together.append((docs[i], marks[i]))
    together.sort(key=lambda x: x[1], reverse=True)

    best = []
    for doc, mark in together[:top_k]:
        best.append(doc)
    return best


def search(query):
    # Full search: hybrid + RRF + rerank. Returns best docs.
    dense_shop = load_dense()
    bm25_tool = load_bm25()

    n = config.TOP_K * 2
    dense_hits = dense_shop.similarity_search(query, k=n)
    bm25_hits = bm25_tool.invoke(query)

    mixed = rrf_mix(dense_hits, bm25_hits)
    final = rerank(query, mixed, top_k=config.TOP_K)
    return final


# If we run this file, ask a test question
if __name__ == "__main__":
    answers = search("test question")
    for a in answers:
        print(a.page_content[:200], "| page:", a.metadata["page"])
