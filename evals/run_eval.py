# Step 7: Test our search using ONLY LangChain.
# Hit Rate = did we find right page?
# MRR = 1 / rank of right page (best = 1).
# NDCG = 1 / log2(rank+1) (best = 1).
# Like giving marks to our search.

import json
import math
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from app.indexing import get_embedding_model
from app.retrieval import hybrid_search, rerank


def load_demo_docs():
    # Make Documents from tiny demo book
    with open("evals/demo_docs.json") as f:
        rows = json.load(f)
    docs = []
    for i, r in enumerate(rows):
        docs.append(Document(
            page_content=r["text"],
            metadata={"page": r["page"], "source": r["source"],
                      "chunk_id": f"child-{i}", "parent_id": f"parent-{i}",
                      "parent_text": r["text"]},
        ))
    return docs


def find_rank(hits, want_page):
    # Where is the right page? 1,2,3... or 0 if not found.
    for i, d in enumerate(hits):
        if d.metadata["page"] == want_page:
            return i + 1
    return 0


def run_eval():
    docs = load_demo_docs()
    with open("evals/golden.json") as f:
        cases = json.load(f)

    # Build both search tools in memory (no files)
    model = get_embedding_model()
    dense_shop = FAISS.from_documents(docs, model)
    bm25_tool = BM25Retriever.from_documents(docs)
    bm25_tool.k = 10

    hits = 0
    mrr_sum = 0
    ndcg_sum = 0
    normal_n = 0
    adv_ok = 0
    adv_n = 0

    for c in cases:
        q = c["question"]
        if c["type"] == "adversarial":
            # Bad question should have low score (no good match).
            # We check: top score is low OR no page match.
            adv_n = adv_n + 1
            if q.strip() == "":
                adv_ok = adv_ok + 1
                continue
            d_hits = dense_shop.similarity_search(q, k=5)
            # If best hit has nothing to do with question, count as correct no-answer.
            # Simple rule: if question words not in best hit, we would say "I don't know".
            best_text = d_hits[0].page_content.lower() if len(d_hits) > 0 else ""
            qw = [w for w in q.lower().split() if len(w) > 3]
            found = any(w in best_text for w in qw)
            if not found:
                adv_ok = adv_ok + 1
            continue

        normal_n = normal_n + 1
        mixed = hybrid_search(q, dense_shop, bm25_tool)
        final = rerank(q, mixed, top_k=5)

        rank = find_rank(final, c["page"])
        if rank > 0:
            hits = hits + 1
            mrr_sum = mrr_sum + 1 / rank
            ndcg_sum = ndcg_sum + 1 / math.log2(rank + 1)

    print(f"Normal: {normal_n}, Adversarial: {adv_n}")
    print(f"Hit@5: {hits / normal_n:.2f}")
    print(f"MRR: {mrr_sum / normal_n:.2f}")
    print(f"NDCG: {ndcg_sum / normal_n:.2f}")
    print(f"No-answer correct: {adv_ok}/{adv_n} = {adv_ok / adv_n:.2f}")


if __name__ == "__main__":
    run_eval()
