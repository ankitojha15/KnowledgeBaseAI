# End-to-end test: checks all steps work together.
# Run with: python -m tests.test_all
# Very simple: each test is 3-5 lines.

from langchain_core.documents import Document


def test_config():
    from app import config
    assert config.PDF_FOLDER == "data/pdfs"
    assert config.TOP_K == 5
    print("config ok")


def test_chunking():
    from app.chunking import make_chunks
    d = [Document(page_content="Hello world. " * 50, metadata={"page": 1, "source": "demo.pdf"})]
    p, c = make_chunks(d)
    assert len(p) >= 1 and len(c) >= 1
    assert "parent_id" in c[0].metadata
    print(f"chunking ok: {len(p)} parents, {len(c)} childs")


def test_search_mix():
    from app.retrieval import rrf_mix, rerank
    a = [Document(page_content="Cats like milk", metadata={"chunk_id": "c0", "page": 1, "source": "a"})]
    b = [Document(page_content="Dogs like bones", metadata={"chunk_id": "c1", "page": 1, "source": "a"})]
    m = rrf_mix(a, b)
    assert len(m) == 2
    r = rerank("cat", a + b, top_k=1)
    assert "Cats" in r[0].page_content
    print("search mix + rerank ok")


def test_answer_helpers():
    from app.generation import build_context, verify_citations
    fake = [Document(page_content="Cats like milk", metadata={"page": 1, "source": "a.pdf", "chunk_id": "c0", "parent_id": "p0", "parent_text": "Cats like milk"})]
    ctx = build_context(fake)
    assert "[Page 1" in ctx
    v = verify_citations("Cats [Page 1]", fake)
    assert v["ok"] is True
    print("answer helpers ok")


def test_golden():
    import json
    with open("evals/golden.json") as f:
        cases = json.load(f)
    assert len(cases) >= 50
    print(f"golden ok: {len(cases)} cases")


def test_api():
    from fastapi.testclient import TestClient
    from app.main_api import app
    c = TestClient(app)
    assert c.get("/health").json() == {"ok": True}
    print("api ok")


if __name__ == "__main__":
    test_config()
    test_chunking()
    test_search_mix()
    test_answer_helpers()
    test_golden()
    test_api()
    print("ALL TESTS PASSED")
