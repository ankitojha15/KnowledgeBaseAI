# Step 6: Write answer using ONLY LangChain.
# Rewrite question -> search -> build text -> ask LLM -> check pages.
# Like: fix question, find book pages, then tell answer with page numbers.

import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app import config

load_dotenv()


def get_llm():
    # Smart writer (Groq is free).
    return ChatGroq(model=config.LLM_MODEL, temperature=0)


def rewrite_query(query):
    # Make question better for search. If no key, keep original.
    try:
        llm = get_llm()
        msg = f"Rewrite this question for book search. Keep meaning. Return only new question. Question: {query}"
        out = llm.invoke(msg)
        return out.content.strip()
    except Exception:
        return query


def build_context(docs):
    # Join docs into one text with page tags. Use big parent text.
    # Also removes same parent twice.
    seen = []
    parts = []
    for d in docs:
        pid = d.metadata.get("parent_id", d.metadata.get("chunk_id"))
        if pid in seen:
            continue
        seen.append(pid)
        text = d.metadata.get("parent_text", d.page_content)
        page = d.metadata["page"]
        source = d.metadata["source"]
        parts.append(f"[Page {page} from {source}]\n{text}")
    return "\n\n".join(parts)


def verify_citations(answer, docs):
    # Check: every [Page X] in answer must be in our docs.
    pages_in_docs = []
    for d in docs:
        pages_in_docs.append(str(d.metadata["page"]))
    pages_in_answer = re.findall(r"\[Page (\d+)", answer)
    good = []
    bad = []
    for p in pages_in_answer:
        if p in pages_in_docs:
            good.append(p)
        else:
            bad.append(p)
    return {"good": good, "bad": bad, "ok": len(bad) == 0}


def answer_question(query):
    # Full job: rewrite -> search -> answer with pages.
    from app.retrieval import search

    new_q = rewrite_query(query)
    docs = search(new_q)

    if len(docs) == 0:
        return {"answer": "I don't know from the PDFs.", "citations": [], "no_answer": True}

    context = build_context(docs)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer only from context. Give short answer. Add [Page X] after each fact. If not in context, say: I don't know from the PDFs."),
        ("human", "Question: {q}\n\nContext:\n{ctx}"),
    ])
    chain = prompt | llm
    out = chain.invoke({"q": query, "ctx": context})
    answer = out.content.strip()

    # No-answer check
    if "I don't know from the PDFs" in answer:
        return {"answer": answer, "citations": [], "no_answer": True}

    # Collect citations
    check = verify_citations(answer, docs)
    cites = []
    for d in docs:
        cites.append({"page": d.metadata["page"], "source": d.metadata["source"]})

    # Keep only cited pages
    only_good = []
    for c in cites:
        if str(c["page"]) in check["good"]:
            if c not in only_good:
                only_good.append(c)

    return {"answer": answer, "citations": only_good, "no_answer": False, "rewrite": new_q}


# If we run this file, test helper parts (no API call)
if __name__ == "__main__":
    from langchain_core.documents import Document
    fake = [Document(page_content="Cats like milk", metadata={"page": 1, "source": "a.pdf", "chunk_id": "c0", "parent_id": "p0", "parent_text": "Cats like milk"})]
    print(build_context(fake))
    print(verify_citations("Cats drink [Page 1]", fake))
