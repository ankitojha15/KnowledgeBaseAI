# Simple nice screen using Streamlit + LangChain.
# Run with: ./aienv/bin/python -m streamlit run app/ui_app.py

import streamlit as st
import os
from app import config
from app.indexing import build_all
from app.generation import answer_question

# Page look
st.set_page_config(page_title="KnowledgeBaseAI", page_icon="📚", layout="centered")
st.title("📚 KnowledgeBaseAI")
st.caption("Ask questions to your PDFs. Answers come with page numbers.")

# Steps at the start
c1, c2, c3 = st.columns(3)
c1.info("**1. Upload**\nPDFs below")
c2.info("**2. Build**\nClick Build Index")
c3.info("**3. Ask**\nType + Ask")
st.divider()

# 1. Upload
st.subheader("1. Upload PDFs")
files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
if files:
    os.makedirs(config.PDF_FOLDER, exist_ok=True)
    for f in files:
        path = os.path.join(config.PDF_FOLDER, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(files)} PDF(s).")

# 2. Build
st.subheader("2. Build Index")
if st.button("🔨 Build Index", use_container_width=True):
    with st.spinner("Reading PDFs..."):
        build_all()
    st.success("Index done! Now ask below.")

st.divider()

# 3. Ask
st.subheader("3. Ask a question")
q = st.text_input("Your question:", placeholder="Example: What is this PDF about?")
if st.button("✨ Ask", use_container_width=True) and q:
    with st.spinner("Thinking..."):
        out = answer_question(q)
    st.subheader("Answer")
    st.success(out["answer"])
    if out.get("citations"):
        with st.expander("📄 Pages used"):
            for c in out["citations"]:
                st.write(f"- Page {c['page']} from {c['source']}")
    if out.get("no_answer"):
        st.warning("No answer found in PDFs.")
