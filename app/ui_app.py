# Step 9: Simple screen using Streamlit + LangChain.
# Run with: streamlit run app/ui_app.py

import streamlit as st
from app.indexing import build_all
from app.generation import answer_question

st.title("KnowledgeBaseAI - Ask your PDFs")

st.write("1. Put PDFs in data/pdfs. 2. Click Build. 3. Ask.")

if st.button("Build Index"):
    with st.spinner("Reading PDFs..."):
        build_all()
    st.success("Index done!")

q = st.text_input("Your question:")

if st.button("Ask") and q:
    with st.spinner("Thinking..."):
        out = answer_question(q)
    st.write("**Answer:**")
    st.write(out["answer"])
    if out.get("citations"):
        st.write("**Pages:**")
        for c in out["citations"]:
            st.write(f"- Page {c['page']} from {c['source']}")
    if out.get("no_answer"):
        st.warning("No answer found in PDFs.")
