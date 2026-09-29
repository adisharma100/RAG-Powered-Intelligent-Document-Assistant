"""Streamlit chat UI.   streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from rag_assistant import RAGAssistant

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")


@st.cache_resource
def get_assistant() -> RAGAssistant:
    return RAGAssistant()


assistant = get_assistant()

with st.sidebar:
    st.header("Knowledge base")
    st.caption(f"{assistant.count()} chunks indexed")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads):
        with tempfile.TemporaryDirectory() as tmp:
            added = 0
            for up in uploads:
                p = Path(tmp) / up.name
                p.write_bytes(up.getbuffer())
                added += assistant.ingest(p)
        st.success(f"Added {added} chunks")
        st.rerun()
    if st.button("Clear index"):
        assistant.reset()
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about your documents..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            ans = assistant.ask(question)
        st.markdown(ans.answer)
        for i, s in enumerate(ans.sources, start=1):
            with st.expander(f"[{i}] {s.source}" + (f" - page {s.page}" if s.page else "") + f"  (relevance {s.score})"):
                st.write(s.snippet)
    st.session_state.history.append({"role": "assistant", "content": ans.answer})
