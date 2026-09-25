from __future__ import annotations

import streamlit as st

from src.config import APP_TITLE
from src.rag_pipeline import RAGError, answer_question
from src.vectorstore import VectorStoreError, get_vectorstore


st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")


def render_history() -> None:
    st.sidebar.title("Question History")
    history = st.session_state.get("question_history", [])
    if not history:
        st.sidebar.caption("Questions asked in this session will appear here.")
        return
    for index, question in enumerate(history, start=1):
        st.sidebar.write(f"{index}. {question}")


def main() -> None:
    st.title(APP_TITLE)
    st.write("Ask grounded questions about the Tesla financial report.")
    render_history()

    try:
        vectorstore = get_vectorstore()
    except FileNotFoundError:
        st.error("Tesla financial report PDF was not found. Please place the PDF in the configured data directory.")
        st.stop()
    except VectorStoreError as error:
        st.error(f"Unable to prepare the document index: {error}")
        st.stop()

    question = st.text_input("Ask a question about the Tesla financial report")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Searching the report and preparing an answer..."):
                answer, sources, expansions = answer_question(question, vectorstore)
        except RAGError as error:
            st.error(str(error))
            return

        st.session_state.setdefault("question_history", []).append(question.strip())
        st.subheader("Query Expansion & Retrieval Trace")
        st.markdown(f"**Original question:** {question.strip()}")
        with st.expander("View the 3 generated query expansions", expanded=True):
            for index, expansion in enumerate(expansions, start=1):
                st.write(f"{index}. {expansion}")
        st.caption("Chroma searched the original question and each expansion, then ranked and deduplicated the best 5 chunks.")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources / Retrieved Context")
        if not sources:
            st.info("No relevant report context was found.")
            return
        for index, source in enumerate(sources, start=1):
            page_label = f"Page {source['page']}" if source["page"] is not None else "Page unavailable"
            with st.expander(f"Source {index} — {page_label}"):
                st.caption(source["document"])
                st.write(source["text"])


if __name__ == "__main__":
    main()
