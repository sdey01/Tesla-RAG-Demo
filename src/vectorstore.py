from __future__ import annotations

import streamlit as st
from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, INDEX_MARKER, PDF_PATH
from src.embeddings import get_embeddings
from src.ingestion import load_and_split_pdf


class VectorStoreError(RuntimeError):
    """Raised when the local document index cannot be loaded or created."""


def get_vectorstore() -> Chroma:
    try:
        embeddings = get_embeddings()
        if INDEX_MARKER.exists():
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_PERSIST_DIRECTORY),
            )

        chunks = load_and_split_pdf(PDF_PATH)
        if not chunks:
            raise VectorStoreError("The Tesla PDF did not contain readable text.")
        CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_PERSIST_DIRECTORY),
        )
        INDEX_MARKER.write_text("Indexed from: " + str(PDF_PATH), encoding="utf-8")
        return vectorstore
    except FileNotFoundError:
        raise
    except Exception as error:
        raise VectorStoreError(str(error)) from error
