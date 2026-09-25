from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_and_split_pdf(pdf_path: Path) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    documents = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["source_document"] = pdf_path.name
        chunk.metadata["chunk_id"] = chunk_id
    return chunks
