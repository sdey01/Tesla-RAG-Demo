from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma

from src.config import MISSING_EVIDENCE_MESSAGE
from src.llm import get_llm
from src.retrieval import retrieve_documents


class RAGError(RuntimeError):
    """Raised for user-facing RAG failures."""


def _source_details(documents: list[Any]) -> list[dict[str, Any]]:
    sources = []
    for document in documents:
        metadata = document.metadata
        page = metadata.get("page")
        sources.append(
            {
                "document": metadata.get("source_document", metadata.get("source", "Tesla financial report")),
                "page": page + 1 if isinstance(page, int) else None,
                "text": document.page_content,
            }
        )
    return sources


def answer_question(
    question: str,
    vectorstore: Chroma,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    try:
        retrieval = retrieve_documents(vectorstore, question)
        documents = retrieval.documents
        if not documents:
            return MISSING_EVIDENCE_MESSAGE, [], retrieval.expansions

        context = "\n\n".join(
            f"[Source {index} | {doc.metadata.get('source_document', 'Tesla financial report')} | "
            f"page {doc.metadata.get('page', 'unknown')}]:\n{doc.page_content}"
            for index, doc in enumerate(documents, start=1)
        )
        prompt = f"""You are a financial-report question answering assistant.

Answer the user's question using only the retrieved context from the provided Tesla financial report.

Rules:
1. Use the retrieved context as the primary source of truth.
2. Do not invent facts or numbers.
3. Do not fabricate citations.
4. Do not use external knowledge to fill gaps.
5. If the context is insufficient, say exactly: {MISSING_EVIDENCE_MESSAGE}
6. Give a concise and factual answer.
7. Preserve the reporting period and units from the source.
8. When useful, mention the relevant report page.

Retrieved Context:
{context}

Question:
{question}
"""
        response = get_llm().invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return answer, _source_details(documents), retrieval.expansions
    except RuntimeError as error:
        raise RAGError(str(error)) from error
    except Exception as error:
        raise RAGError("Unable to generate an answer right now. Please try again.") from error
