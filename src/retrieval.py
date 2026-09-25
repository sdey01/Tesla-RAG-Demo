from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import TOP_K
from src.llm import get_llm


EXPANSION_COUNT = 3


@dataclass
class RetrievalResult:
    documents: list[Document]
    expansions: list[str]


def expand_query(question: str) -> list[str]:
    prompt = f"""Generate exactly {EXPANSION_COUNT} alternative search queries for the user's question.
Keep each query concise, factual, and focused on wording likely to appear in a Tesla financial report.
Return only the queries, one per line, with no explanations.

User question:
{question}
"""
    response = get_llm().invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    expansions = []
    for line in content.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned and cleaned not in expansions:
            expansions.append(cleaned)
    if len(expansions) < EXPANSION_COUNT:
        raise RuntimeError("Unable to generate three query expansions. Please try again.")
    return expansions[:EXPANSION_COUNT]


def retrieve_documents(vectorstore: Chroma, question: str) -> RetrievalResult:
    expansions = expand_query(question)
    search_queries = [question, *expansions]
    documents_by_id: dict[str, tuple[float, Document]] = {}
    for search_query in search_queries:
        results = vectorstore.similarity_search_with_relevance_scores(search_query, k=TOP_K)
        for document, score in results:
            chunk_id = str(document.metadata.get("chunk_id", document.page_content))
            if chunk_id not in documents_by_id or score > documents_by_id[chunk_id][0]:
                documents_by_id[chunk_id] = (score, document)
    ranked_documents = sorted(documents_by_id.values(), key=lambda item: item[0], reverse=True)
    return RetrievalResult(
        documents=[document for _, document in ranked_documents[:TOP_K]],
        expansions=expansions,
    )
