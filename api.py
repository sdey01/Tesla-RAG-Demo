from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGError, answer_question
from src.vectorstore import VectorStoreError, get_vectorstore


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

app = FastAPI(title="Tesla Financial Report RAG API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    expansions: list[str]
    sources: list[dict[str, object]]


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    try:
        vectorstore = get_vectorstore()
        answer, sources, expansions = answer_question(question, vectorstore)
        return AskResponse(
            question=question,
            answer=answer,
            expansions=expansions,
            sources=sources,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Tesla financial report PDF was not found. Please place the PDF in the configured data directory.",
        ) from None
    except VectorStoreError:
        raise HTTPException(status_code=503, detail="Unable to prepare the document index.") from None
    except RAGError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
