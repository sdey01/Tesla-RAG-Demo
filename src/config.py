from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=True)
if ENV_FILE.exists():
    os.environ["OPENAI_API_KEY"] = dotenv_values(ENV_FILE).get("OPENAI_API_KEY") or ""

APP_TITLE = "Tesla Financial Report RAG Assistant"
PDF_PATH = Path(os.getenv("TESLA_PDF_PATH", PROJECT_ROOT / "data" / "tesla_financial_report.pdf"))
if not PDF_PATH.is_absolute():
    PDF_PATH = PROJECT_ROOT / PDF_PATH
CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
INDEX_MARKER = CHROMA_PERSIST_DIRECTORY / ".index_complete"
COLLECTION_NAME = "tesla_financial_report"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
OPENAI_MODEL = "gpt-5.4-mini"

MISSING_EVIDENCE_MESSAGE = (
    "I could not find sufficient information in the retrieved Tesla financial report "
    "context to answer that question reliably."
)
