from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL


def get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured. Please add it to your .env file.")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
