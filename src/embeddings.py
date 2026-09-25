from __future__ import annotations

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
