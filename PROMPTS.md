# Reproducibility Prompts

This document records the safe, user-facing prompts and requirements used to build the Tesla RAG Demo. It is intended to help reproduce the application without exposing credentials or private system instructions.

## Prompt 1: Initial Application

> Use the Tesla financial report PRD to build a complete local RAG application. Use Python, Streamlit, LangChain, `PyPDFLoader`, `RecursiveCharacterTextSplitter`, local `sentence-transformers/all-MiniLM-L6-v2` embeddings, ChromaDB, and OpenAI GPT-5.4 Mini. Load the API key from `.env`, keep the implementation CPU-compatible and simple, show retrieved sources and page metadata, maintain session-level question history, and include setup documentation, requirements, `.env.example`, and `.gitignore`.

## Prompt 2: Initial Smoke Test

> After building the application, run a small end-to-end smoke test with two sample Tesla financial-report questions. Verify PDF ingestion, chunking, local retrieval, source pages, and the live answer path when an API key is available. Report any limitation honestly when credentials or evidence are unavailable.

## Prompt 3: Query Expansion

> Add a basic query-expansion step. For each user question, use the existing LLM to generate exactly three alternative search queries. Search Chroma with the original query and the three expansions, deduplicate the results, and keep the best five chunks. Leave the rest of the RAG architecture unchanged.

## Prompt 4: Transparent UI

> Show the complete query-expansion flow in the Streamlit interface. Display the original user question, the three generated expansions, the retrieval behavior, the retrieved source chunks and page numbers, and the final grounded answer.

## Prompt 5: GitHub Publication

> Publish the completed Tesla RAG Demo to a new GitHub repository. Do not upload `.env`, API keys, ChromaDB data, virtual environments, caches, or other generated files. Verify the ignore rules and perform an additional secret check before uploading. Include this documentation file in the repository.

## Reproduction Notes

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` locally; never place it in source, documentation, or GitHub.
3. Place the Tesla PDF at `data/tesla_financial_report.pdf`.
4. Install `requirements.txt`.
5. Run `streamlit run app.py`.
6. Ask a question and inspect the expansion trace, retrieved evidence, and grounded answer.

No API key, token, or private chat-system instruction is included in this file.
