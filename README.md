# Tesla Financial Report RAG Assistant

A small, local Streamlit RAG application for asking grounded questions about one Tesla financial report. It loads the PDF with LangChain's `PyPDFLoader`, splits it with `RecursiveCharacterTextSplitter`, embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2`, expands each question into three report-focused search queries, stores chunks in ChromaDB, and sends the top five retrieved chunks to OpenAI GPT-5.4 Mini.

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key for live answers
- A CPU-only machine is supported

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project includes a `.env` placeholder. Add your key:

```text
OPENAI_API_KEY=your_api_key_here
```

Never commit `.env`. It is already ignored by Git. Copy `.env.example` when setting up another checkout.

## PDF placement

Place the Tesla report at:

```text
data/tesla_financial_report.pdf
```

The supplied report can be copied there as `tsla-20231231-gen.pdf`, or set `TESLA_PDF_PATH` in `.env` to another local PDF path. The first startup loads and chunks the PDF and creates a local `chroma_db/` index. Later startups reuse that index.

## Run

```powershell
streamlit run app.py
```

The browser UI shows the original question, all three generated expansions, the five retrieved source chunks, source filename, page numbers, generated answer, and session-only question history in the sidebar.

## How it works

```text
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter
    -> local MiniLM embeddings -> ChromaDB
    -> 3 query expansions -> top 5 deduplicated results -> GPT-5.4 Mini -> answer + sources
```

The prompt instructs the model to use only retrieved report context and to say when the evidence is insufficient. No web search, trading recommendations, authentication, or permanent question history is included.

## Example questions

- What was Tesla's total revenue?
- How did automotive revenue change year over year?
- What was Tesla's net income?
- What were Tesla's major operating expenses?
- What risks did Tesla identify?
- What was Tesla's cash and cash equivalents?
- What did Tesla report about capital expenditures?

## Troubleshooting

- **PDF not found:** confirm the file is at `data/tesla_financial_report.pdf`, or set `TESLA_PDF_PATH` in `.env`.
- **Missing API key:** set `OPENAI_API_KEY` in `.env` and restart Streamlit.
- **Stale index after replacing the PDF:** stop Streamlit, delete `chroma_db/`, then start it again.
- **Slow first startup:** downloading the MiniLM model and creating the first local index are one-time operations on a new machine.
- **LLM error:** verify the API key, model availability for your account, and network access.

## Project structure

```text
app.py
README.md
PROMPTS.md
requirements.txt
.env.example
.gitignore
data/tesla_financial_report.pdf
src/
  config.py
  embeddings.py
  ingestion.py
  llm.py
  rag_pipeline.py
  retrieval.py
  vectorstore.py
chroma_db/        # generated locally and ignored by Git
```
