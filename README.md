# Banking Compliance RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) application for banking compliance documents. It lets users upload one or more PDF files, index them into a FAISS vector store, and ask grounded questions through a Streamlit chat interface.

## Overview

The application uses:

- `Streamlit` for the UI
- `LangChain` for retrieval orchestration
- `FAISS` for local vector storage
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `Groq` with `llama-3.1-8b-instant` for answer generation

The assistant is configured to answer only from the uploaded document context. If the answer is not present in the indexed content, it responds with `I do not know.`

## Features

- Upload a single PDF or multiple PDFs
- Store uploaded PDFs in the local `data/` folder
- Rebuild the FAISS knowledge base from the uploaded files
- Ask banking compliance questions in a chat UI
- Show retrieved source document names and page numbers
- Replace existing PDFs when you want a clean knowledge base

## Project Structure

```text
Banking Compliance/
├── medibot.py
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── requirements.txt
├── data/
│   └── *.pdf
└── vectorstore/
    └── db_faiss/
        ├── index.faiss
        └── index.pkl
```

## Main Files

### `medibot.py`

Primary Streamlit application.

Responsibilities:

- renders the banking compliance chat UI
- supports inline PDF upload through the `+` control
- saves PDFs into `data/`
- rebuilds the FAISS vector database
- loads the vector store and runs retrieval QA
- displays answers with supporting source references

### `create_memory_for_llm.py`

Standalone ingestion script that:

- loads all PDFs from `data/`
- splits them into chunks
- creates embeddings
- saves the FAISS index into `vectorstore/db_faiss`

Use this when you want to rebuild the vector store manually outside the UI.

### `connect_memory_with_llm.py`

Standalone retrieval script for local testing with a Hugging Face text-generation pipeline instead of the Streamlit app.

## How It Works

### 1. Document ingestion

PDFs are read from the `data/` directory using `PyPDFLoader`.

### 2. Text chunking

Documents are split with:

- chunk size: `500`
- chunk overlap: `50`

### 3. Embedding generation

Each chunk is embedded using:

- `sentence-transformers/all-MiniLM-L6-v2`

### 4. Vector storage

Embeddings are stored locally in FAISS under:

- `vectorstore/db_faiss`

### 5. Retrieval and answer generation

When a user asks a question:

- top `3` relevant chunks are retrieved
- the retrieved context is passed to the Groq LLM
- the assistant answers only from that context

## Setup

### Prerequisites

- Python 3.10+
- a Groq API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can also provide the key through Streamlit secrets.

## Running the App

Start the Streamlit UI:

```bash
streamlit run medibot.py
```

Then:

1. Open the app in your browser.
2. Use the `+` button to upload one or more banking compliance PDFs.
3. Click `Save and Rebuild`.
4. Ask questions in the bottom chat bar.

## Manual Vector Store Rebuild

If you want to rebuild the vector database from the existing files in `data/`, run:

```bash
python create_memory_for_llm.py
```

## Usage Notes

- The app supports both single-PDF and multi-PDF workflows.
- Uploaded PDFs are persisted in `data/`.
- The vector store is persisted in `vectorstore/db_faiss`.
- If you upload a new set of documents and want to remove the old set, use `Replace existing PDFs`.
- If the vector store is missing, the app will prompt you to upload PDFs and rebuild the knowledge base.

## Current Retrieval Prompt Behavior

The assistant is instructed to:

- behave as a banking compliance assistant
- use only the provided context
- avoid inventing regulatory or compliance guidance
- answer concisely

## Limitations

- The quality of answers depends on the uploaded PDFs.
- This is a document-grounded assistant, not a live regulatory feed.
- It does not validate whether a document is current, legally binding, or jurisdiction-specific.
- Streamlit layout customization is limited, so the modern chat-style UI is implemented with custom layout and CSS adjustments.

## Example Questions

- What is the Basel II framework?
- What is a credit conversion factor?
- How are off-balance-sheet exposures treated?
- What does the leverage ratio cover?

## Future Improvements

- automatic stale-index detection
- document removal from the UI
- source snippet highlighting
- better metadata display
- support for additional document formats
- stronger compliance-specific prompt templates

