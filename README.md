# Banking Compliance RAG Assistant

A Retrieval-Augmented Generation (RAG) application for banking compliance documents.
Upload PDFs, ask grounded questions — answers come strictly from your documents.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BANKING COMPLIANCE RAG                           │
├──────────────┬──────────────────────────────┬───────────────────────────┤
│   INGESTION  │         RETRIEVAL             │       GENERATION          │
│              │                              │                           │
│  PDF Upload  │   Query Expansion (HyDE)     │   Groq LLM                │
│      ↓       │          ↓                   │   llama-3.1-8b-instant    │
│  PyPDFLoader │   Multi-Query Variants       │          ↑                │
│      ↓       │          ↓                   │   Prompt Template         │
│  Text Chunks │   FAISS Vector Search        │   (context + history)     │
│  (800/150)   │          ↓                   │          ↑                │
│      ↓       │   L2 → Cosine Score          │   Top-K Chunks            │
│  Embeddings  │          ↓                   │   (threshold ≥ 0.20)      │
│  (MiniLM)    │   Threshold Filter           │                           │
│      ↓       │          ↓                   │   Fallback:               │
│  FAISS Index │   Source Citations           │   "Hey, sorry I don't     │
│  (local)     │                              │    know the answer..."    │
└──────────────┴──────────────────────────────┴───────────────────────────┘
```

---

## Project Structure

```
Banking Compliance/
│
├── medibot.py                  # Streamlit app — entry point
├── config.py                   # All tuneable constants (single source of truth)
├── create_memory_for_llm.py    # CLI: rebuild vector store from data/
├── connect_memory_with_llm.py  # CLI: local test with HuggingFace pipeline
├── requirements.txt
├── .env                        # GROQ_API_KEY (not committed)
│
├── services/
│   ├── ingestion.py            # PDF loading, chunking, FAISS build/update
│   ├── retrieval.py            # Score-aware retrieval + multi-query merge
│   └── audit.py                # Append-only JSONL governance log
│
├── data/                       # Uploaded PDFs (persisted)
│   └── *.pdf
│
├── vectorstore/
│   ├── db_faiss/               # FAISS index files
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── registry.json           # Document index (hash, chunk count, timestamp)
│
└── logs/
    └── audit.jsonl             # Governance / audit trail
```

---

## Component Responsibilities

### `medibot.py` — Application Layer
- Streamlit chat UI with sidebar document management
- Passes conversation history (last 6 turns) to the LLM as secondary context
- Coordinates ingestion, retrieval, and answer generation
- Renders source citations with relevance scores per answer

### `config.py` — Configuration
- Single file for all tuneable constants: chunk size, overlap, similarity threshold, model names, paths
- Change retrieval or model behaviour here without touching service code

### `services/ingestion.py` — Ingestion Service
- Loads PDFs with `PyPDFLoader`, enriches chunk metadata (filename, page, table flag, timestamp)
- Incremental indexing: MD5 hash comparison skips unchanged files
- Force-rebuild replaces the entire index from scratch
- Optional `pdfplumber` table detection — degrades gracefully if not installed

### `services/retrieval.py` — Retrieval Service
- Converts raw FAISS L2 distances to cosine similarity scores `[0, 1]`
- Single-query and multi-query (deduplicated, best-score merge) retrieval modes
- Returns `(chunks, above_threshold, max_score)` — caller decides on fallback

### `services/audit.py` — Audit Service
- Append-only JSONL log at `logs/audit.jsonl`
- Logs: uploads, indexing events, questions asked (truncated to 200 chars), answers with confidence scores
- Deliberately excludes raw document text and full session data

---

## Data Flow

### Ingestion (one-time or on upload)

```
PDF file(s)
    └─▶ PyPDFLoader          — extract raw text per page
         └─▶ pdfplumber      — detect table pages (optional)
              └─▶ RecursiveCharacterTextSplitter  — chunk (800 chars / 150 overlap)
                   └─▶ HuggingFaceEmbeddings      — embed each chunk (MiniLM)
                        └─▶ FAISS.save_local()    — persist index + registry
```

### Query (per user message)

```
User question
    ├─▶ HyDE               — generate hypothetical compliance passage (optional)
    ├─▶ Multi-Query        — 3 synonym-rephrased variants (optional)
    └─▶ retrieve_multi_query()
             └─▶ FAISS similarity search (per query variant)
                  └─▶ L2 → cosine score, dedup by content hash
                       └─▶ threshold filter (≥ 0.20)
                            ├── below threshold → "Hey, sorry I don't know..."
                            └── above threshold → build context string
                                     └─▶ PromptTemplate
                                          (context + conversation history)
                                               └─▶ Groq LLM → answer + citations
```

---

## Modern RAG Techniques

| Technique | Purpose | Config flag |
|---|---|---|
| **HyDE** | Generates a hypothetical doc excerpt to bridge vocabulary gaps between user phrasing and document language | `USE_HYDE = True` |
| **Multi-Query** | Rephrases the question 3 ways with regulatory synonyms to widen recall | `USE_MULTI_QUERY = True` |
| **Score-aware retrieval** | L2 → cosine conversion gives a calibrated `[0, 1]` score; threshold filters noise | `SIMILARITY_THRESHOLD = 0.20` |
| **Incremental indexing** | MD5 hash check skips unchanged PDFs on re-upload | always on |
| **Conversation history** | Last 6 turns injected into prompt as secondary context for follow-up awareness | always on |

---

## Key Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `SIMILARITY_THRESHOLD` | `0.20` | Minimum cosine score to treat a chunk as relevant |
| `TOP_K` | `5` | Max chunks retrieved per query |
| `MULTI_QUERY_COUNT` | `3` | Number of rephrased query variants |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model for answer generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer |

---

## Setup

**Prerequisites:** Python 3.10+, a Groq API key

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run medibot.py
```

**Or rebuild the vector store from the CLI:**

```bash
python create_memory_for_llm.py
```

---

## Answer Behaviour

| Situation | Response |
|---|---|
| Answer found in retrieved chunks | Direct, citation-backed answer |
| Follow-up question | Uses conversation history (last 6 turns) + retrieved context |
| No chunks above threshold | `"Hey, sorry I don't know the answer to this."` |
| No knowledge base loaded | `"Hey, sorry I don't know the answer to this."` |

---

## Example Questions

- What is the Basel III leverage ratio?
- What is a credit conversion factor?
- How are off-balance-sheet exposures treated?
- What are the CET1 capital buffer requirements?
- Explain more. *(follow-up — uses conversation history)*

---

## Limitations

- Answer quality depends entirely on the uploaded PDFs
- Not a live regulatory feed — documents must be manually updated
- Does not validate whether a document is current, legally binding, or jurisdiction-specific
- No user authentication or multi-tenant isolation

---

## Roadmap

| Phase | Feature |
|---|---|
| 2 | Hybrid search (vector + BM25 keyword) for exact regulatory term matching |
| 2 | Metadata filtering by document type, jurisdiction, effective date |
| 3 | Saved query history and user feedback mechanism |
| 3 | Exportable audit trails for regulatory examinations |
| 4 | Automated ingestion from regulatory RSS feeds |
| 4 | Scanned PDF support (OCR) and structured regulatory XML |
