# ─── Retrieval ────────────────────────────────────────────────────────────────
# Number of chunks to retrieve per query
TOP_K = 5

# Minimum cosine-similarity score to treat a chunk as useful.
# retrieval.py converts raw FAISS L2 distances → cosine scores via
#   score = max(0, 1 - L2² / 2)
# giving a reliable [0, 1] range regardless of embedding normalisation.
# Empirical calibration on MiniLM + Basel III PDF:
#   > 0.60  — highly relevant (leverage ratio, capital buffers, etc.)
#   0.25–0.59 — moderately relevant (terms referenced but not fully defined)
#   < 0.25  — noise / off-topic
# 0.20 is the practical lower bound to catch partial topic matches while
# still filtering genuinely unrelated queries.
SIMILARITY_THRESHOLD = 0.20

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # characters per chunk (up from 500 for better context)
CHUNK_OVERLAP = 150       # overlap keeps cross-boundary context (up from 50)

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"
REGISTRY_PATH = "vectorstore/registry.json"   # tracks indexed documents
AUDIT_LOG_PATH = "logs/audit.jsonl"           # append-only governance log

# ─── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# ─── Modern RAG ───────────────────────────────────────────────────────────────
# HyDE: generate a hypothetical document excerpt to bridge vocabulary gaps.
# When a query uses words absent from the documents, an LLM-generated passage
# that "answers" the question uses the document's own terminology, giving the
# embedder a much better retrieval signal.
USE_HYDE = True

# Multi-query: rephrase the question N ways with alternative regulatory synonyms,
# retrieve for each variant, deduplicate, and return the best-scoring unique chunks.
USE_MULTI_QUERY = True
MULTI_QUERY_COUNT = 3     # additional query variants (besides the original)

# ─── UI ───────────────────────────────────────────────────────────────────────
MAX_AUDIT_DISPLAY = 15    # number of recent audit events shown in the sidebar
