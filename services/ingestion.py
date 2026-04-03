"""
services/ingestion.py
─────────────────────
Handles PDF loading, enriched chunking, embedding, and FAISS index management.

Key design decisions:
- Metadata (filename, page, upload_timestamp, has_table) is attached at
  ingestion so retrieval can surface citations and apply filters later.
- Incremental indexing: files whose content hash matches the registry entry
  are skipped; only new/changed files are embedded and merged into the store.
- Force-rebuild clears the registry and rebuilds from scratch — used when
  the user chooses "Replace all existing PDFs".
- Table detection uses pdfplumber (optional). If not installed the app works
  identically; pages are simply not tagged with has_table=True.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, DB_FAISS_PATH, REGISTRY_PATH,
)


# ─── File registry helpers ────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """MD5 of file bytes — used to detect whether a PDF has changed."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_registry() -> dict:
    """Return the document registry (filename → metadata dict)."""
    p = Path(REGISTRY_PATH)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_registry(registry: dict) -> None:
    p = Path(REGISTRY_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, indent=2))


# ─── Table detection (optional pdfplumber enhancement) ────────────────────────

def _table_pages(pdf_path: Path) -> set:
    """Return 0-indexed page numbers that contain at least one table."""
    try:
        import pdfplumber  # optional dependency
        pages_with_tables: set = set()
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                if page.extract_tables():
                    pages_with_tables.add(i)
        return pages_with_tables
    except Exception:
        # pdfplumber not installed or parse error — degrade gracefully
        return set()


# ─── Core chunking logic ──────────────────────────────────────────────────────

def load_and_chunk_pdf(pdf_path: Path) -> list:
    """
    Load a PDF, enrich page metadata, then split into overlapping chunks.

    Metadata added per chunk:
      filename        — bare filename (e.g. "basel_framework.pdf")
      page            — 0-indexed page number from PyPDFLoader
      upload_timestamp — ISO UTC timestamp of this ingestion run
      total_pages     — total pages in the document
      has_table       — True if the source page contains a table (pdfplumber)
      source          — normalised to bare filename (overrides full path)
    """
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        return []

    upload_ts = datetime.now(timezone.utc).isoformat()
    total_pages = len(pages)
    table_pg = _table_pages(pdf_path)

    for page_doc in pages:
        raw_page = page_doc.metadata.get("page", 0)
        page_doc.metadata.update({
            "filename": pdf_path.name,
            "source": pdf_path.name,          # normalise — was the full path
            "upload_timestamp": upload_ts,
            "total_pages": total_pages,
            "has_table": raw_page in table_pg,
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph/sentence boundaries before hard-splitting
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


# ─── Embedding model (shared / cached by callers) ─────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # unit vectors → scores in [0,1]
    )


# ─── Vector store build / update ──────────────────────────────────────────────

def build_vectorstore(pdf_files: list, force_rebuild: bool = False) -> dict:
    """
    Build or incrementally update the FAISS vector store.

    Args:
        pdf_files:     iterable of Path-like objects for all current PDFs.
        force_rebuild: if True, ignore registry and rebuild from scratch.

    Returns a dict:
        indexed  — list of filenames that were newly embedded
        skipped  — list of filenames that were unchanged and skipped
        total_new_chunks — number of chunks newly added

    Incremental logic:
    - A file is skipped when its MD5 hash matches the registry AND the FAISS
      store already exists on disk.
    - New/changed files are embedded and *merged* into the existing store.
    - On force_rebuild the existing store is replaced entirely.
    """
    registry: dict = {} if force_rebuild else load_registry()
    faiss_path = Path(DB_FAISS_PATH)

    new_chunks: list = []
    indexed_files: list = []
    skipped_files: list = []

    for raw_path in pdf_files:
        pdf_path = Path(raw_path)
        h = file_hash(pdf_path)
        existing = registry.get(pdf_path.name, {})

        already_ok = (
            not force_rebuild
            and existing.get("hash") == h
            and existing.get("status") == "indexed"
            and faiss_path.exists()
        )

        if already_ok:
            skipped_files.append(pdf_path.name)
            continue

        chunks = load_and_chunk_pdf(pdf_path)
        new_chunks.extend(chunks)
        indexed_files.append(pdf_path.name)

        last_page = chunks[-1].metadata.get("total_pages", 0) if chunks else 0
        registry[pdf_path.name] = {
            "hash": h,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
            "page_count": last_page,
            "status": "indexed",
        }

    if not new_chunks and not skipped_files:
        raise ValueError("No content could be extracted from the provided PDFs.")

    embedding_model = get_embedding_model()

    if new_chunks:
        if skipped_files and faiss_path.exists():
            # Incremental: merge new embeddings into the existing store
            existing_db = FAISS.load_local(
                DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
            )
            new_db = FAISS.from_documents(new_chunks, embedding_model)
            existing_db.merge_from(new_db)
            existing_db.save_local(DB_FAISS_PATH)
        else:
            # Full build (first run or force_rebuild)
            db = FAISS.from_documents(new_chunks, embedding_model)
            db.save_local(DB_FAISS_PATH)

    # Remove registry entries for PDFs that no longer exist in data/
    current_names = {Path(p).name for p in pdf_files}
    registry = {k: v for k, v in registry.items() if k in current_names}
    save_registry(registry)

    return {
        "indexed": indexed_files,
        "skipped": skipped_files,
        "total_new_chunks": len(new_chunks),
    }
