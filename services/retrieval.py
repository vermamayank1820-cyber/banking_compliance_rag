"""
services/retrieval.py
─────────────────────
Score-aware retrieval on top of FAISS.

Design:
- Uses similarity_search_with_score (raw L2 distances, lower = more similar)
  instead of similarity_search_with_relevance_scores, which can return scores
  outside [0, 1] for the MiniLM embedding model and trigger LangChain warnings.
- Converts L2 distance → cosine similarity via: score = max(0, 1 - L2²/2)
  This is mathematically correct for unit-normalised embeddings and clamps
  the range to [0, 1] for embeddings that deviate slightly from unit norm.
- Filters out chunks below SIMILARITY_THRESHOLD before returning them to the
  caller; the caller then decides to show "I do not know" if nothing passes.
- Optional filename filter restricts search to a single uploaded document;
  we fetch 2×k candidates first so we still get k results after filtering.
- Returns (relevant_chunks, above_threshold, max_score).

Modern RAG additions:
- retrieve_multi_query(): accepts multiple query strings (original + HyDE
  hypothetical document + rephrased variants), retrieves candidates for each,
  deduplicates by content hash, and returns the top-k unique chunks ranked by
  their best score across all queries.  This bridges vocabulary gaps — when the
  user's words are absent from the document, the HyDE or rephrased variant
  still finds the right chunks.
"""

import hashlib
import math
from typing import Optional

from langchain_community.vectorstores import FAISS

from config import TOP_K, SIMILARITY_THRESHOLD


def _l2_to_cosine(l2_distance: float) -> float:
    """
    Convert a FAISS L2 distance to a [0, 1] cosine-similarity score.

    For unit-normalised vectors:
        cosine_similarity = 1 - L2² / 2

    We clamp to [0, 1] to handle minor floating-point deviations from
    unit norm (which would otherwise yield scores just outside that range).
    """
    cosine = 1.0 - (l2_distance ** 2) / 2.0
    return max(0.0, min(1.0, cosine))


def retrieve(
    db: FAISS,
    query: str,
    k: int = TOP_K,
    filter_filename: Optional[str] = None,
) -> tuple[list, bool, float]:
    """
    Retrieve relevant chunks for a query.

    Args:
        db:              Loaded FAISS vector store.
        query:           User question.
        k:               Maximum number of chunks to return.
        filter_filename: If set, only chunks from this filename are returned.

    Returns:
        relevant        — list of (Document, cosine_score) tuples above threshold
        above_threshold — True if at least one chunk scored >= SIMILARITY_THRESHOLD
        max_score       — highest cosine score among all fetched candidates
    """
    fetch_k = k * 2 if filter_filename else k

    try:
        # Returns (Document, L2_distance) — lower distance = more similar
        raw_docs = db.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        return [], False, 0.0

    if not raw_docs:
        return [], False, 0.0

    # Convert L2 distances → cosine similarity scores
    scored_docs = [(doc, _l2_to_cosine(dist)) for doc, dist in raw_docs]

    # Apply filename filter post-retrieval (FAISS CPU has no native metadata filter)
    if filter_filename:
        scored_docs = [
            (doc, score)
            for doc, score in scored_docs
            if doc.metadata.get("filename") == filter_filename
        ]

    scored_docs = scored_docs[:k]

    if not scored_docs:
        return [], False, 0.0

    max_score = max(score for _, score in scored_docs)
    above_threshold = max_score >= SIMILARITY_THRESHOLD

    # Only pass chunks that individually meet the threshold
    relevant = [
        (doc, score) for doc, score in scored_docs
        if score >= SIMILARITY_THRESHOLD
    ]

    return relevant, above_threshold, max_score


def retrieve_multi_query(
    db: FAISS,
    queries: list[str],
    k: int = TOP_K,
    filter_filename: Optional[str] = None,
) -> tuple[list, bool, float]:
    """
    Modern RAG retrieval: merge results from multiple query strings.

    Each query string (original question, HyDE hypothetical passage, or
    rephrased variant) is searched independently.  Results are deduplicated
    by a content hash so the same chunk is never returned twice, and each
    chunk keeps its *best* score across all queries.  The final list is
    re-ranked by that best score and truncated to k.

    This is the core mechanism that handles vocabulary mismatch:
      • A user asks "what is bank leverage?"
      • HyDE generates a passage containing "Tier 1 capital ratio", "leverage
        ratio", "Basel III Article 429" — the document's actual phrasing.
      • The HyDE embedding finds the right chunks even though the user's
        two-word query didn't.

    Args:
        db:              Loaded FAISS vector store.
        queries:         List of query strings (original + HyDE + variants).
        k:               Maximum unique chunks to return.
        filter_filename: If set, only chunks from this filename are returned.

    Returns:
        relevant        — list of (Document, cosine_score) tuples above threshold
        above_threshold — True if at least one chunk scored >= SIMILARITY_THRESHOLD
        max_score       — highest cosine score among all deduped candidates
    """
    # content_hash -> (Document, best_cosine_score)
    seen: dict[str, tuple] = {}

    fetch_k = k * 2 if filter_filename else k

    for query in queries:
        if not query or not query.strip():
            continue
        try:
            raw_docs = db.similarity_search_with_score(query, k=fetch_k)
        except Exception:
            continue

        scored = [(doc, _l2_to_cosine(dist)) for doc, dist in raw_docs]

        if filter_filename:
            scored = [
                (doc, score) for doc, score in scored
                if doc.metadata.get("filename") == filter_filename
            ]

        for doc, score in scored:
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            if key not in seen or score > seen[key][1]:
                seen[key] = (doc, score)

    if not seen:
        return [], False, 0.0

    # Re-rank by best score, keep top k
    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:k]
    max_score = merged[0][1]
    above_threshold = max_score >= SIMILARITY_THRESHOLD

    relevant = [(doc, score) for doc, score in merged if score >= SIMILARITY_THRESHOLD]
    return relevant, above_threshold, max_score
