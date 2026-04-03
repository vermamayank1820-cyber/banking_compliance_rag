"""
services/audit.py
─────────────────
Lightweight governance / audit trail.

Events are appended as newline-delimited JSON (JSONL) to AUDIT_LOG_PATH.
Each line is a self-contained JSON object — easy to tail, grep, or import
into any analytics tool later.

What we log (and what we deliberately don't):
  ✓ Event type and timestamp
  ✓ Filenames involved (not content)
  ✓ Questions asked (truncated to 200 chars to avoid PII over-capture)
  ✓ Source citations returned with each answer
  ✓ Confidence / threshold info
  ✗ Raw document text   — never stored in the audit log
  ✗ Full user sessions  — only individual events
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import AUDIT_LOG_PATH, MAX_AUDIT_DISPLAY


def _write(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


# ─── Public log helpers ───────────────────────────────────────────────────────

def log_upload(filenames: list, replace_existing: bool) -> None:
    _write({
        "event": "document_upload",
        "filenames": filenames,
        "replace_existing": replace_existing,
    })


def log_indexing_start(filenames: list) -> None:
    _write({"event": "indexing_start", "filenames": filenames})


def log_indexing_complete(indexed: list, chunk_count: int, skipped: list) -> None:
    _write({
        "event": "indexing_complete",
        "indexed": indexed,
        "skipped": skipped,
        "chunk_count": chunk_count,
    })


def log_indexing_failed(filenames: list, error: str) -> None:
    _write({
        "event": "indexing_failed",
        "filenames": filenames,
        "error": error[:300],
    })


def log_question(question: str, filter_used: Optional[str] = None) -> None:
    _write({
        "event": "question_asked",
        "question": question[:200],
        "filter": filter_used,
    })


def log_answer(
    question: str,
    sources: list,
    max_confidence: float,
    low_confidence: bool,
) -> None:
    _write({
        "event": "answer_generated",
        "question": question[:200],
        "sources": sources,
        "max_confidence": round(max_confidence, 4),
        "low_confidence": low_confidence,
    })


def log_no_answer(question: str, reason: str) -> None:
    _write({
        "event": "no_answer",
        "question": question[:200],
        "reason": reason,
    })


# ─── Log reader ───────────────────────────────────────────────────────────────

def get_recent_logs(n: int = MAX_AUDIT_DISPLAY) -> list:
    """Return the most-recent n events, newest first."""
    path = Path(AUDIT_LOG_PATH)
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = []
    for ln in lines:
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return list(reversed(records[-n:]))
