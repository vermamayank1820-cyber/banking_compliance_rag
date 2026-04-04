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

import dataclasses
import json
import math
import numbers
from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from config import AUDIT_LOG_PATH, MAX_AUDIT_DISPLAY


def safe_serialize(obj: Any, _seen: set[int] | None = None) -> Any:
    """
    Convert arbitrary Python objects into JSON-safe values without flattening
    all unknown types to plain strings.

    Rules:
    - Preserve native JSON primitives as-is
    - Convert datetime/date/time to ISO 8601 strings
    - Convert mappings and sequences recursively
    - Convert dataclasses and pydantic-style objects to structured dicts
    - Convert enums to their underlying value
    - Convert Path to string
    - Convert bytes safely to UTF-8 text when possible
    - Fall back to a typed repr payload for unsupported objects
    - Protect against circular references
    """
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)

    if isinstance(obj, numbers.Integral):
        return int(obj)

    if isinstance(obj, numbers.Real):
        value = float(obj)
        return value if math.isfinite(value) else str(value)

    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Enum):
        return safe_serialize(obj.value, _seen)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "__type__": "bytes",
                "encoding": "utf-8",
                "value": obj.decode("utf-8", errors="replace"),
            }

    item_method = getattr(obj, "item", None)
    if callable(item_method):
        try:
            return safe_serialize(item_method(), _seen)
        except Exception:
            pass

    tolist_method = getattr(obj, "tolist", None)
    if callable(tolist_method):
        try:
            return safe_serialize(tolist_method(), _seen)
        except Exception:
            pass

    obj_id = id(obj)
    if obj_id in _seen:
        return {"__type__": type(obj).__name__, "__circular__": True}

    if isinstance(obj, Mapping):
        _seen.add(obj_id)
        try:
            return {
                str(key): safe_serialize(value, _seen)
                for key, value in obj.items()
            }
        finally:
            _seen.discard(obj_id)

    if isinstance(obj, (list, tuple, set, frozenset)):
        _seen.add(obj_id)
        try:
            return [safe_serialize(item, _seen) for item in obj]
        finally:
            _seen.discard(obj_id)

    if dataclasses.is_dataclass(obj):
        _seen.add(obj_id)
        try:
            return safe_serialize(dataclasses.asdict(obj), _seen)
        finally:
            _seen.discard(obj_id)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        _seen.add(obj_id)
        try:
            return safe_serialize(model_dump(), _seen)
        finally:
            _seen.discard(obj_id)

    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        _seen.add(obj_id)
        try:
            return safe_serialize(dict_method(), _seen)
        finally:
            _seen.discard(obj_id)

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict) and obj_dict:
        _seen.add(obj_id)
        try:
            public_fields = {
                key: value
                for key, value in obj_dict.items()
                if not key.startswith("_")
            }
            if public_fields:
                return {
                    "__type__": type(obj).__name__,
                    "fields": safe_serialize(public_fields, _seen),
                }
        finally:
            _seen.discard(obj_id)

    return {
        "__type__": type(obj).__name__,
        "__repr__": repr(obj),
    }


def _write(event: dict) -> None:
    record = dict(event)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    serialized = safe_serialize(record)
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(serialized, ensure_ascii=False) + "\n")


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
