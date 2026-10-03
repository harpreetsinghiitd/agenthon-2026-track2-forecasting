"""Content-hash cache for LLM calls — read-through, write-on-miss.

Role in the pipeline
--------------------
Phase 3's determinism contract requires that identical (model, prompt, schema)
triples produce identical outputs.  The cache guarantees this even when the
House model is non-deterministic: the *first* call's response becomes the
canonical answer for that key.

Key = ``sha256(model_name + prompt + schema_json)`` (truncated to 32 hex chars).
Value = JSON object with ``content`` (the assistant text) and ``meta`` (timestamps,
model, etc.).  Stored under ``cache/llm/<key>.json``.

Read-through semantics:
* ``get(key)`` → returns cached content or ``None``.
* ``set(key, content)`` → writes atomically (tmp + rename).

The cache is *not* an LRU — it is append-only.  The organiser's scoring
environment provides a fresh cache dir per run; local smoke runs can share one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

__all__ = ["LLMCacheError", "LLMCache"]

_LOG = logging.getLogger(__name__)


class LLMCacheError(RuntimeError):
    """Raised on cache corruption or I/O failure."""


def _cache_key(model: str, prompt: str, schema: dict[str, Any] | None) -> str:
    """Deterministic cache key: model + prompt + canonical schema."""
    schema_str = json.dumps(schema, sort_keys=True, separators=(",", ":")) if schema else ""
    blob = f"{model}|{prompt}|{schema_str}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class LLMCache:
    """Disk-backed content-hash cache for LLM responses."""

    def __init__(self, cache_dir: Path | str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, model: str, prompt: str, schema: dict[str, Any] | None = None) -> str | None:
        """Return cached content or ``None`` if missing."""
        key = _cache_key(model, prompt, schema)
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("content")
        except (json.JSONDecodeError, OSError) as exc:
            _LOG.warning("Cache read failed for %s: %s; treating as miss", key, exc)
            return None

    def set(self, model: str, prompt: str, content: str, schema: dict[str, Any] | None = None) -> None:
        """Write content atomically under the computed key."""
        key = _cache_key(model, prompt, schema)
        tmp = self._dir / f"{key}.tmp"
        path = self._dir / f"{key}.json"
        payload = {
            "content": content,
            "model": model,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "schema_hash": hashlib.sha256(
                (json.dumps(schema, sort_keys=True) if schema else "").encode("utf-8")
            ).hexdigest()[:16],
        }
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise LLMCacheError(f"Cache write failed for {key}: {exc}") from exc

    def has(self, model: str, prompt: str, schema: dict[str, Any] | None = None) -> bool:
        """Check existence without reading the full payload."""
        return self._path(_cache_key(model, prompt, schema)).is_file()