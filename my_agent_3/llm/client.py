"""Executive summary: stdlib-only client for the organizer's house-model endpoint.

Reads ``MODEL_ENDPOINT``, ``MODEL_NAME`` and ``MODEL_TOKEN`` from the
environment, posts one chat-completion request with thinking disabled, and
caches responses on disk by content hash so re-runs are byte-identical and do
not re-bill the endpoint. Uses only the standard library: the agent image
guarantees no ``requests`` install.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

__all__ = ["LLMUnavailable", "chat_completion"]

DEFAULT_TIMEOUT = 45.0
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 1.5
_MAX_OUTPUT_TOKENS = 1024


class LLMUnavailable(RuntimeError):
    """Raised when the endpoint is unconfigured, unreachable, or refuses."""


def _endpoint_url() -> str:
    endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise LLMUnavailable("MODEL_ENDPOINT is not configured")
    return f"{endpoint}/v1/chat/completions"


def _cache_dir() -> pathlib.Path | None:
    override = os.environ.get("LLM_CACHE_DIR")
    root = (
        pathlib.Path(override)
        if override
        else pathlib.Path(__file__).resolve().parent.parent / "cache" / "llm"
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only install (the submission image runs as non-root): fall back
        # to the temp dir rather than disable text conditioning entirely.
        import tempfile

        root = pathlib.Path(tempfile.gettempdir()) / "my_agent_3_llm_cache"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
    return root


def _cached_call(key: str) -> str | None:
    root = _cache_dir()
    if root is None:
        return None
    path = root / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    except (OSError, ValueError, KeyError):
        return None  # a corrupt cache entry is a miss, never a failure


def _store(key: str, content: str) -> None:
    root = _cache_dir()
    if root is None:
        return
    path = root / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({"content": content}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # a failed cache write must never fail the forecast


def chat_completion(messages: list[dict], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Return the assistant's message content, cached by request content hash.

    Raises ``LLMUnavailable`` when the endpoint is unconfigured, when every
    retry is exhausted, or when a 4xx marks the request itself as faulty.
    Callers are expected to treat any failure as "run without text" rather
    than propagate.
    """
    payload = {
        "model": os.environ.get("MODEL_NAME", "house"),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        # Thinking is on by default on the house route; a bounded JSON signal
        # needs it off (docs/NVIDIA-STACK.md, raw-HTTP form).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    cached = _cached_call(key)
    if cached is not None:
        return cached

    url = _endpoint_url()
    token = os.environ.get("MODEL_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode("utf-8")

    last_error = "unknown error"
    for attempt in range(_MAX_RETRIES):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise LLMUnavailable(f"endpoint rejected the request: {exc.code}") from exc
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        except (KeyError, IndexError, ValueError) as exc:
            last_error = f"malformed response: {exc}"
        else:
            _store(key, content)
            return content
        if attempt + 1 < _MAX_RETRIES:
            time.sleep(_BASE_BACKOFF_S * (2**attempt))
    raise LLMUnavailable(f"house model call failed after {_MAX_RETRIES} attempts: {last_error}")
