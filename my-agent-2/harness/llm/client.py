"""House model client — the only network egress in the container.

Role in the pipeline
--------------------
Phase 3 introduces the LLM as a feature extractor over the frozen corpus.
This module wraps the organiser's audited model endpoint.  It:

* POSTs to ``$MODEL_ENDPOINT/v1/chat/completions`` with a JSON body.
* Forces ``enable_thinking=false`` via ``extra_body`` so the House model
  returns structured JSON without chain-of-thought tokens.
* Retries with exponential backoff (base 1.5s, max 4 attempts).
* Logs every call: prompt hash, response hash, duration, status.

The client is deliberately thin — no retry logic for 4xx (those are prompt
bugs), no request/response transformation beyond the envelope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

__all__ = ["LLMClientError", "LLMClient"]

_LOG = logging.getLogger(__name__)

#: Environment variable names (validated at client instantiation, not import).
_MODEL_ENDPOINT_ENV = "MODEL_ENDPOINT"
_MODEL_NAME_ENV = "MODEL_NAME"
_MODEL_TOKEN_ENV = "MODEL_TOKEN"


class LLMClientError(RuntimeError):
    """Raised when the model endpoint returns an error or the response is malformed."""


@dataclass(frozen=True, slots=True)
class _CallRecord:
    """What we log for audit / debugging."""
    prompt_hash: str
    response_hash: str | None
    duration_ms: int
    status: int
    error: str | None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"{var} must be set in the environment")
    return val


class LLMClient:
    """Synchronous House model client with retries and audit logging."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        token: str | None = None,
        max_retries: int = 4,
        base_backoff: float = 1.5,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = (endpoint or _require_env(_MODEL_ENDPOINT_ENV)).rstrip("/") + "/v1/chat/completions"
        self._model = model or _require_env(_MODEL_NAME_ENV)
        self._token = token or _require_env(_MODEL_TOKEN_ENV)
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Call the House model and return the raw assistant content.

        Args:
            system_prompt: Fixed system instructions (cache key component).
            user_prompt: Per-document / per-card prompt.
            schema: Optional JSON Schema; passed as ``response_format`` for
                structured output.  If omitted, plain text is returned.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Output token budget.

        Returns:
            The assistant message content as a string.

        Raises:
            LLMClientError: non-2xx after retries, or response missing content.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}

        prompt_hash = _sha256(system_prompt + "|" + user_prompt)
        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                resp = self._session.post(self._endpoint, json=payload, timeout=self._timeout)
                duration_ms = int((time.perf_counter() - start) * 1000)
            except requests.RequestException as exc:
                duration_ms = int((time.perf_counter() - start) * 1000)
                _LOG.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt + 1, self._max_retries, exc
                )
                if attempt >= self._max_retries:
                    raise LLMClientError(f"LLM request failed after retries: {exc}") from exc
                time.sleep(self._base_backoff * (2**attempt))
                attempt += 1
                continue

            if 200 <= resp.status_code < 300:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise LLMClientError(f"Malformed response: {data}") from exc
                _LOG.info(
                    "LLM call ok | prompt=%s resp=%s dur=%dms",
                    prompt_hash,
                    _sha256(content),
                    duration_ms,
                )
                return content

            # 4xx = prompt bug (don't retry); 5xx = transient
            if 400 <= resp.status_code < 500:
                _LOG.error("LLM 4xx: %s %s", resp.status_code, resp.text[:500])
                raise LLMClientError(f"LLM {resp.status_code}: {resp.text[:500]}")
            if attempt >= self._max_retries:
                _LOG.error("LLM 5xx exhausted: %s %s", resp.status_code, resp.text[:500])
                raise LLMClientError(f"LLM {resp.status_code} after retries: {resp.text[:500]}")

            _LOG.warning(
                "LLM 5xx (attempt %d/%d): %s", attempt + 1, self._max_retries, resp.status_code
            )
            time.sleep(self._base_backoff * (2**attempt))
            attempt += 1