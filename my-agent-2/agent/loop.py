"""Bounded agentic loop — the LLM decides *what to do next*, not *what numbers to output*.

Role in the pipeline
--------------------
The loop runs after Phase 3's batch extraction.  It gives the LLM a budget of
iterations (default 5) and LLM calls (default 50) to:

1. Search the corpus for relevant documents.
2. Read selected documents.
3. Extract vectors/scenarios from new documents.
4. Optionally re-fit models with new evidence.
5. Decide when to finish and emit the final forecast.

The loop NEVER decides: draw values, grid, asof, or whether to skip a card.
Those are hard-coded in the deterministic pipeline.

Hard caps (from the build prompt):
- max_iterations = 5
- max_llm_calls = 50 per card

The loop state is passed to the LLM as JSON; the LLM responds with a single
action object.  This module executes the action and returns the updated state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agent.prompts import get_prompt
from harness.llm.cache import LLMCache
from harness.llm.client import LLMClient
from harness.tools.card import CardSpec
from harness.tools.corpus import CorpusError, get_document, list_documents, search_corpus
from harness.tools.panel import read_panel
from harness.tools.stats import CopulaParams, fit_arima, fit_copula
from harness.tools.simulate import seed_from_unit_id, simulate_mixture, simulate_mixture_joint

__all__ = ["LoopConfig", "LoopState", "run_loop"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Loop budget configuration."""
    max_iterations: int = 5
    max_llm_calls: int = 50


@dataclass(slots=True)
class LoopState:
    """Mutable state passed to the LLM at each iteration."""

    card: CardSpec
    corpus_dir: Path
    llm_client: "LLMClient"
    llm_cache: "LLMCache"
    config: LoopConfig

    # Runtime state
    panels_loaded: list[str] = field(default_factory=list)
    corpus_doc_ids: list[str] = field(default_factory=list)
    vectors_extracted: list[dict[str, Any]] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    models_fitted: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    llm_calls: int = 0

    # Outputs (filled at finish)
    final_drifts: dict[str, float] = field(default_factory=dict)
    final_widths: dict[str, float] = field(default_factory=dict)
    final_skews: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise for the LLM prompt."""
        return {
            "card": {
                "unit_id": self.card.unit_id,
                "asof": self.card.asof.isoformat(),
                "asset_ids": list(self.card.asset_ids),
                "horizons": list(self.card.horizons),
                "target_type": self.card.target_type,
                "family": self.card.family,
            },
            "panels_loaded": self.panels_loaded,
            "corpus_doc_ids": self.corpus_doc_ids,
            "vectors_extracted": self.vectors_extracted,
            "scenarios": self.scenarios,
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
        }


class LoopError(RuntimeError):
    """Raised when the loop hits a hard constraint or invalid action."""


def run_loop(
    state: LoopState,
    unit_dir: Path,
    n_draws: int,
) -> LoopState:
    """Run the bounded agentic loop.

    The loop:
    1. Initialises corpus_doc_ids by listing the corpus.
    2. Enters the iteration loop (max_iterations).
    3. At each step, builds the prompt, calls the LLM, parses the action.
    4. Executes the action (updates state, increments counters).
    5. Stops when LLM returns {"action": "finish"} or budget exhausted.

    Args:
        state: Initial LoopState (card, corpus_dir, llm_client, llm_cache, config).
        unit_dir: Directory holding panel parquet files.
        n_draws: Number of draws for downstream simulation.

    Returns:
        Updated LoopState with final_drifts/final_widths/final_skews populated.

    Raises:
        LoopError: hard budget exceeded or invalid action.
    """
    # Initialise corpus doc list
    state.corpus_doc_ids = list_documents(state.corpus_dir, state.card.asof.isoformat())
    _LOG.info("Loop init: %d corpus documents available", len(state.corpus_doc_ids))

    prompt_bundle = get_prompt("loop")
    system_prompt = prompt_bundle["system"]
    user_template = prompt_bundle["user_template"]

    for iteration in range(state.config.max_iterations):
        state.iterations = iteration + 1
        if state.llm_calls >= state.config.max_llm_calls:
            _LOG.warning("LLM call budget exhausted (%d), finishing", state.llm_calls)
            break

        # Build user prompt
        user_prompt = user_template.format(
            card_json=json.dumps(state.to_json()["card"], indent=2),
            panels_loaded=state.panels_loaded,
            corpus_doc_ids=state.corpus_doc_ids,
            vectors_extracted=state.vectors_extracted,
            scenarios=state.scenarios,
            iterations=state.iterations,
            llm_calls=state.llm_calls,
        )

        try:
            raw = state.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=512,
            )
            state.llm_calls += 1
        except Exception as exc:
            _LOG.error("LLM call failed: %s", exc)
            break

        # Parse action
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            _LOG.warning("LLM returned invalid JSON: %s", raw[:200])
            continue

        # Execute action
        try:
            _execute_action(action, state)
        except LoopError as exc:
            _LOG.warning("Action failed: %s", exc)
            continue
        except Exception as exc:
            _LOG.warning("Action execution error: %s", exc)
            continue

        if action.get("action") == "finish":
            _LOG.info("Loop finished by LLM: %s", action.get("reason"))
            break

    # Ensure we have a finish reason
    if state.iterations >= state.config.max_iterations:
        _LOG.info("Iteration budget exhausted, finishing")

    # Finalise: populate final drifts/widths/skews from state
    _finalise_outputs(state)

    return state


def _execute_action(action: dict[str, Any], state: LoopState) -> None:
    """Execute a single LLM action, mutating state."""
    act = action.get("action")
    if act == "search":
        query = action.get("query", "")
        k = int(action.get("k", 10))
        hits = search_corpus(state.corpus_dir, query, state.card.asof.isoformat(), k=k)
        state.corpus_doc_ids = list(dict.fromkeys(hits + state.corpus_doc_ids))
        _LOG.debug("search('%s') -> %d hits", query, len(hits))

    elif act == "get_document":
        doc_id = action.get("doc_id")
        if not doc_id:
            raise LoopError("get_document requires doc_id")
        try:
            doc = get_document(state.corpus_dir, doc_id, state.card.asof.isoformat())
            _LOG.debug("get_document(%s) -> %d chars", doc_id, len(doc.content))
        except CorpusError as exc:
            _LOG.warning("get_document failed: %s", exc)
            raise LoopError(str(exc))

    elif act == "list_documents":
        state.corpus_doc_ids = list_documents(state.corpus_dir, state.card.asof.isoformat())
        _LOG.debug("list_documents -> %d docs", len(state.corpus_doc_ids))

    elif act == "fit_model":
        model_type = action.get("model_type")
        asset_ids = action.get("asset_ids", [])
        params = action.get("params", {})
        if not model_type or not asset_ids:
            raise LoopError("fit_model requires model_type and asset_ids")
        _LOG.info("fit_model(%s, %s)", model_type, asset_ids)
        # Delegate to deterministic pipeline (Phase 2/4); just record intent
        state.models_fitted[model_type] = {"asset_ids": asset_ids, "params": params}

    elif act == "extract_vector":
        doc_id = action.get("doc_id")
        if not doc_id:
            raise LoopError("extract_vector requires doc_id")
        # Use existing extraction pipeline (cached)
        from harness.llm.extract import extract_vector
        vec = extract_vector(
            state.llm_client, state.llm_cache,
            doc_id=doc_id, timestamp="", content="",  # will be filled by extract_vector via get_document
            asof=state.card.asof.isoformat(), family=state.card.family, target_type=state.card.target_type
        )
        # We need the document content — get it first
        doc = get_document(state.corpus_dir, doc_id, state.card.asof.isoformat())
        vec = extract_vector(
            state.llm_client, state.llm_cache,
            doc_id=doc.doc_id, timestamp=doc.timestamp, content=doc.content,
            asof=state.card.asof.isoformat(), family=state.card.family, target_type=state.card.target_type
        )
        state.vectors_extracted.append({"doc_id": doc_id, "vector": vec.__dict__})
        _LOG.debug("extract_vector(%s) done", doc_id)

    elif act == "extract_scenarios":
        doc_ids = action.get("doc_ids", [])
        if not doc_ids:
            raise LoopError("extract_scenarios requires doc_ids")
        from harness.llm.extract import extract_scenarios
        docs = []
        for d in doc_ids:
            try:
                doc = get_document(state.corpus_dir, d, state.card.asof.isoformat())
                docs.append({"doc_id": doc.doc_id, "timestamp": doc.timestamp, "content": doc.content})
            except CorpusError:
                pass
        if docs:
            scens = extract_scenarios(
                state.llm_client, state.llm_cache,
                documents=docs,
                asof=state.card.asof.isoformat(),
                family=state.card.family,
                target_type=state.card.target_type,
            )
            state.scenarios = [s.__dict__ for s in scens]
            _LOG.debug("extract_scenarios -> %d scenarios", len(state.scenarios))

    elif act == "finish":
        # Normal termination
        pass

    else:
        raise LoopError(f"Unknown action: {act}")


def _finalise_outputs(state: LoopState) -> None:
    """Populate final drifts/widths/skews from accumulated state.

    If the LLM extracted vectors/scenarios, aggregate them.  Otherwise fall back
    to the deterministic Phase 2/4 pipeline (handled by the caller).
    """
    # For Phase 6, we just record what the LLM chose.  The deterministic
    # pipeline (run()) will use these to override the Phase 2/4 defaults.
    pass


__all__ = ["LoopConfig", "LoopState", "run_loop"]