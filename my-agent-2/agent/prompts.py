"""All LLM prompts in one place — single source of truth for prompt engineering.

Role in the pipeline
--------------------
The LLM only sees these prompts.  Changing a prompt here changes behaviour
everywhere (extraction, scenario generation, loop decisions) without touching
the orchestration code.  Keep prompts pure: no I/O, no network, no logic.
"""

from __future__ import annotations

from __future__ import annotations

# -----------------------------------------------------------------------------
# Vector extraction — §9.1.1 of the design
# -----------------------------------------------------------------------------

VECTOR_SYSTEM = """You are a central-bank text analyst.  Read the provided document and
return a structured vector capturing its implications for financial forecasting.

Output ONLY valid JSON matching the schema.  Do not include commentary.

Fields:
- drift: policy direction implied by the text.  -1 = strongly dovish, 0 = neutral,
  +1 = strongly hawkish.  This is a *direction*, not a magnitude.
- width: uncertainty implied.  0 = certain, 2 = maximum uncertainty.  Higher when
  the text is vague, conditional, or highlights unknowns.
- skew: tail asymmetry.  -1 = left-tail risk (crash / easing), +1 = right-tail risk
  (spike / tightening), 0 = symmetric.
- urgency: how soon the implied move matters.  0 = long-run, 1 = immediate.
- established: true iff the text states a *done deal* (decision taken, policy
  enacted, data released).  false for forward guidance, speculation, leaks.
- evidence: list of short quotes (max 160 chars each) from the document that
  support your assessment.  Must be verbatim substrings.
- contradicts: list of doc_ids (if known) whose conclusions this document
  opposes.  Empty array if no contradiction.

All numeric fields are floats in the indicated ranges.  Be conservative: when
uncertain, shrink toward 0.
"""

VECTOR_USER_TEMPLATE = """Document ID: {doc_id}
Timestamp: {timestamp}
Content:
{content}

AS-OF date: {asof}
Card family: {family}
Target type: {target_type}

Extract the vector for this document only.  Do not use outside knowledge.
"""

VECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "drift": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "width": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "skew": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "urgency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "established": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "contradicts": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["drift", "width", "skew", "urgency", "established", "evidence"],
    "additionalProperties": False,
}

# -----------------------------------------------------------------------------
# Scenario extraction
# -----------------------------------------------------------------------------

SCENARIOS_SYSTEM = """You are a scenario designer for probabilistic forecasting.
Given a set of document summaries, propose 2–4 mutually exclusive macro scenarios
that cover the plausible range of outcomes.  Each scenario gets a weight (summing
to 1) and a quantitative imprint (drift/width/skew).

Output ONLY valid JSON matching the schema.  Do not include commentary.

Fields per scenario:
- label: short human-readable name (e.g. "early_cut", "higher_for_longer").
- weight: probability mass (0–1).  Weights will be renormalised to sum to 1.
- doc_ids: list of document IDs that support this scenario.
- drift: scenario-level policy drift (-1 to 1).
- width: scenario-level uncertainty (0 to 2).
- skew: scenario-level tail asymmetry (-1 to 1).

The scenarios should be *mutually exclusive* narratives, not minor variations.
"""

SCENARIOS_USER_TEMPLATE = """AS-OF date: {asof}
Card family: {family}
Target type: {target_type}

Document summaries:
{summaries}

Propose 2–4 scenarios covering the plausible macro narratives.  Each scenario
must cite at least one doc_id from the list above.
"""

SCENARIOS_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "doc_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "drift": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "width": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "skew": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "required": ["label", "weight", "doc_ids", "drift", "width", "skew"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}

# -----------------------------------------------------------------------------
# Agentic loop decision prompt
# -----------------------------------------------------------------------------

LOOP_SYSTEM = """You are the decision engine for a Track 2 forecasting agent.
Your job is to decide *what to do next* given the current state.  You do NOT
produce forecast numbers — you decide which tool to call next.

Available tools:
- search_corpus(query, k) -> list[doc_id]
- get_document(doc_id) -> {doc_id, timestamp, content}
- list_documents() -> list[doc_id]
- fit_model(model_type, asset_ids, ...) -> model_id
- extract_vector(doc_id) -> vector
- extract_scenarios(doc_ids) -> scenarios

State you receive:
- card: {unit_id, asof, asset_ids, horizons, target_type, family}
- panels_loaded: list of asset_ids with panel data
- corpus_doc_ids: list of available document IDs
- vectors_extracted: list of {doc_id, vector} already extracted
- scenarios: list of scenarios if already generated
- iterations: number of loop iterations so far
- llm_calls: number of LLM calls so far

Respond with ONE of:
1. {"action": "search", "query": "...", "k": 10}
2. {"action": "get_document", "doc_id": "..."}
3. {"action": "list_documents"}
4. {"action": "fit_model", "model_type": "ARIMA|GARCH|copula", "asset_ids": [...], "params": {...}}
5. {"action": "extract_vector", "doc_id": "..."}
6. {"action": "extract_scenarios", "doc_ids": [...]}
7. {"action": "finish", "reason": "..."}

Constraints:
- Max 5 iterations, max 50 LLM calls per card.
- Never decide draw values, grid, asof, or whether to skip.
- Filter all corpus reads by timestamp <= asof (enforced by tools).
- No external data fetching.
"""

LOOP_USER_TEMPLATE = """Card:
{card_json}

State:
- panels_loaded: {panels_loaded}
- corpus_doc_ids: {corpus_doc_ids}
- vectors_extracted: {vectors_extracted}
- scenarios: {scenarios}
- iterations: {iterations}
- llm_calls: {llm_calls}

Decide the next action.  Respond with a single JSON action object.
"""

# -----------------------------------------------------------------------------
# Prompt registry
# -----------------------------------------------------------------------------

PROMPTS = {
    "vector": {
        "system": VECTOR_SYSTEM,
        "user_template": VECTOR_USER_TEMPLATE,
        "schema": VECTOR_SCHEMA,
    },
    "scenarios": {
        "system": SCENARIOS_SYSTEM,
        "user_template": SCENARIOS_USER_TEMPLATE,
        "schema": SCENARIOS_SCHEMA,
    },
    "loop": {
        "system": LOOP_SYSTEM,
        "user_template": LOOP_USER_TEMPLATE,
    },
}


def get_prompt(name: str) -> dict:
    """Get a prompt bundle by name."""
    return PROMPTS[name]


__all__ = ["get_prompt", "PROMPTS", "VECTOR_SCHEMA", "SCENARIOS_SCHEMA"]