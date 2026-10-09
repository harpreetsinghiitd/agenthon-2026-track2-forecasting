"""Executive summary: prompts that elicit bounded, evidence-cited corpus signals.

The witness vocabulary is fixed here and validated in ``signals.py``: the model
reports directions and intensities only, never values in target units, and
every non-neutral field must cite named documents. The established/inferred
distinction and the widen-don't-average rule come from the solver playbook
(docs/SOLVER-PLAYBOOK.md §3).
"""

from __future__ import annotations

MAX_DOC_CHARS = 4000
MAX_DOCUMENTS = 12

SYSTEM_PROMPT = """You are a central-bank corpus analyst. You read ONLY the documents provided, all dated on or before the as-of date. You never forecast values, never use outside knowledge, and never mention outcomes you may remember about this period.

Return ONLY a JSON object with exactly these fields:
{
  "mean_shift_sigma": number in [-0.5, 0.5],
  "variance_scale": number in [0.75, 1.75],
  "tail_direction": "left" | "right" | "none",
  "tail_strength": number in [0.0, 1.0],
  "correlation_scale": number in [1.0, 1.5],
  "shock_probability": number in [0.0, 1.0],
  "basis": "established" | "inferred",
  "evidence": "one short paragraph citing document names and dates"
}

Field meanings:
- mean_shift_sigma: expected drift of the target versus a random walk over the card's horizons, in standard deviations. Positive = upward. Be modest; when unsure return 0.
- variance_scale: uncertainty multiplier versus what recent data implies. Above 1 widens, below 1 narrows.
- tail_direction / tail_strength: which tail the corpus gives extra weight to, and how much.
- correlation_scale: cross-asset co-movement multiplier; 1.0 when the card has one asset or the corpus says nothing.
- shock_probability: probability the corpus assigns to a discrete event (vote, decision, peg test) inside the horizon.
- basis: "established" ONLY when documents state decisions taken, data released, or levels announced as done deals; tone, guidance and speculation are "inferred".
- evidence: cite the document name and date behind every non-neutral field.

Hard rules:
- Inferred signal may move the centre but must NOT narrow: if basis is "inferred", variance_scale must be >= 1.0.
- When documents conflict, do NOT average their tones: raise variance_scale and state which document takes which side.
- Output the JSON object only. No markdown, no commentary.
"""


def build_messages(documents: list[dict], asof: str, card: dict) -> list[dict]:
    """Return [system, user] messages carrying the corpus and card context."""
    recent = sorted(
        documents, key=lambda d: str(d.get("timestamp", "")), reverse=True
    )[:MAX_DOCUMENTS]
    corpus = "\n\n".join(
        f"### {doc['name']} (dated {doc['timestamp']})\n"
        f"{str(doc['text'])[:MAX_DOC_CHARS]}"
        for doc in sorted(recent, key=lambda d: str(d.get("timestamp", "")))
    )
    targets = card.get("targets", {})
    user = (
        f"As-of date: {asof}\n"
        f"Card family: {card.get('metadata', {}).get('category', '')}\n"
        f"Target type: {targets.get('target_type', '')}; "
        f"assets: {list(targets.get('asset_ids', []))}; "
        f"horizons in business days: {list(targets.get('horizons', []))}\n\n"
        f"Documents (all dated on or before the as-of date):\n\n{corpus}\n\n"
        "Assess the corpus and return the JSON signal object only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
