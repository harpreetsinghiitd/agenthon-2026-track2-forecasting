"""Text reasoning layer.

Currently a stub. Returns None so the combiner passes draws through unchanged.

When implemented, extract_signals() should:
  1. Filter documents to those with timestamp <= asof.
  2. Call the House model at MODEL_ENDPOINT with a prompt for structured signals.
  3. Return a dict, e.g.:
        {"policy_stance": "hawkish",
         "tail_direction": "left",
         "uncertainty": "high",
         "evidence": "FOMC statement 2024-05-01 turned more hawkish ..."}
"""


def extract_signals(documents: list[dict], asof: str, card: dict) -> dict | None:
    # TODO: implement text reasoning.
    return None