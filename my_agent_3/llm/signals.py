"""Executive summary: placeholder schema for future structured LLM signals."""

NEUTRAL = {
    "mean_shift_sigma": 0.0,
    "variance_scale": 1.0,
    "tail_direction": "none",
    "tail_strength": 0.0,
    "correlation_scale": 1.0,
    "shock_probability": 0.0,
    "evidence": "The LLM layer is a placeholder; no text adjustment was applied.",
}


def parse_signals(raw: str) -> dict:
    """Validate and bound a future model response."""
    raise NotImplementedError("LLM signal parsing has not been implemented")


def extract_signals(documents: list[dict], asof: str, card: dict) -> dict:
    """Return neutral signals until endpoint integration is implemented."""
    return dict(NEUTRAL)
