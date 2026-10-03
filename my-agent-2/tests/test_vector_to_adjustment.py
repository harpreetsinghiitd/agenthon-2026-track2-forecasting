"""Tests for `harness.bridge.vector_to_adjustment` — deterministic bridge."""

from __future__ import annotations

import pytest

from harness.bridge.vector_to_adjustment import (
    Adjustment,
    BridgeError,
    aggregate_vectors,
    vector_to_adjustment,
)
from harness.llm.extract import Vector


def _vec(**kw) -> Vector:
    defaults = dict(drift=0.0, width=1.0, skew=0.0, urgency=0.5, established=False, evidence=("e",), contradicts=())
    defaults.update(kw)
    return Vector(**defaults)


def test_drift_capped_at_015_horizon_sd() -> None:
    v = _vec(drift=1.0, urgency=1.0, established=True)
    adj = vector_to_adjustment(v, horizon_sd=0.10)
    assert adj.drift == pytest.approx(0.015)  # 0.15 * 0.10


def test_drift_sign_preserved() -> None:
    v = _vec(drift=-0.5, urgency=1.0, established=True)
    adj = vector_to_adjustment(v, horizon_sd=0.10)
    assert adj.drift < 0


def test_urgency_scales_drift() -> None:
    v1 = _vec(drift=1.0, urgency=0.0, established=True)
    v2 = _vec(drift=1.0, urgency=1.0, established=True)
    adj1 = vector_to_adjustment(v1, horizon_sd=0.10)
    adj2 = vector_to_adjustment(v2, horizon_sd=0.10)
    assert abs(adj2.drift) > abs(adj1.drift)


def test_width_narrowing_only_if_established() -> None:
    v_est = _vec(width=0.5, established=True)
    v_inf = _vec(width=0.5, established=False)
    adj_est = vector_to_adjustment(v_est, horizon_sd=0.10)
    adj_inf = vector_to_adjustment(v_inf, horizon_sd=0.10)
    assert adj_est.width_mult == 0.5
    assert adj_inf.width_mult == 1.0  # inferred cannot narrow


def test_width_widening_always_allowed() -> None:
    v = _vec(width=1.5, established=False)
    adj = vector_to_adjustment(v, horizon_sd=0.10)
    assert adj.width_mult == 1.5


def test_skew_passed_through() -> None:
    v = _vec(skew=0.7)
    adj = vector_to_adjustment(v, horizon_sd=0.10)
    assert adj.skew == 0.7


def test_horizon_sd_must_be_positive() -> None:
    v = _vec()
    with pytest.raises(BridgeError, match="positive"):
        vector_to_adjustment(v, horizon_sd=0.0)
    with pytest.raises(BridgeError, match="positive"):
        vector_to_adjustment(v, horizon_sd=-0.1)


def test_aggregate_vectors_weighted_mean() -> None:
    v1 = _vec(drift=1.0, width=1.0, skew=0.0, urgency=1.0, established=True)
    v2 = _vec(drift=-1.0, width=1.0, skew=0.0, urgency=1.0, established=True)
    agg = aggregate_vectors([v1, v2], weights=[1.0, 1.0])
    assert agg.drift == pytest.approx(0.0)


def test_aggregate_vectors_default_weights_urgency_x_established() -> None:
    v1 = _vec(drift=1.0, urgency=1.0, established=True)   # weight = 2.0
    v2 = _vec(drift=0.0, urgency=0.5, established=False)  # weight = 0.5
    agg = aggregate_vectors([v1, v2])
    # weighted mean: (1.0*2 + 0*0.5) / 2.5 = 0.8
    assert agg.drift == pytest.approx(0.8)


def test_aggregate_vectors_established_all_true() -> None:
    v1 = _vec(established=True)
    v2 = _vec(established=True)
    agg = aggregate_vectors([v1, v2])
    assert agg.established is True


def test_aggregate_vectors_established_any_false() -> None:
    v1 = _vec(established=True)
    v2 = _vec(established=False)
    agg = aggregate_vectors([v1, v2])
    assert agg.established is False


def test_aggregate_vectors_evidence_deduped() -> None:
    v1 = _vec(evidence=("a", "b"))
    v2 = _vec(evidence=("b", "c"))
    agg = aggregate_vectors([v1, v2])
    assert agg.evidence == ("a", "b", "c")


def test_aggregate_vectors_empty_raises() -> None:
    with pytest.raises(BridgeError, match="empty"):
        aggregate_vectors([])


def test_aggregate_vectors_weights_mismatch_raises() -> None:
    with pytest.raises(BridgeError, match="length"):
        aggregate_vectors([_vec(), _vec()], weights=[1.0])