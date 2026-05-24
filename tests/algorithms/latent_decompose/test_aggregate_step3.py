"""Step 3 metric aggregation + Go/No-Go gate."""
from __future__ import annotations

import pytest

from interactive_world_sim.algorithms.latent_decompose.diagnostics.aggregate_step3 import (
    Decision,
    decide,
    summarise,
)


def test_summarise_mean_std():
    rows = [
        {"config": "1.5R_0H", "seed": 0, "fvd": 100.0},
        {"config": "1.5R_0H", "seed": 1, "fvd": 110.0},
        {"config": "1.5R_0H", "seed": 2, "fvd": 120.0},
    ]
    s = summarise(rows, metric="fvd")
    assert s["1.5R_0H"]["mean"] == pytest.approx(110.0)
    assert s["1.5R_0H"]["std"]  == pytest.approx(8.16496580927726)


def test_decide_paper_strong():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 110.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.gap == pytest.approx(0.10)
    assert d.outcome == Decision.STRONG


def test_decide_paper_weak():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 130.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.WEAK


def test_decide_diagnose():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 200.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.DIAGNOSE


def test_decide_aborts_when_sanity_fails():
    """0R_1.0H must be strictly worse than 1.5R_0H."""
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 105.0},
         "0R_1.0H": {"mean":  90.0}}                  # human-only BETTER → abort
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.ABORT
