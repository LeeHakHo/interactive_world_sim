"""Linear-ramp schedule for lambda_adv."""
from __future__ import annotations

import pytest

from interactive_world_sim.algorithms.latent_decompose.align_losses import (
    linear_ramp,
)


@pytest.mark.parametrize(
    "step, expected",
    [
        (0,     0.0),
        (1999,  0.0),
        (2000,  0.0),
        (6000,  0.15),    # midpoint of [2000, 10000] @ [0.0, 0.3]
        (10000, 0.3),
        (50000, 0.3),
    ],
)
def test_linear_ramp_values(step, expected):
    out = linear_ramp(
        step,
        start_step=2000,
        end_step=10000,
        start_value=0.0,
        end_value=0.3,
    )
    assert abs(out - expected) < 1e-9
