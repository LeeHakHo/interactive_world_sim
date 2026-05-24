"""Shared pytest fixtures for the IWS test suite."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Make every test deterministic. Seeds are reset per-test."""
    seed = int(os.environ.get("IWS_TEST_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
