"""Shared fixtures: load R reference JSON from tests/fixtures/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / f"{name}.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def r_meta() -> dict:
    return load_fixture("meta")


@pytest.fixture(scope="session")
def xpl_space(r_meta) -> np.ndarray:
    return np.array(r_meta["xpl_space"], dtype=np.int8)


@pytest.fixture(scope="session")
def r_tol(r_meta) -> float:
    return float(r_meta["double_eps_tol"])
