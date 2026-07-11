"""Assert halfsplit_question selects from R's candidate set.

R's candidate set is the set of items tied at the minimum distance-to-0.5,
computed by kmassesshalfsplit. When that set is a singleton the pick is
deterministic and must equal R's item exactly. When it has >1 member the pick
is random (R uses sample(); we inject an RNG) and every pick must land inside
R's candidate set.

Fixtures are self-describing (`candidates`, `deterministic`), so these tests
adapt to whichever prior happens to tie — we don't hardcode that assumption.
"""
from __future__ import annotations

import numpy as np
import pytest

from abinavi_kst import halfsplit_question
from conftest import load_fixture

_ALL_HS = ["halfsplit_H3_example", "halfsplit_H1_unique", "halfsplit_H2_tie"]


def _candidate_set(d: dict) -> set[int]:
    c = d["candidates"]
    return set(c) if isinstance(c, list) else {c}


@pytest.mark.parametrize("fixture", _ALL_HS)
def test_halfsplit_deterministic_matches_r(fixture, xpl_space):
    """When R's candidate set is a singleton, Python must pick that exact
    item across every seed."""
    d = load_fixture(fixture)
    if not d["deterministic"]:
        pytest.skip(f"{fixture} is a tie; covered by the candidate-set test")
    (r_item,) = _candidate_set(d)
    probs = np.array(d["inputs"]["probs"], dtype=np.float64)
    for seed in range(20):
        got = halfsplit_question(probs, xpl_space, np.random.default_rng(seed))
        assert got == r_item


@pytest.mark.parametrize("fixture", _ALL_HS)
def test_halfsplit_tie_stays_in_candidate_set(fixture, xpl_space):
    """On a tie, every Python pick must be one of R's candidates, and the RNG
    must actually be consulted (>=2 distinct picks)."""
    d = load_fixture(fixture)
    if d["deterministic"]:
        pytest.skip(f"{fixture} is deterministic; covered by the exact-match test")
    cands = _candidate_set(d)
    probs = np.array(d["inputs"]["probs"], dtype=np.float64)
    picks = {
        halfsplit_question(probs, xpl_space, np.random.default_rng(s))
        for s in range(200)
    }
    assert picks.issubset(cands)
    assert len(picks) >= 2  # RNG actually consulted


@pytest.mark.parametrize("fixture", _ALL_HS)
def test_halfsplit_item_prob_matches_r(fixture, xpl_space, r_tol):
    """The per-item mastery probabilities Python computes must equal R's."""
    d = load_fixture(fixture)
    probs = np.array(d["inputs"]["probs"], dtype=np.float64)
    py_ip = probs @ xpl_space.astype(np.float64)
    np.testing.assert_allclose(py_ip, np.array(d["item_prob"]), atol=r_tol, rtol=0)
