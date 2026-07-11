"""Assert simplified_assessment mirrors R's kmsassess result structure."""
from __future__ import annotations

import numpy as np
import pytest

from abinavi_kst import simplified_assessment
from conftest import load_fixture


def _probs_multiset(p):
    return np.sort(np.asarray(p, dtype=np.float64))


@pytest.mark.parametrize("fixture", ["sassess_S1_allcorrect", "sassess_S1_allwrong"])
def test_sassess_state_matches_r(fixture, xpl_space, r_tol):
    d = load_fixture(fixture)
    r = np.array(d["inputs"]["r"], dtype=np.int8)
    res = simplified_assessment(
        r, xpl_space, beta=0.1, eta=0.1, threshold=0.55,
        rng=np.random.default_rng(0),
    )
    # state exact:
    assert list(res.state) == list(d["state"])
    # probs as an order-invariant multiset (row order may differ):
    final = res.probs if not isinstance(res.probs, list) else res.probs[-1]
    np.testing.assert_allclose(
        _probs_multiset(final), _probs_multiset(d["probs"]), atol=r_tol, rtol=0
    )


def test_sassess_result_has_r_field_names():
    """FULL MIRROR: R returns list(state, probs, queried, qtime, utime).
    Document whether Python exposes the same field names."""
    d_missing = []
    from abinavi_kst import AssessmentResult
    for field in ("state", "probs", "queried", "qtime", "utime"):
        if not hasattr(AssessmentResult, "__dataclass_fields__") or \
           field not in AssessmentResult.__dataclass_fields__:
            d_missing.append(field)
    assert d_missing == [], f"AssessmentResult missing R fields: {d_missing}"


def test_sassess_threshold_low_warns_not_raises(xpl_space):
    """R warns (does not stop) when threshold<=0.5 (fixture error=None)."""
    d = load_fixture("sassess_warn_threshold")
    assert d["error"] is None
    r = np.array([1, 1, 1, 1], dtype=np.int8)
    with pytest.warns(UserWarning):
        simplified_assessment(r, xpl_space, threshold=0.4, rng=np.random.default_rng(0))
