"""Assert bayesian_update reproduces R's kmassessbayesian, per fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from abinavi_kst import bayesian_update
from conftest import load_fixture


@pytest.mark.parametrize(
    "fixture",
    ["bayesian_B1_resp1", "bayesian_B2_resp0", "bayesian_B3_vecparams",
     "bayesian_B4_uniformprior",
     "bayesian_B5_q1", "bayesian_B5_q2", "bayesian_B5_q3", "bayesian_B5_q4"],
)
def test_bayesian_matches_r(fixture, xpl_space, r_tol):
    d = load_fixture(fixture)
    i = d["inputs"]
    probs = np.array(i["probs"], dtype=np.float64)
    beta = np.array(i["beta"], dtype=np.float64)
    eta = np.array(i["eta"], dtype=np.float64)
    got = bayesian_update(probs, xpl_space, beta, eta, i["question"], i["response"])
    expected = np.array(d["r_output"], dtype=np.float64)
    np.testing.assert_allclose(got, expected, atol=r_tol, rtol=0)


def test_bayesian_beta_eta_warns_not_raises(xpl_space):
    """R only WARNS when beta+eta>1 and still returns a value (fixture
    bayesian_warn_beta_eta has error=None). Python must mirror: warn, continue."""
    d = load_fixture("bayesian_warn_beta_eta")
    assert d["error"] is None  # R did not stop
    probs = np.array([0.02,0.1,0.07,0.01,0.4,0.17,0.07,0.08,0.08])
    with pytest.warns(UserWarning):
        out = bayesian_update(probs, xpl_space, np.full(4,0.6), np.full(4,0.6), 1, 1)
    assert out is not None
