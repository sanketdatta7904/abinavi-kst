"""Tests for the adaptive assessment functions.

Two layers:
1. **Property tests** that don't depend on R reference output —
   sum-to-1, halfsplit-picks-near-half, monotonic narrowing, edge cases.
2. **Golden trace** from the algorithm developer's email, asserting our
   Python implementation reproduces the exact probability sequence R
   produces for the canonical 5-item structure in ``Basic_Slides_SR``.

Run with ``pytest`` from the package root.
"""

from __future__ import annotations

import numpy as np
import pytest

from abinavi_kst import (
    bayesian_update,
    halfsplit_question,
    simplified_assessment,
)


# ---------------------------------------------------------------------------
# Canonical 5-item structure from Basic_Slides_SR.xlsx
# ---------------------------------------------------------------------------

# The 9 knowledge states (rows) over items (a, b, c, d, e):
BASIC_SLIDES_KS = np.array(
    [
        [0, 0, 0, 0, 0],  # 1: empty
        [0, 0, 0, 0, 1],  # 2: {e}
        [0, 0, 1, 0, 1],  # 3: {c, e}
        [0, 0, 0, 1, 1],  # 4: {d, e}
        [0, 1, 1, 0, 1],  # 5: {b, c, e}
        [0, 0, 1, 1, 1],  # 6: {c, d, e}
        [1, 1, 1, 0, 1],  # 7: {a, b, c, e}
        [0, 1, 1, 1, 1],  # 8: {b, c, d, e}
        [1, 1, 1, 1, 1],  # 9: {a, b, c, d, e} — full mastery
    ],
    dtype=np.int8,
)


# ---------------------------------------------------------------------------
# 1. Property tests for halfsplit_question
# ---------------------------------------------------------------------------


def test_halfsplit_picks_item_closest_to_half_under_uniform_prior():
    """Under a uniform prior, the half-split item is the one whose
    column mean is closest to 0.5."""
    ks = BASIC_SLIDES_KS
    N = ks.shape[0]
    probs = np.full(N, 1.0 / N)
    rng = np.random.default_rng(seed=0)

    # Item probabilities under uniform prior = column means.
    item_probs = ks.mean(axis=0)
    expected_candidates = np.flatnonzero(
        np.abs(item_probs - 0.5) == np.abs(item_probs - 0.5).min()
    ) + 1

    picked = halfsplit_question(probs, ks, rng)
    assert picked in expected_candidates


def test_halfsplit_breaks_ties_uniformly_at_random():
    """When multiple items tie on distance-to-0.5, picks are not
    deterministic across different seeds."""
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])

    seen = set()
    for seed in range(50):
        seen.add(halfsplit_question(probs, ks, np.random.default_rng(seed)))
    # Under a uniform prior, multiple items tie — we expect ≥2 distinct
    # picks across 50 seeds. (Strict equality of tie-breaks across seeds
    # would mean the RNG isn't being consulted.)
    assert len(seen) >= 2


def test_halfsplit_rejects_non_binary_ks():
    rng = np.random.default_rng(seed=0)
    ks = np.array([[0, 0.5], [1, 1]])
    probs = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="binary"):
        halfsplit_question(probs, ks, rng)


def test_halfsplit_rejects_probs_that_dont_sum_to_one():
    rng = np.random.default_rng(seed=0)
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 0.5)  # sums to 4.5
    with pytest.raises(ValueError, match="sum to 1"):
        halfsplit_question(probs, ks, rng)


# ---------------------------------------------------------------------------
# 2. Property tests for bayesian_update
# ---------------------------------------------------------------------------


def test_bayesian_update_preserves_sum_to_one():
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    for response in (0, 1):
        for item in range(1, ks.shape[1] + 1):
            updated = bayesian_update(probs, ks, 0.1, 0.1, item, response)
            assert updated.sum() == pytest.approx(1.0, abs=1e-12)


def test_bayesian_update_does_not_mutate_input():
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    before = probs.copy()
    _ = bayesian_update(probs, ks, 0.1, 0.1, 1, 1)
    np.testing.assert_array_equal(probs, before)


def test_correct_response_raises_prob_of_states_containing_the_item():
    """If beta < eta (i.e. correctness is informative), a correct
    response should raise the total probability mass of states that
    contain the item."""
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    item = 3  # 1-based: item c. Item is in states 3, 5, 6, 7, 8, 9.

    mass_before = probs @ ks[:, item - 1]  # 6/9
    updated = bayesian_update(probs, ks, beta=0.1, eta=0.1, item=item, response=1)
    mass_after = updated @ ks[:, item - 1]
    assert mass_after > mass_before


def test_wrong_response_lowers_prob_of_states_containing_the_item():
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    item = 3
    mass_before = probs @ ks[:, item - 1]
    updated = bayesian_update(probs, ks, beta=0.1, eta=0.1, item=item, response=0)
    mass_after = updated @ ks[:, item - 1]
    assert mass_after < mass_before


def test_bayesian_update_rejects_bad_response():
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    with pytest.raises(ValueError, match="response"):
        bayesian_update(probs, ks, 0.1, 0.1, 1, 2)


def test_bayesian_update_accepts_vector_beta_and_eta():
    """`kmassess` (per-item) signature: beta/eta as length-n vectors."""
    ks = BASIC_SLIDES_KS
    probs = np.full(ks.shape[0], 1.0 / ks.shape[0])
    beta = np.full(ks.shape[1], 0.1)
    eta = np.full(ks.shape[1], 0.1)
    updated = bayesian_update(probs, ks, beta, eta, item=1, response=1)
    assert updated.sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Property tests for simplified_assessment
# ---------------------------------------------------------------------------


def test_assessment_terminates_when_threshold_crossed():
    ks = BASIC_SLIDES_KS
    # Student has full mastery; expect threshold crossed reasonably fast.
    full_mastery = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    result = simplified_assessment(
        full_mastery, ks, rng=np.random.default_rng(seed=42)
    )
    assert result.terminated_early is False
    final = result.probs
    assert final.max() > 0.51
    assert len(result.queried) <= 2 * ks.shape[1]


def test_assessment_diagnoses_full_mastery_correctly():
    ks = BASIC_SLIDES_KS
    full_mastery = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    result = simplified_assessment(
        full_mastery, ks, rng=np.random.default_rng(seed=42)
    )
    assert result.state == (1, 1, 1, 1, 1)


def test_assessment_diagnoses_empty_state_correctly():
    ks = BASIC_SLIDES_KS
    none = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    result = simplified_assessment(none, ks, rng=np.random.default_rng(seed=42))
    assert result.state == (0, 0, 0, 0, 0)


def test_probdev_returns_full_history():
    ks = BASIC_SLIDES_KS
    full_mastery = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    result = simplified_assessment(
        full_mastery, ks, rng=np.random.default_rng(seed=42), probdev=True
    )
    assert isinstance(result.probs, list)
    # prior + one update per queried item
    assert len(result.probs) == len(result.queried) + 1
    # First entry is the uniform prior.
    np.testing.assert_allclose(result.probs[0], np.full(9, 1.0 / 9))


# ---------------------------------------------------------------------------
# 4. Golden trace from the algorithm developer's email
#
# The email gave this exact trace for Basic_Slides_SR.xlsx with the
# student's response pattern that produced queried=[4,2,3,5] and
# final state=[1,1,1,1,1]. We assert byte-equivalent reproduction
# within R's kmdoubleequal tolerance (sqrt(eps) ~= 1.49e-8).
# ---------------------------------------------------------------------------


GOLDEN_QUERIED = (4, 2, 3, 5)
GOLDEN_PROBS = [
    np.full(9, 1.0 / 9),
    np.array(
        [
            0.02040816, 0.02040816, 0.02040816, 0.02040816,
            0.18367347, 0.18367347, 0.18367347, 0.18367347, 0.18367347,
        ]
    ),
    np.array(
        [
            0.003558719, 0.003558719, 0.032028470, 0.032028470,
            0.032028470, 0.032028470,
            0.288256228, 0.288256228, 0.288256228,
        ]
    ),
    np.array(
        [
            0.0006064281, 0.0006064281, 0.0054578532, 0.0491206792,
            0.0054578532, 0.0054578532,
            0.0491206792, 0.4420861128, 0.4420861128,
        ]
    ),
    np.array(
        [
            0.000133672, 0.000133672, 0.001203048, 0.010827429,
            0.001203048, 0.001203048,
            0.010827429, 0.097446865, 0.877021789,
        ]
    ),
]
GOLDEN_FINAL_STATE = (1, 1, 1, 1, 1)


def _replay_with_seed(seed: int):
    """Run the assessment with a full-mastery student and a fixed seed."""
    full = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    return simplified_assessment(
        full, BASIC_SLIDES_KS, rng=np.random.default_rng(seed), probdev=True
    )


def test_golden_trace_state_matches_email():
    """For any seed where halfsplit ties don't matter, the final state
    is the same full-mastery vector the email reports."""
    result = _replay_with_seed(seed=0)
    assert result.state == GOLDEN_FINAL_STATE


@pytest.mark.xfail(
    reason=(
        "The email's trace places 5 states above and 4 states below after "
        "answering item 4, but no item in the union-closure of "
        "Basic_Slides_SR.xlsx appears in exactly 5 states (frequencies "
        "are a=2, b=4, c=6, d=4, e=8). This indicates the email's example "
        "was generated from a *different* surmise relation than the one "
        "in the Excel file we received. Need to ask the algo team for "
        "the exact SR that produced the email's trace before this can "
        "be a proper golden test."
    ),
    strict=True,
)
def test_golden_probs_match_email_when_query_sequence_is_forced():
    """Multiset-equality check on bayesian_update math against R's reference.

    The R reference's column/row ordering of the knowledge structure
    may differ from ours, but the *probability multiset* — the bag of
    unique values and how many states each value applies to — is
    invariant under relabeling. Marked xfail because the email's
    numbers and the SR file don't refer to the same structure (see
    @xfail message above).
    """
    R_TOL = float(np.sqrt(np.finfo(np.float64).eps))

    ks = BASIC_SLIDES_KS
    probs = np.full(9, 1.0 / 9)
    np.testing.assert_allclose(
        np.sort(probs), np.sort(GOLDEN_PROBS[0]), atol=R_TOL
    )

    for step, item in enumerate(GOLDEN_QUERIED, start=1):
        probs = bayesian_update(probs, ks, beta=0.1, eta=0.1, item=item, response=1)
        np.testing.assert_allclose(
            np.sort(probs), np.sort(GOLDEN_PROBS[step]), atol=R_TOL,
            err_msg=(
                f"probs (as sorted multiset) after item {item} (step {step}) "
                f"diverged from R's reference"
            ),
        )


def test_halfsplit_can_pick_either_of_two_tied_items():
    """Confirms halfsplit's tie-break is RNG-driven.

    Under the uniform prior on BASIC_SLIDES_KS, items 2 (b) and 4 (d)
    tie on distance to 0.5 — both have item-mastery probability 4/9.
    Across many seeds we expect both to appear. (R's run happened to
    pick 4 first; our implementation can pick either, depending on the
    seed — and that's correct behavior.)
    """
    probs = np.full(9, 1.0 / 9)
    picks = {
        halfsplit_question(probs, BASIC_SLIDES_KS, np.random.default_rng(seed))
        for seed in range(200)
    }
    assert {2, 4}.issubset(picks), (
        f"halfsplit should pick both tied items 2 and 4 across 200 seeds, "
        f"got picks={picks}"
    )
