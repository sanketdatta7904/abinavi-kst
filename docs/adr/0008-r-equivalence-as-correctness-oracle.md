# ADR-0008: R-equivalence as the correctness oracle (testing strategy)

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the test suite)

## Context

The routines are translations of `kstMatrix`. "Correct" is best defined as
"produces the same numbers as the R source," not "matches an independently
derived formula" — the latter would risk re-deriving a different algorithm.

## Decision

Correctness is defined as **numerical agreement with `kstMatrix`**. Expected
values are never hand-derived; instead `tests/r_reference/generate_fixtures.R`
runs the real R package, serialises outputs to JSON fixtures, and the Python
tests assert agreement within `_R_FLOAT_TOL` (`sqrt(float64 eps)`).

## Consequences

- **Benefit**: any arithmetic drift from the R source is caught immediately.
- **Cost**: regenerating fixtures needs an R + `kstMatrix` environment
  (non-trivial CI). The JSON fixtures are git-ignored (`tests/fixtures/`) and must
  be regenerated with the `Rscript` command noted in `.gitignore`.
- **Tie caveat** (important): because R (`sample()`) and Python (`rng.choice()`)
  break halfsplit ties differently, end-to-end `probs` comparison on
  tie-dependent paths is path-dependent and **not meaningful**. The suite
  therefore compares only the diagnosed `state` end-to-end when ties occur, and
  isolates the update arithmetic via forced-path replays. See
  `tests/test_r_equivalence_caveats.py`. This caveat interacts with
  [ADR-0003](0003-injected-rng.md).

## Evidence

`tests/r_reference/generate_fixtures.R` (the oracle); `tests/conftest.py`
(fixture loading, the `r_tol` fixture from `meta.json`); `_R_FLOAT_TOL` is
defined in `src/abinavi_kst/assessment.py`; `tests/test_r_equivalence_*.py`.
