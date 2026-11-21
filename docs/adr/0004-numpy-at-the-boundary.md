# ADR-0004: NumPy at the boundary — int8 structures, float64 probabilities

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package's design rules)

## Context

The numerical core must be fast, memory-efficient for large knowledge
structures, and precise enough to match R within its float tolerance.

## Decision

Knowledge structures (`ks`) are represented as `np.ndarray` with dtype **`int8`**
by convention; probability distributions (`probs`) are **`float64`**. No Python
lists or dicts in the numerical core. The `int8` structure is cast to float64
only where arithmetic demands it (e.g. `ks.astype(np.float64)` in the `probs @
ks` product).

**Note**: the `int8` dtype is a *convention the package follows for its own
structures*, not an enforced precondition — `_validate_ks` checks shape and
binary content but not dtype, so a caller may pass any dtype holding binary
values. `float64` for `probs` *is* effectively required, because the
sum-to-one check and R-tolerance comparisons depend on it.

## Consequences

- **Benefit**: `int8` keeps large `N × n` structures compact; `float64` is
  required to hit R's `kmdoubleequal` tolerance (`_R_FLOAT_TOL`).
- **Cost**: callers must construct NumPy arrays; there is no convenience
  coercion from lists beyond `np.asarray(prior, dtype=float64)`.

## Evidence

`README.md` design rule 4; `src/abinavi_kst/assessment.py` (dtype handling,
`_validate_ks` validating shape/binary only); `tests/conftest.py` (fixtures
loaded with explicit `dtype=np.int8`).
