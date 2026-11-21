# ADR-0002: Frozen / non-mutating inputs

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package's design rules)

## Context

Callers (especially a web backend that checkpoints state between requests, and
the `probdev` history feature) need to hold onto probability snapshots and trust
they won't change underfoot.

## Decision

Functions **never mutate their arguments**. `bayesian_update` returns a new
array; `simplified_assessment` copies `prior` on entry before iterating.

## Consequences

- **Benefit**: callers can retain `probs` snapshots safely; no mutation-at-a-
  distance bugs; `probdev` history is just a list of the returned arrays.
- **Cost**: one extra array allocation per update step. Negligible at the state
  counts this package targets.

## Evidence

`README.md` design rule 2; `src/abinavi_kst/assessment.py` (`bayesian_update`
creates a new `posterior`; `simplified_assessment` copies `prior`);
`tests/test_assessment.py::test_bayesian_update_does_not_mutate_input`.
