# ADR-0001: Pure functions only — no classes for the math, no globals

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package's design rules)

## Context

The package provides adaptive-assessment math (halfsplit selection, Bayesian
update, the assessment loop). A web backend consumes it across HTTP requests,
and a test suite must pin behaviour deterministically.

## Decision

All routines are **module-level pure functions** (`halfsplit_question`,
`bayesian_update`, `simplified_assessment`). No `Assessor` or `Session` class
wraps mutable state; there are no module globals.

## Consequences

- **Benefit**: every function is independently testable, composable, and
  serialisable without pickling class instances.
- **Cost**: a caller that wants one-question-per-request (a web backend) must
  track `probs` and `queried` itself across requests — the package offers no
  stateful session object. `simplified_assessment` is the reference loop to copy.
- This is why the consuming backend (`abinavi-be`) wraps these primitives in its
  own `KstpyAdaptiveAssessor` at an `AdaptiveAssessor` port rather than using a
  session object from here.

## Evidence

`README.md` design rule 1; `src/abinavi_kst/assessment.py` module docstring.
