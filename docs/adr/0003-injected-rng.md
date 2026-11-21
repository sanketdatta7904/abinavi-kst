# ADR-0003: Injected RNG (seedable randomness)

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package's design rules)

## Context

Halfsplit selection breaks ties at random. For the R-equivalence tests (see
[ADR-0008](0008-r-equivalence-as-correctness-oracle.md)) and for any
reproducible run, that randomness must be controllable.

## Decision

`halfsplit_question` and `simplified_assessment` accept an explicit
`numpy.random.Generator` (`rng`). No routine calls a global `np.random.*`
function. `simplified_assessment` creates `np.random.default_rng()` as a
convenience default when `rng=None`.

## Consequences

- **Benefit**: fully deterministic tests without `monkeypatch`; production can
  pass a per-request generator.
- **Cost**: callers must supply or accept a default `rng`.
- **Caveat**: R (`sample()`) and Python (`rng.choice()`) resolve ties via
  different algorithms, so a seed does not make Python match R on tie-dependent
  paths. This shapes the testing strategy — see
  [ADR-0008](0008-r-equivalence-as-correctness-oracle.md).

## Evidence

`README.md` design rule 3; `src/abinavi_kst/assessment.py` (`rng.choice` in
halfsplit; `rng` parameters). Every R-equivalence test pins
`rng=np.random.default_rng(seed=...)`.
