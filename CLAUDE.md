# CLAUDE.md — abinavi-kst

Pure-function adaptive-assessment math for Knowledge Space Theory (KST). A
line-by-line Python port of three `kstMatrix` (R) routines, shaped to contribute
upstream to `kstpy`. Consumed by the `abinavi-be` backend.

## Start here

- **`GLOSSARY.md`** — the KST vocabulary (knowledge state, surmise relation,
  BLIM, beta/eta, halfsplit, Bayesian update, threshold). Read before touching
  the math or naming anything in an issue/test.
- **`docs/adr/`** — the load-bearing decisions. Don't re-litigate these:
  - [0001](docs/adr/0001-pure-functions-only.md) pure functions, no session class
  - [0002](docs/adr/0002-frozen-non-mutating-inputs.md) inputs never mutated
  - [0003](docs/adr/0003-injected-rng.md) injected, seedable RNG
  - [0004](docs/adr/0004-numpy-at-the-boundary.md) int8 structures / float64 probs
  - [0005](docs/adr/0005-one-based-item-indices.md) 1-based public item indices
  - [0006](docs/adr/0006-gpl3-licensing.md) GPL-3 (copyleft, forced)
  - [0007](docs/adr/0007-shaped-for-kstpy-upstream.md) destined for kstpy
  - [0008](docs/adr/0008-r-equivalence-as-correctness-oracle.md) R-equivalence testing

## Layout

```
src/abinavi_kst/
  assessment.py   # the entire implementation
  __init__.py     # public API: AssessmentResult, bayesian_update,
                  #   halfsplit_question, simplified_assessment
tests/
  test_assessment.py           # property tests (sum-to-one, non-mutation, ties)
  test_r_equivalence_*.py       # assert parity with the R source
  r_reference/generate_fixtures.R  # the R oracle that produces fixtures
  conftest.py                   # fixture loading, the r_tol fixture (from meta.json)
```

`interactive_assessment` exists in `assessment.py` but is **not exported** — it
is a single-process test helper, not public API.

## Public API (the whole surface)

```python
from abinavi_kst import (
    halfsplit_question,    # -> next 1-based item to ask
    bayesian_update,       # -> new posterior over states (never mutates)
    simplified_assessment, # -> AssessmentResult (full non-interactive loop)
    AssessmentResult,
)
```

A caller needing one-question-per-request (the web backend) drives
`halfsplit_question` + `bayesian_update` itself and tracks `probs`/`queried`
across calls — there is no session object here (ADR-0001).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # Python-only tests + R-equivalence tests
```

Regenerating R fixtures (only if the R source changes) needs R + `kstMatrix`:

```bash
Rscript tests/r_reference/generate_fixtures.R   # writes tests/fixtures/ (git-ignored)
```

## When changing the math

The R source is the correctness oracle (ADR-0008). Any change to arithmetic
must keep the R-equivalence tests green; if R and Python differ only by
tie-breaking, compare the diagnosed `state`, not the full `probs` path.
