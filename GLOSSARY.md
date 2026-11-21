# Glossary — abinavi-kst

The domain vocabulary of this package: adaptive assessment under **Knowledge
Space Theory (KST)**. Terms are named as the code names them. When a document,
test name, or issue title refers to one of these concepts, use the term defined
here rather than a synonym.

This package is a line-by-line Python port of three routines from the R package
[`kstMatrix`](https://cran.r-project.org/package=kstMatrix) (`kmsassess`,
`kmassesshalfsplit`, `kmassessbayesian`). Where a term maps to an R name, that
mapping is noted, because the R source is the correctness oracle (see
[ADR-0008](docs/adr/0008-r-equivalence-as-correctness-oracle.md)).

---

## Knowledge Space Theory

### Item
One assessable question or skill. Identified publicly by a **1-based** integer
and internally by its **0-based** column position in the `ks` matrix. Items are
the columns of `ks`. See [item indices](#1-based-vs-0-based-item-indices).

### Knowledge state
A subset of items a student has mastered, represented as one **row** of the `ks`
matrix: a binary vector of length `n` (number of items), where `1` at position
`j` means item `j+1` is mastered.

### Knowledge structure / knowledge space (`ks`)
The full set of all possible knowledge states, encoded as an `N × n` NumPy array
of binary values (`N` states × `n` items). The package's own structures use
dtype `int8` (see [ADR-0004](docs/adr/0004-numpy-at-the-boundary.md)), but this
is a convention, **not enforced** — `_validate_ks` checks shape and binary
content only, not dtype. Each row is one knowledge state. A *knowledge space* is
the special case where the family of states is closed under union. This package
**consumes** an already-closed `ks` matrix; it does not construct or validate the
structure itself (see [surmise relation](#surmise-relation)). Referred to
throughout the code and the R source as `ks`.

### Surmise relation
A partial order on items expressing prerequisites: if item A surmises item B,
you cannot know B without knowing A. The surmise relation is what determines
which knowledge states are possible. In the R ecosystem a `kmbasis` encodes it
and `kmspace()` closes it into a `ks` matrix. **This package never sees the
surmise relation** — callers supply the closed `ks` directly. (The backend that
consumes this package reconstructs prerequisite structure separately.)

### BLIM (Basic Local Independence Model)
The probabilistic response model linking a student's true knowledge state to
their observed answers. It has exactly two per-item parameters: `beta` and
`eta`. The [Bayesian update](#bayesian-update--posterior) is derived from BLIM.

### beta (careless error)
Probability that a student who **knows** an item answers it **incorrectly** (a
slip). Per-item, in `[0, 1]`. `simplified_assessment` uses one scalar for all
items (default `0.1`).

### eta (lucky guess)
Probability that a student who does **not know** an item answers it
**correctly** by guessing. Per-item, in `[0, 1]`. Default scalar `0.1`. The code
warns when `beta + eta > 1` for an item (the model degenerates).

---

## The two primitives

### Halfsplit question selection
The adaptive item-selection rule: pick the item whose probability of mastery
under the current belief is closest to `0.5` — the item that most evenly splits
the probability mass over remaining states. Ties are broken uniformly at random
via an **injected** `rng` (see
[ADR-0003](docs/adr/0003-injected-rng.md)). The public entry point is
`halfsplit_question`; it returns a 1-based item number. Ports
`kmassesshalfsplit.R`.

### `item_prob`
A length-`n` float64 vector where `item_prob[q] = Σ_s probs[s] · ks[s, q]`: the
belief-weighted marginal probability that item `q+1` is mastered. Computed as
the matrix-vector product `probs @ ks`. Halfsplit picks `argmin |item_prob −
0.5|`.

### Bayesian update / posterior
After a student responds to an item, the belief over states is updated by Bayes'
rule (prior × likelihood ÷ normaliser). The BLIM likelihood for state `s` given
item `q` is: `1−beta[q]` (correct & item in `s`), `beta[q]` (wrong & item in
`s`), `eta[q]` (correct & item not in `s`), `1−eta[q]` (wrong & item not in
`s`). The public entry point is `bayesian_update`; it returns a **new** array
and never mutates its inputs. Ports `kmassessbayesian.R`.

### `probs`
The length-`N` float64 probability distribution over knowledge states;
`probs[s]` is the current belief the student is in state `s`. Must sum to 1
within `_R_FLOAT_TOL`. Initialised to a uniform prior (`1/N` each).

---

## The assessment loop

### `simplified_assessment`
The full non-interactive adaptive loop (ports `kmsassess` / `kmassess.R`). Given
a pre-recorded binary `responses` vector, it repeatedly calls
`halfsplit_question` then `bayesian_update` until the stopping criterion is met,
and returns an `AssessmentResult`. This is the reference loop; a web backend
needing one-question-per-request **drives the two primitives itself** rather
than calling this (see [ADR-0001](docs/adr/0001-pure-functions-only.md)).

### `threshold` (stopping criterion)
A scalar in `(0, 1]` (recommended `> 0.5`; default `0.51`). The loop terminates
when `max(probs) > threshold` — i.e. when a single knowledge state has gathered
enough probability mass to be the diagnosis. The code warns if `threshold ≤
0.5`.

### Runaway guard
A safety cap: if the loop would ask more than `2 · n` questions without crossing
the threshold, it emits warnings and returns `None`. Mirrors the R guard in
`kmassess.R`.

### `responses`
A length-`n` binary array where `responses[i−1]` is the student's answer to
1-based item `i`. The loop reads an answer only when halfsplit selects that
item. Matches R's `kmsassess(r=...)`.

### `AssessmentResult`
Frozen dataclass returned by the loop. Fields mirror R's `kmassess` return list:
- `state` — the diagnosed [knowledge state](#knowledge-state) as a binary tuple
  (the `ks` row(s) at the `argmax` of `probs`).
- `probs` — the final distribution, or the full history if `probdev=True`.
- `queried` — the 1-based sequence of items asked.
- `qtime` / `utime` — timing averages, present only for API parity with R;
  non-functional and never asserted on.

### `probdev` flag
When `True`, `simplified_assessment` returns the entire sequence of
distributions (prior + one per update) in `result.probs` instead of just the
final vector.

---

## Conventions

### 1-based vs 0-based item indices
The **public API is 1-based** (items numbered `1..n`), matching `kstMatrix`'s
output so `queried` aligns with the R reference. Conversion to 0-based happens
immediately at each boundary (`q = item − 1`). See
[ADR-0005](docs/adr/0005-one-based-item-indices.md).

### `_R_FLOAT_TOL`
`sqrt(float64 eps) ≈ 1.49e-8`, mirroring R's `kmdoubleequal` tolerance. Defined
in `src/abinavi_kst/assessment.py`; used for the `probs` sum-to-one check. (The
R-equivalence tests use a separate `r_tol` fixture loaded from `meta.json`.)

### Multi-winner state flattening
When two or more states tie at the maximum probability on termination, R
flattens the winning submatrix **column-major** (`as.integer(ks[winners,])`);
the Python port mirrors this with `ks[winners].reshape(-1, order="F")`. Only
reachable under ties.
