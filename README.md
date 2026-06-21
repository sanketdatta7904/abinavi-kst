# abinavi-kst

Pure-function adaptive-assessment routines for knowledge space theory
(KST), implemented as a small standalone Python package.

The functions in this package are direct line-by-line translations of
`kmsassess`, `kmassesshalfsplit`, and `kmassessbayesian` from the R
package [`kstMatrix`](https://cran.r-project.org/package=kstMatrix)
(GPL-3, Hockemeyer et al.). They are shaped to match an upstream
contribution to [`kstpy`](https://pypi.org/project/kstpy/), which today
covers KST structure construction but does not yet expose adaptive
assessment.

## Design rules

1. **Pure functions only.** No classes for the math. No globals.
2. **Frozen inputs.** Functions never mutate their arguments.
3. **Injected randomness.** Tie-breaking in `halfsplit_question` accepts
   an explicit `numpy.random.Generator` rather than relying on global
   state — so tests can pin behavior with a seed.
4. **NumPy at the boundary.** Knowledge structures are `np.ndarray` of
   dtype `int8`. Probability distributions are `np.ndarray` of `float64`.
5. **R-compatible item indices.** The public API uses 1-based item
   numbers to match `kstMatrix`'s output (so `result["queried"]` aligns
   with the R reference). Internal NumPy code is 0-based.

## Public API

```python
from abinavi_kst import (
    halfsplit_question,    # pick the next item to ask
    bayesian_update,       # posterior P(state | response)
    simplified_assessment, # full kmsassess equivalent (non-interactive)
)
```

For the interactive (per-question) use case in a web backend, drive the
two primitives yourself; see `simplified_assessment` for the reference
loop.

## License

GPL-3.0-only, matching `kstMatrix` and `kstpy`.
