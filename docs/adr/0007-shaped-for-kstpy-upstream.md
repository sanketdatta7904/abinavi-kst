# ADR-0007: Shaped as an upstream kstpy contribution

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package metadata)

## Context

[`kstpy`](https://pypi.org/project/kstpy/) is the established Python KST package,
but it covers knowledge-structure *construction* and does not yet expose
*adaptive assessment*. This package fills that gap and intends to merge upstream.

## Decision

The public API, naming, and signatures are deliberately shaped to fit `kstpy`'s
style, so the routines can be contributed upstream. The package is explicitly a
**temporary home** for functions destined for `kstpy`.

## Consequences

- **Benefit**: a clear deprecation path — once the upstream PR lands, consumers
  migrate from `abinavi-kst` to `kstpy` with minimal churn.
- **Cost**: unreviewed kstpy conventions may later force renames or parameter
  reordering here. The migration itself is **unscheduled**.
- The consuming backend already anticipates this: its KST wrapper is written so
  that swapping `abinavi_kst` for `kstpy` touches only one module.

## Evidence

`pyproject.toml` description; `README.md` intro.
