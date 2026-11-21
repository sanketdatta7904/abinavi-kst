# ADR-0005: 1-based public item indices matching R

- **Status**: Accepted
- **Date**: 2026-11-21 (recorded retroactively from the package's design rules)

## Context

This package is a translation of `kstMatrix`, whose output (`result$queried`)
numbers items from 1. The R package is the correctness oracle, and fixtures are
compared directly against it.

## Decision

The **public API is 1-based**: `halfsplit_question` returns a 1-based item
number; `bayesian_update` and `simplified_assessment` accept 1-based `item` /
`queried` values. Internal NumPy code is 0-based, converting at the boundary
with `q = item - 1`.

## Consequences

- **Benefit**: `queried` aligns directly with the R reference — no renumbering
  when comparing fixtures or reasoning against the R source.
- **Cost**: every boundary conversion is a potential off-by-one site; the two
  index conventions coexist and must not be confused.

## Evidence

`README.md` design rule 5; `src/abinavi_kst/assessment.py` module docstring
("Item indexing"), the `+1` on halfsplit return and `item - 1` conversions.
