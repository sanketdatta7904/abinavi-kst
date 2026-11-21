# ADR-0006: GPL-3 licensing inherited from kstMatrix / kstpy

- **Status**: Accepted (forced by upstream copyleft)
- **Date**: 2026-11-21 (recorded retroactively from the package metadata)

## Context

The three routines are **line-by-line translations** of GPL-3 R code in
`kstMatrix`, and the package is shaped to contribute upstream to `kstpy` (also
GPL-3).

## Decision

The package is licensed **GPL-3.0-only**, matching both `kstMatrix` (the R
source) and `kstpy` (the target upstream).

## Consequences

- This was **not a free choice** — it is a copyleft consequence of translating
  GPL-3 source line-by-line.
- **Downstream impact**: any project that consumes `abinavi-kst` is bound by
  GPL-3. The consuming backend (`abinavi-be`) takes this on by depending on the
  package. Revisit only if the math is re-derived independently from a
  non-GPL specification (a large undertaking, not currently planned).

## Evidence

`pyproject.toml` (`license = "GPL-3.0-only"`); `README.md` License section.
