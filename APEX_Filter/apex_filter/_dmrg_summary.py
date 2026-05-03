"""Internal helpers for Step 8/9 DMRG summary semantics."""

from __future__ import annotations


_DMRG_SOURCE_MODE_CONVERGED_ONLY = "converged_only"
_DMRG_SOURCE_MODE_UNCONVERGED_FALLBACK = "unconverged_fallback"


def _dmrg_source_mode_allows_ranking(source_mode: str | None) -> bool:
    """Return whether a DMRG summary entry may influence ranking directly."""
    return source_mode != _DMRG_SOURCE_MODE_UNCONVERGED_FALLBACK
