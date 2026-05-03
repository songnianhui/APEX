"""Shared selection-file parsing helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_active_selection(filepath: str) -> tuple[list[int], int]:
    """Parse ``selection.txt`` and return sorted orbital indices plus ``n-electrons``."""
    n_electrons = None
    n_orbital = None
    indices = []

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped and not any(
                c.isalpha() for c in stripped.split(":")[0].replace("-", "")
            ):
                parts = stripped.split(":", 1)
                key = parts[0].strip().lower()
                val = parts[1].strip()
                if key == "n-electrons":
                    n_electrons = int(val)
                elif key == "n-orbital":
                    n_orbital = int(val)
                continue
            tokens = stripped.split()
            try:
                indices.extend(int(t) for t in tokens)
            except ValueError:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    if key == "n-electrons":
                        n_electrons = int(val)
                    elif key == "n-orbital":
                        n_orbital = int(val)

    if n_electrons is None:
        raise ValueError(f"'n-electrons' not found in selection file: {filepath}")
    if n_orbital is None:
        raise ValueError(f"'n-orbital' not found in selection file: {filepath}")

    indices = sorted(set(indices))
    if len(indices) != n_orbital:
        raise ValueError(
            "n-orbital "
            f"({n_orbital}) does not match number of indices ({len(indices)}) "
            f"in selection file {filepath}: {indices}"
        )

    logger.info(
        "Loaded selection from %s: %d orbitals, %d electrons",
        filepath,
        n_orbital,
        n_electrons,
    )
    return indices, n_electrons
