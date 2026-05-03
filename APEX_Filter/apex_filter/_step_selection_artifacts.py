"""Internal helpers for removing stale step-level selection artifacts."""

from __future__ import annotations

import os


_STALE_SELECTION_ARTIFACTS = (
    "selection_candidates.csv",
    "selection_worklist.csv",
    "selection_guide.md",
    "selection_candidates.json",
    "pick_labels_all.json",
    "pick_labels_template.json",
)


def _cleanup_step_selection_artifacts(step_dir: str) -> None:
    """Remove stale pick/selection helper files from a step directory."""
    for stale in _STALE_SELECTION_ARTIFACTS:
        stale_path = os.path.join(step_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
