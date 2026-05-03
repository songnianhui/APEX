"""Shared helpers for deterministic checkpoint-file discovery."""

from __future__ import annotations

import glob
import os


def find_chkfile(scf_dir: str) -> str:
    """Return the authoritative chkfile in an SCF output directory.

    The canonical V1.0 behavior is deterministic and non-interactive:
    choose the largest non-empty ``.chk`` file.
    """
    chk_candidates = [
        path for path in glob.glob(os.path.join(scf_dir, "*.chk")) if os.path.getsize(path) > 0
    ]
    if not chk_candidates:
        raise FileNotFoundError(f"No valid chkfile found in {scf_dir}/")

    chk_candidates.sort(key=lambda path: os.path.getsize(path), reverse=True)
    chkfile = chk_candidates[0]

    if len(chk_candidates) > 1:
        print(
            f"  Multiple chkfiles found in {scf_dir}/; "
            "using the largest non-empty candidate."
        )

    print(f"  Using chkfile: {os.path.basename(chkfile)}")
    return chkfile
