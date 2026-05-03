"""Smoke test for the Fe2S2 compare script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "compare_fe2s2_runs.py"
CURRENT = REPO_ROOT / "examples" / "fe2s2"
BASELINE = Path("/Users/snh/Projects/APEX_bk/examples/fe2s2")


def test_compare_fe2s2_runs_script_handles_current_two_state_bundle():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--current",
            str(CURRENT),
            "--baseline",
            str(BASELINE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "## APEX_CAS DMRG Matrix Compare" in stdout
    assert "dmrg_1rdm_eigenvalue_max_abs" in stdout
    assert "dmrg_1rdm_basis_rotation_likely" in stdout
    assert "## Step8 DMRG Ladder" in stdout
    assert (
        "Fe1↑Fe2↓|2xFe(III)|d:none" in stdout
        or "(no common Step8 records yet)" in stdout
    )
    assert "## APEX_CAS FCIDUMP Deep Compare" in stdout
    assert "## Chan Bundle Presence" in stdout
