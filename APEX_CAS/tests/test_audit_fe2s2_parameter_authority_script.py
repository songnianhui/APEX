"""Smoke test for the committed Fe2S2 parameter-authority audit script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_audit_fe2s2_parameter_authority_script_reports_clean_bundle():
    script = REPO_ROOT / "scripts" / "audit_fe2s2_parameter_authority.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK"
    assert payload["checked"] == 19
    assert payload["issues"] == []
