from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_parameter_input_surface_audit_script_reports_ok():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "audit_parameter_input_surfaces.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK"
    assert payload["issues"] == []
    assert payload["checked"]["cas_template_keys"] > 0
    assert payload["checked"]["filter_template_keys"] > 0
    assert payload["checked"]["method_control_sections"] > 0
