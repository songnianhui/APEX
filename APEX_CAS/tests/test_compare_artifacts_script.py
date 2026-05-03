"""Smoke tests for the generic compare_artifacts CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "compare_artifacts.py"


def test_compare_artifacts_script_handles_json_payloads(tmp_path):
    ref = tmp_path / "ref.json"
    new = tmp_path / "new.json"
    ref.write_text(json.dumps({"energy": -1.0, "converged": True}))
    new.write_text(json.dumps({"energy": -1.1, "converged": True}))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(ref), str(new), "--format", "json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["kind"] == "json"
    assert abs(payload["numeric_summary"]["max_abs"] - 0.1) < 1e-12
