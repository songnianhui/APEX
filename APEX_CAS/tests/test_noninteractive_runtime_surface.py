"""Regression coverage for non-interactive Step 1-10 runtime modules."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path("/Users/snh/Projects/APEX")


def test_step10_runtime_contains_no_input_calls():
    roots = [
        REPO_ROOT / "APEX_CAS" / "apex_cas",
        REPO_ROOT / "APEX_Filter" / "apex_filter",
        REPO_ROOT / "shared",
    ]
    offenders: list[str] = []
    for root in roots:
        for path in root.glob("*.py"):
            if path.name in {"steps_fno.py", "fno_truncation.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "input(" in text:
                offenders.append(str(path))
    assert offenders == []


def test_step10_runtime_contains_no_sys_exit_calls():
    roots = [
        REPO_ROOT / "APEX_CAS" / "apex_cas",
        REPO_ROOT / "APEX_Filter" / "apex_filter",
        REPO_ROOT / "shared",
    ]
    offenders: list[str] = []
    for root in roots:
        for path in root.glob("*.py"):
            if path.name in {"steps_fno.py", "fno_truncation.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "sys.exit(" in text:
                offenders.append(str(path))
    assert offenders == []
