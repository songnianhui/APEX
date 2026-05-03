#!/usr/bin/env python3
"""Audit committed Fe2S2 artifacts for parameter-authority consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "APEX_CAS"))
sys.path.insert(0, str(REPO_ROOT / "APEX_Filter"))
sys.path.insert(0, str(REPO_ROOT))

from shared.settings_payloads import (  # noqa: E402
    ACTIVE_SPACE_CC_RECORD_ONLY_KEYS,
    find_effective_parameter_leaks,
    missing_normalized_settings_sections,
)


CASE_DIR = REPO_ROOT / "examples" / "fe2s2"
OUTPUTS_DIR = CASE_DIR / "outputs"
SESSION_DIR = CASE_DIR / "filter_session"
STEM = "C4H12Fe2S6_uks_BP86_tzp-dkh"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_h5_settings(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return json.loads(h5["metadata"].attrs["settings_json"])


def _audit_payload(
    *,
    label: str,
    payload: dict,
    record_only_keys: set[str] | None = None,
) -> dict:
    return {
        "label": label,
        "missing_sections": sorted(missing_normalized_settings_sections(payload)),
        "effective_parameter_leaks": sorted(
            find_effective_parameter_leaks(
                payload,
                record_only_keys=record_only_keys,
            )
        ),
    }


def _iter_checks() -> list[dict]:
    checks: list[dict] = []

    scf_info = _load_json(OUTPUTS_DIR / "scf" / f"{STEM}_scf_info.json")
    checks.append(_audit_payload(label="apex_cas.scf_info", payload=scf_info["settings"]["scf"]))

    cas_info = _load_json(OUTPUTS_DIR / "scf" / f"{STEM}_cas_info.json")
    checks.append(_audit_payload(label="apex_cas.cas_info", payload=cas_info))

    fcidump_info = _load_json(OUTPUTS_DIR / "fcidump" / f"{STEM}_fcidump_info.json")
    checks.append(_audit_payload(label="apex_cas.fcidump_info", payload=fcidump_info["settings"]["fcidump"]))

    dmrg_info = _load_json(OUTPUTS_DIR / "fcidump" / "dmrg" / f"{STEM}_sz_M500_dmrg_info.json")
    checks.append(_audit_payload(label="apex_cas.testcas_dmrg_info", payload=dmrg_info["settings"]["dmrg"]))

    cas_h5_settings = _load_h5_settings(OUTPUTS_DIR / "orbitals" / f"{STEM}_cas_data.h5")
    checks.append(_audit_payload(label="apex_cas.cas_data_h5", payload=cas_h5_settings))

    step3_dir = SESSION_DIR / "step3_uhf" / "results"
    step6_dir = SESSION_DIR / "step6_ccsdt" / "scripts"
    step7_dir = SESSION_DIR / "step7_dmrg_basis" / "results"
    step8_dir = SESSION_DIR / "step8_dmrg" / "results"

    for path in sorted(step3_dir.glob("*_uhf.h5")):
        checks.append(_audit_payload(label=f"apex_filter.step3::{path.name}", payload=_load_h5_settings(path)))
    for path in sorted(step6_dir.glob("*_ccsdt_results.h5")):
        checks.append(
            _audit_payload(
                label=f"apex_filter.step6::{path.name}",
                payload=_load_h5_settings(path),
                record_only_keys=set(ACTIVE_SPACE_CC_RECORD_ONLY_KEYS),
            )
        )
    for path in sorted(step7_dir.glob("*_dmrg_basis.h5")):
        checks.append(_audit_payload(label=f"apex_filter.step7::{path.name}", payload=_load_h5_settings(path)))
    for path in sorted(step8_dir.glob("*_dmrg.h5")):
        checks.append(_audit_payload(label=f"apex_filter.step8::{path.name}", payload=_load_h5_settings(path)))

    return checks


def main() -> int:
    checks = _iter_checks()
    issues = [row for row in checks if row["missing_sections"] or row["effective_parameter_leaks"]]
    payload = {
        "status": "OK" if not issues else "ISSUES_FOUND",
        "checked": len(checks),
        "issues": issues,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
