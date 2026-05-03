"""Internal helpers for resolving APEX_CAS case-side artifacts."""

from __future__ import annotations

import os
from pathlib import Path as _Path

import yaml

from shared.apex_cas_provenance import load_apex_cas_provenance as _load_apex_cas_provenance
from shared.artifact_paths import (
    resolve_cluster_info_path as _resolve_cluster_info_path,
    resolve_structure_path as _resolve_structure_path,
)


def _resolve_case_dir_from_fcidump_path(fcidump_path: str) -> str:
    return str(_Path(fcidump_path).resolve().parents[2])


def _resolve_cas_settings_path(case_dir: str) -> str | None:
    inputs_dir = os.path.join(case_dir, "inputs")
    if not os.path.isdir(inputs_dir):
        return None
    matches = [
        os.path.join(inputs_dir, name)
        for name in sorted(os.listdir(inputs_dir))
        if name.endswith("_cas_settings.yaml") or name == "cas_settings.yaml"
    ]
    if len(matches) == 1:
        return os.path.abspath(matches[0])
    return None


def _resolve_cas_data_h5_path(case_dir: str) -> str | None:
    provenance = _load_apex_cas_provenance(case_dir)
    stem = provenance.get("stem", "")
    if not stem:
        return None
    path = os.path.join(case_dir, "outputs", "orbitals", f"{stem}_cas_data.h5")
    return os.path.abspath(path) if os.path.isfile(path) else None


def _preferred_step3_state_path(uhf_dir: str, safe_label: str) -> str:
    h5_path = os.path.join(uhf_dir, f"{safe_label}_uhf.h5")
    if os.path.isfile(h5_path):
        return h5_path
    return os.path.join(uhf_dir, f"{safe_label}_uhf.npz")


def _preferred_step7_basis_path(basis_dir: str, safe_label: str) -> str:
    h5_path = os.path.join(basis_dir, f"{safe_label}_dmrg_basis.h5")
    if os.path.isfile(h5_path):
        return h5_path
    return os.path.join(basis_dir, f"{safe_label}_dmrg_basis.npz")


def _build_case_observable_inputs(state: dict, cfg) -> dict | None:
    config_path = state.get("config_path")
    if not config_path or not os.path.isfile(config_path):
        return None
    config_raw = yaml.safe_load(_Path(config_path).read_text()) or {}
    case_dir = _resolve_case_dir_from_fcidump_path(state["fcidump_path"])
    config_dir = os.path.dirname(os.path.abspath(config_path))
    xyz_path = _resolve_structure_path(config_raw, case_dir)
    cluster_info_path = _resolve_cluster_info_path(config_raw, case_dir, config_dir)
    cas_settings_path = _resolve_cas_settings_path(case_dir)
    cas_data_h5_path = _resolve_cas_data_h5_path(case_dir)
    if not all([xyz_path, cluster_info_path, cas_settings_path, cas_data_h5_path]):
        return None
    return {
        "xyz_path": xyz_path,
        "cluster_info_path": cluster_info_path,
        "cas_settings_path": cas_settings_path,
        "cas_data_h5_path": cas_data_h5_path,
        "label": cfg.label,
        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
    }
