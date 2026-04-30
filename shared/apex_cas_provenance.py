"""Shared helpers for consuming apex-cas stage sidecars in downstream tools."""

from __future__ import annotations

import os
from typing import Any

from shared.artifact_paths import load_json_if_exists
from shared.setting_utils import apply_overrides


_NON_SETTING_KEYS = {
    "apex_cas_case_dir",
    "charge",
    "spin",
    "symmetry_group",
    "reduction_symmetry",
    "family_scheme",
    "benchmark_profile",
    "config_reduction_mode",
    "structure_path",
    "cluster_info_path",
    "fcidump_path",
    "fcidump_ecore_path",
    "initial_guess",
    "uhf_filter",
    "filtration",
    "workdir",
}


def load_apex_cas_provenance(case_dir: str) -> dict[str, Any]:
    """Load stage sidecars emitted by ``apex-cas`` for provenance consumers."""
    from apex_cas.state_io import find_chkfile

    scf_dir = os.path.join(case_dir, "outputs", "scf")
    chkfile = find_chkfile(scf_dir)
    stem = os.path.splitext(os.path.basename(chkfile))[0]
    output_dir = os.path.join(case_dir, "outputs")

    scf_info = load_json_if_exists(os.path.join(output_dir, "scf", f"{stem}_scf_info.json"))
    cas_info = load_json_if_exists(os.path.join(output_dir, "scf", f"{stem}_cas_info.json"))

    return {
        "stem": stem,
        "scf_info": scf_info or {},
        "cas_info": cas_info or {},
    }


def build_effective_settings_from_apex_cas(
    *,
    config_raw: dict[str, Any],
    case_dir: str,
    settings_cls,
    provenance_loader=load_apex_cas_provenance,
) -> tuple[Any, dict[str, Any]]:
    """Build downstream effective settings from apex-cas sidecars plus YAML overrides."""
    provenance = provenance_loader(case_dir)
    valid_setting_keys = set(settings_cls.__dataclass_fields__.keys())
    sidecar_overrides: dict[str, Any] = {}
    scf_settings = provenance["scf_info"].get("settings", {}).get("scf", {})
    cas_build_settings = provenance["cas_info"].get("settings", {}).get("cas_build", {})
    sidecar_overrides.update({k: v for k, v in scf_settings.items() if k in valid_setting_keys})
    sidecar_overrides.update({k: v for k, v in cas_build_settings.items() if k in valid_setting_keys})

    yaml_for_settings = dict(config_raw)
    for key in _NON_SETTING_KEYS:
        yaml_for_settings.pop(key, None)

    settings = settings_cls()
    if sidecar_overrides:
        settings = apply_overrides(settings, **sidecar_overrides)
    if yaml_for_settings:
        settings = apply_overrides(settings, **yaml_for_settings)

    provenance["effective_settings_source"] = {
        "apex_cas_sidecar_keys": sorted(sidecar_overrides.keys()),
        "filter_override_keys": sorted(yaml_for_settings.keys()),
    }
    return settings, provenance
