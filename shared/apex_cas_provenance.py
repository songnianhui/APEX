"""Shared helpers for consuming apex-cas stage sidecars in downstream tools."""

from __future__ import annotations

import os
from typing import Any as _Any

from shared.artifact_paths import load_json_if_exists as _load_json_if_exists
from shared.chkfiles import find_chkfile as _find_chkfile
from shared.setting_utils import apply_overrides as _apply_overrides


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


def load_apex_cas_provenance(case_dir: str) -> dict[str, _Any]:
    """Load stage sidecars emitted by ``apex-cas`` for provenance consumers."""
    scf_dir = os.path.join(case_dir, "outputs", "scf")
    chkfile = _find_chkfile(scf_dir)
    stem = os.path.splitext(os.path.basename(chkfile))[0]
    output_dir = os.path.join(case_dir, "outputs")

    scf_info = _load_json_if_exists(os.path.join(output_dir, "scf", f"{stem}_scf_info.json"))
    cas_info = _load_json_if_exists(os.path.join(output_dir, "scf", f"{stem}_cas_info.json"))

    return {
        "stem": stem,
        "scf_info": scf_info or {},
        "cas_info": cas_info or {},
    }


def build_effective_settings_from_apex_cas(
    *,
    config_raw: dict[str, _Any],
    case_dir: str,
    settings_cls,
    provenance_loader=load_apex_cas_provenance,
) -> tuple[_Any, dict[str, _Any]]:
    """Build downstream effective settings from apex-cas sidecars plus YAML overrides."""

    def _translate_localization_settings(method: str | None, params: dict[str, _Any]) -> dict[str, _Any]:
        if not isinstance(params, dict):
            return {}
        method_key = (method or "").strip().lower()
        if method_key == "pm":
            mapping = {
                "pop_method": "pm_pop_method",
                "conv_tol": "pm_conv_tol",
                "conv_tol_grad": "pm_conv_tol_grad",
                "max_cycle": "pm_max_cycle",
                "exponent": "pm_exponent",
                "init_guess": "pm_init_guess",
            }
        elif method_key == "boys":
            mapping = {
                "conv_tol": "boys_conv_tol",
                "conv_tol_grad": "boys_conv_tol_grad",
                "max_cycle": "boys_max_cycle",
            }
        else:
            return {}
        return {
            target: params[source]
            for source, target in mapping.items()
            if source in params
        }

    provenance = provenance_loader(case_dir)
    valid_setting_keys = set(settings_cls.__dataclass_fields__.keys())
    sidecar_overrides: dict[str, _Any] = {}
    scf_settings = provenance["scf_info"].get("settings", {}).get("scf", {})
    cas_info = provenance["cas_info"]
    requested_config = cas_info.get("requested_config", {})
    cas_build_settings = requested_config.get("cas_build")
    if not isinstance(cas_build_settings, dict):
        cas_build_settings = {}
    localization_payload = (
        cas_info.get("effective_method", {}).get("localization", {})
    )
    localization_settings = _translate_localization_settings(
        localization_payload.get("method"),
        localization_payload.get("parameters", {}),
    )
    sidecar_overrides.update({k: v for k, v in scf_settings.items() if k in valid_setting_keys})
    sidecar_overrides.update({k: v for k, v in cas_build_settings.items() if k in valid_setting_keys})
    sidecar_overrides.update({k: v for k, v in localization_settings.items() if k in valid_setting_keys})

    yaml_for_settings = dict(config_raw)
    for key in _NON_SETTING_KEYS:
        yaml_for_settings.pop(key, None)

    settings = settings_cls()
    if sidecar_overrides:
        settings = _apply_overrides(settings, **sidecar_overrides)
    if yaml_for_settings:
        settings = _apply_overrides(settings, **yaml_for_settings)

    provenance["effective_settings_source"] = {
        "apex_cas_sidecar_keys": sorted(sidecar_overrides.keys()),
        "filter_override_keys": sorted(yaml_for_settings.keys()),
    }
    return settings, provenance
