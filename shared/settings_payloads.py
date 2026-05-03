"""Shared helpers for normalized requested/effective settings payloads."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable as _Iterable

METHOD_IDENTITY_KEYS = frozenset(
    {
        "theory",
        "backend",
        "basis_mode",
        "source_method",
        "schedule_mode",
        "dmrg_mode",
        "symm_type",
        "code",
        "scf_method",
        "xc_functional",
        "relativistic",
        "solvation_model",
        "localization_method",
        "cpt_cas_type",
        "ordering_matrix_mode",
        "ordering_objective",
    }
)

ACTIVE_SPACE_CC_RECORD_ONLY_KEYS = frozenset({"basis_set"})


def build_base_settings_payload(
    source_settings,
    *,
    control_source: str | None = None,
    theory: str | None = None,
    **overrides,
) -> dict:
    """Build a flat settings payload from saved state settings plus overrides."""
    payload: dict = {}
    if source_settings is not None:
        if dataclasses.is_dataclass(source_settings):
            payload.update(dataclasses.asdict(source_settings))
        else:
            payload.update(dict(source_settings))
    if control_source is not None:
        payload["control_source"] = control_source
    if theory is not None:
        payload["theory"] = theory
    payload.update(overrides)
    return payload


def extend_settings_payload(settings_payload: dict | None, **overrides) -> dict | None:
    """Return a copied payload with additional flat fields applied."""
    if settings_payload is None:
        return None
    return {
        **settings_payload,
        **overrides,
    }


def build_effective_localization_payload(settings, settings_payload: dict | None) -> dict:
    """Build the localization block actually used by a route."""
    method = ""
    if settings_payload:
        method = str(settings_payload.get("localization_method", "") or "")
    method = method.strip().lower()

    if not method:
        return {"method": "", "parameters": {}}

    if method == "pm":
        params = {
            "pop_method": getattr(settings, "pm_pop_method", "mulliken"),
            "conv_tol": getattr(settings, "pm_conv_tol", 1e-8),
            "conv_tol_grad": getattr(settings, "pm_conv_tol_grad", None),
            "max_cycle": getattr(settings, "pm_max_cycle", 100),
            "exponent": getattr(settings, "pm_exponent", 2),
            "init_guess": getattr(settings, "pm_init_guess", "atomic"),
        }
    elif method == "boys":
        params = {
            "conv_tol": getattr(settings, "boys_conv_tol", 1e-7),
            "conv_tol_grad": getattr(settings, "boys_conv_tol_grad", None),
            "max_cycle": getattr(settings, "boys_max_cycle", 150),
        }
    else:
        params = {}

    return {"method": method, "parameters": params}


def build_effective_selection_payload(cas, settings_payload: dict | None) -> dict:
    """Build the actual active-orbital selection description used by the route."""
    method = str(getattr(cas, "selection_method", "") or "").strip().lower()

    if method == "noon":
        params = {
            "occ_lo": 0.02,
            "occ_hi": 1.98,
        }
    elif method == "character":
        params = {}
        if settings_payload and "projection_threshold" in settings_payload:
            params["projection_threshold"] = settings_payload["projection_threshold"]
    elif method == "all":
        params = {}
    else:
        params = {}

    return {"method": method, "parameters": params}


def build_requested_cas_payload(settings_payload: dict | None) -> dict:
    """Build the route/config snapshot requested by the user."""
    payload = {}
    if settings_payload:
        payload["cas_build"] = dict(settings_payload)
    return payload


def build_effective_parameter_payload(cas, settings, settings_payload: dict | None) -> dict:
    """Build only the parameters that actually affected the executed route."""
    localization = build_effective_localization_payload(settings, settings_payload)
    selection = build_effective_selection_payload(cas, settings_payload)
    return {
        "localization": localization.get("parameters", {}),
        "selection": selection.get("parameters", {}),
    }


def normalize_settings_payload(
    settings_payload: dict | None,
    *,
    record_only_keys: _Iterable[str] | None = None,
) -> dict | None:
    """Augment flat settings payloads with requested/effective structure.

    Existing flat top-level keys are preserved for compatibility with current
    summaries and tests. The normalized nested blocks provide the canonical
    authority shape for new artifacts:

    - ``requested_config``
    - ``effective_method``
    - ``effective_parameters``
    """
    if settings_payload is None:
        return None

    raw = dict(settings_payload)
    normalized = dict(raw)
    record_only = {str(key) for key in (record_only_keys or ())}

    method = {}
    for key in METHOD_IDENTITY_KEYS:
        if key in raw:
            method[key] = raw[key]

    requested = {
        key: value
        for key, value in raw.items()
        if key not in {"control_source", "requested_config", "effective_method", "effective_parameters"}
    }
    effective_parameters = {
        key: value
        for key, value in requested.items()
        if key not in METHOD_IDENTITY_KEYS
        and key not in record_only
    }

    current_requested = raw.get("requested_config")
    if isinstance(current_requested, dict):
        normalized["requested_config"] = {**requested, **current_requested}
    else:
        normalized["requested_config"] = requested

    current_method = raw.get("effective_method")
    if isinstance(current_method, dict):
        normalized["effective_method"] = {**method, **current_method}
    else:
        normalized["effective_method"] = method

    current_parameters = raw.get("effective_parameters")
    if isinstance(current_parameters, dict):
        current_parameters = {
            key: value
            for key, value in current_parameters.items()
            if key not in method and key not in record_only
        }
        normalized["effective_parameters"] = {**effective_parameters, **current_parameters}
    else:
        normalized["effective_parameters"] = effective_parameters
    return normalized


def find_effective_parameter_leaks(
    settings_payload: dict | None,
    *,
    method_identity_keys: _Iterable[str] | None = None,
    record_only_keys: _Iterable[str] | None = None,
) -> set[str]:
    """Return keys that leaked into ``effective_parameters``.

    This helper is intended for artifact-audit and regression coverage. It
    accepts either a fully normalized payload or a flat payload that can be
    normalized on the fly.
    """
    normalized = normalize_settings_payload(
        settings_payload,
        record_only_keys=record_only_keys,
    )
    if normalized is None:
        return set()

    forbidden = {str(key) for key in (method_identity_keys or METHOD_IDENTITY_KEYS)}
    forbidden.update(str(key) for key in (record_only_keys or ()))

    leaks: set[str] = set()
    effective_parameters = normalized.get("effective_parameters", {})
    if isinstance(effective_parameters, dict):
        leaks.update(key for key in effective_parameters if key in forbidden)

    raw_effective_parameters = (
        settings_payload.get("effective_parameters")
        if isinstance(settings_payload, dict)
        else None
    )
    if isinstance(raw_effective_parameters, dict):
        leaks.update(key for key in raw_effective_parameters if key in forbidden)

    return leaks


def missing_normalized_settings_sections(settings_payload: dict | None) -> set[str]:
    """Return missing canonical nested sections for a normalized payload."""
    normalized = normalize_settings_payload(settings_payload)
    if normalized is None:
        return {
            "requested_config",
            "effective_method",
            "effective_parameters",
        }

    expected = {
        "requested_config",
        "effective_method",
        "effective_parameters",
    }
    return {
        key
        for key in expected
        if not isinstance(normalized.get(key), dict)
    }
