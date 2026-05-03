"""Settings utilities shared across APEX packages."""

import re
from dataclasses import replace as _replace

import yaml

from .models import ComputationSettings as _ComputationSettings

_ELEMENT_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


DEFAULT_PRESET = _replace(_ComputationSettings(), max_cycle=200)

FAST_PRESET = _replace(
    DEFAULT_PRESET,
    basis_set_default="def2-SVP",
    basis_set_per_element={},
    relativistic="none",
    solvation_model="none",
    conv_tol=1e-6,
    max_cycle=100,
)

PRESETS = {
    "default": DEFAULT_PRESET,
    "fast": FAST_PRESET,
}


def load_basis_file(filepath: str) -> dict:
    """Load per-element basis set overrides from a YAML file."""
    with open(filepath, "r") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {filepath}, "
            f"got {type(data).__name__}"
        )

    result = {}
    for key, value in data.items():
        key_str = str(key)
        if not _ELEMENT_SYMBOL_RE.match(key_str):
            raise ValueError(
                f"Invalid element symbol {key_str!r} in {filepath}. "
                "Must be 1-2 characters with first letter uppercase."
            )
        if not isinstance(value, str):
            raise ValueError(
                f"Basis set for element {key_str!r} must be a string, "
                f"got {type(value).__name__}"
            )
        result[key_str] = value

    return result


def apply_overrides(settings: _ComputationSettings, **overrides) -> _ComputationSettings:
    """Apply keyword overrides to a ComputationSettings instance."""
    basis_per_element_override = overrides.pop("basis_set_per_element", None)
    basis_file = overrides.pop("basis_set_file", None)
    merged_basis = dict(settings.basis_set_per_element)

    if basis_per_element_override is not None:
        if len(basis_per_element_override) == 0:
            merged_basis = {}
        else:
            merged_basis.update(basis_per_element_override)

    if basis_file is not None:
        merged_basis.update(load_basis_file(basis_file))

    return _replace(settings, basis_set_per_element=merged_basis, **overrides)


def load_cas_settings_file(filepath: str) -> dict:
    """Load CAS settings overrides from a YAML file."""
    with open(filepath, "r") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {filepath}, "
            f"got {type(data).__name__}"
        )

    if "basis_per_element" in data:
        data["basis_set_per_element"] = data.pop("basis_per_element")

    return data


def settings_from_preset(preset_name: str, **overrides) -> _ComputationSettings:
    """Get a preset configuration and apply overrides in one call."""
    if preset_name not in PRESETS:
        raise KeyError(
            f"Unknown preset {preset_name!r}. "
            f"Available presets: {sorted(PRESETS)}"
        )
    return apply_overrides(PRESETS[preset_name], **overrides)


def build_basis_dict(cluster_info, settings: _ComputationSettings):
    """Build a PySCF basis specification from cluster info and settings."""
    elements = set()

    for metal in cluster_info.metals:
        elements.add(metal.element)

    for bridge in cluster_info.bridging_atoms:
        elements.add(bridge.element)

    for ligand in cluster_info.terminal_ligands:
        for idx in ligand.atom_indices:
            if (
                cluster_info.all_elements is not None
                and idx < len(cluster_info.all_elements)
            ):
                elements.add(cluster_info.all_elements[idx])

    if cluster_info.all_elements:
        elements.update(cluster_info.all_elements)

    return {element: settings.get_basis(element) for element in sorted(elements)}
