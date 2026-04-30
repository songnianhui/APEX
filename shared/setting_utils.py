"""Settings utilities shared across APEX packages."""

import re
from dataclasses import replace

import yaml

from .models import ComputationSettings

_ELEMENT_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


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


def apply_overrides(settings: ComputationSettings, **overrides) -> ComputationSettings:
    """Apply keyword overrides to a ComputationSettings instance."""
    basis_per_element_override = overrides.pop("basis_set_per_element", None)
    merged_basis = dict(settings.basis_set_per_element)

    if basis_per_element_override is not None:
        merged_basis.update(basis_per_element_override)

    return replace(settings, basis_set_per_element=merged_basis, **overrides)


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


load_scf_settings_file = load_cas_settings_file


def build_basis_dict(cluster_info, settings: ComputationSettings):
    """Build a PySCF basis specification from cluster info and settings."""
    if settings.basis_set_file:
        return settings.basis_set_file

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
