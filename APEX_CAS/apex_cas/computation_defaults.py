"""Computation preset configurations and settings utilities.

Provides preset configurations for different computation scenarios.
Default settings match Chan 2019 paper parameters.
"""

import re
from dataclasses import replace

import yaml

from . import ComputationSettings

# Default preset = Chan 2019 paper settings
# (instantiating ComputationSettings() uses its default values which are Chan 2019)
CHAN_2019_PRESET = ComputationSettings()

# Fast preset for testing/small systems
FAST_PRESET = ComputationSettings(
    scf_method="uks",
    xc_functional="B3LYP",
    basis_set_default="def2-SVP",
    basis_set_per_element={},       # All elements use default (def2-SVP)
    relativistic="none",
    solvation_model="none",         # no solvation in fast mode
    solvation_epsilon=4.0,          # doesn't matter when model is "none"
    conv_tol=1e-6,                  # looser convergence for speed
    max_cycle=100,
)

PRESETS = {
    "default": CHAN_2019_PRESET,
    "fast": FAST_PRESET,
}

# Regex for valid element symbols: 1-2 characters, first uppercase letter,
# optional second lowercase letter.
_ELEMENT_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


def load_basis_file(filepath: str) -> dict:
    """Load per-element basis set overrides from a YAML file.

    Args:
        filepath: path to YAML file with format::

            Fe: "def2-QZVP"
            Mo: "def2-TZVP"
            S: "def2-TZVP"
            C: "def2-SVP"
            H: "def2-SVP"

    Returns:
        dict mapping element symbol (str) to basis set name (str).

    Raises:
        ValueError: If a key is not a valid element symbol or a value is
            not a string.
    """
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
    """Apply keyword overrides to a ComputationSettings instance.

    Returns a new ComputationSettings instance with the overrides applied
    (the original is not mutated).

    Special handling:
        * ``basis_set_per_element``: the override dict is *merged* with the
          existing per-element mapping (override values take precedence).
        * ``basis_set_file``: if provided, the YAML file is loaded and its
          contents are merged into the per-element mapping.

    Args:
        settings: base ComputationSettings instance.
        **overrides: keyword arguments matching ComputationSettings fields.

    Returns:
        New ComputationSettings with overrides applied.
    """
    # Extract special keys before passing the rest to dataclasses.replace()
    basis_per_element_override = overrides.pop("basis_set_per_element", None)
    basis_file = overrides.pop("basis_set_file", None)

    # Build the merged per-element dict
    merged_basis = dict(settings.basis_set_per_element)

    if basis_per_element_override is not None:
        merged_basis.update(basis_per_element_override)

    if basis_file is not None:
        file_basis = load_basis_file(basis_file)
        merged_basis.update(file_basis)

    return replace(settings, basis_set_per_element=merged_basis, **overrides)


def load_cas_settings_file(filepath: str) -> dict:
    """Load CAS settings overrides from a YAML file.

    The YAML file may contain any field of ComputationSettings, plus optional
    ``charge``, ``spin``, ``preset``, and ``localization_method`` fields that
    are consumed by the CLI layer, e.g.::

        preset: "default"
        scf_method: "uks"
        xc_functional: "BP86"
        basis_set_default: "def2-TZVP"
        basis_per_element:
          Fe: "def2-TZVP"
          S:  "def2-TZVP"
        init_guess: "huckel"
        scf_damp: 0.3
        scf_level_shift: 0.1
        localization_method: "pm"
        charge: -2
        spin: 0.0

    Commented-out lines are ignored. Only uncommented fields are returned.

    Args:
        filepath: path to YAML file with CAS settings.

    Returns:
        dict of keyword arguments suitable for ``apply_overrides()``.
        Non-ComputationSettings keys (charge, spin, localization_method) are
        passed through as-is for the CLI layer to consume.

    Raises:
        FileNotFoundError: if *filepath* does not exist.
        ValueError: if the YAML content is not a mapping.
    """
    with open(filepath, "r") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {filepath}, "
            f"got {type(data).__name__}"
        )

    # Map YAML key "basis_per_element" to the dataclass field name
    if "basis_per_element" in data:
        data["basis_set_per_element"] = data.pop("basis_per_element")

    return data


# Backward-compatible alias
load_scf_settings_file = load_cas_settings_file


def settings_from_preset(preset_name: str, **overrides) -> ComputationSettings:
    """Get a preset configuration and apply overrides in one call.

    Args:
        preset_name: "default" or "fast".
        **overrides: keyword overrides forwarded to :func:`apply_overrides`.

    Returns:
        Configured ComputationSettings instance.

    Raises:
        KeyError: If *preset_name* is not a known preset.
    """
    if preset_name not in PRESETS:
        raise KeyError(
            f"Unknown preset {preset_name!r}. "
            f"Available presets: {sorted(PRESETS)}"
        )
    base = PRESETS[preset_name]
    return apply_overrides(base, **overrides)


def build_basis_dict(cluster_info, settings: ComputationSettings) -> dict:
    """Build a per-element basis set dict for PySCF from cluster info and settings.

    Collects all unique elements from metals, bridging_atoms, and
    terminal_ligands, then resolves the basis set for each element using
    ``settings.get_basis(element)``.

    Args:
        cluster_info: ClusterInfo object with metals, bridging_atoms, and
            terminal_ligands attributes.
        settings: ComputationSettings with basis configuration.

    Returns:
        dict mapping element symbol to basis set name, e.g.
        ``{"Fe": "def2-TZVP", "S": "def2-TZVP", "H": "def2-SVP"}``.
    """
    elements = set()

    # Collect from metal centers
    for metal in cluster_info.metals:
        elements.add(metal.element)

    # Collect from bridging atoms
    for bridge in cluster_info.bridging_atoms:
        elements.add(bridge.element)

    # Collect from terminal ligands -- iterate over all atom indices
    # and resolve the element from cluster_info.all_elements
    for ligand in cluster_info.terminal_ligands:
        for idx in ligand.atom_indices:
            if (
                cluster_info.all_elements is not None
                and idx < len(cluster_info.all_elements)
            ):
                elements.add(cluster_info.all_elements[idx])

    # Collect from all atoms in the structure to ensure complete coverage
    # (metals, bridging atoms, terminal ligand donors may miss e.g. H in CH3)
    if cluster_info.all_elements:
        elements.update(cluster_info.all_elements)

    # Build the result dict
    return {element: settings.get_basis(element) for element in sorted(elements)}
