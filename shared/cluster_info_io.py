"""Shared loader for explicit cluster-info annotations.

The goal is to let users provide chemically meaningful annotations without
requiring one-line-per-atom editing for large systems. The supported merge
layers are:

1. ``defaults``: role-based defaults applied after roles are known
2. ``groups``: ordered batch assignments selected by simple predicates
3. ``atoms``: final per-atom overrides
"""

from __future__ import annotations

import os
from copy import deepcopy

import yaml


_CLUSTER_META_KEYS = {
    "total_charge",
    "target_spin",
    "symmetry_group",
    "reduction_symmetry",
    "family_scheme",
    "benchmark_profile",
    "config_reduction_mode",
}


def load_cluster_info_yaml(
    filepath: str,
    *,
    elements: list[str] | None = None,
) -> dict:
    """Load and normalize a cluster_info.yaml file.

    Returns a dict with three top-level keys:
    - ``cluster``: cluster-level metadata
    - ``defaults``: lightweight role-based defaults
    - ``atom_annotations``: merged per-atom annotation map keyed by atom index
    """
    with open(filepath, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {filepath}, "
            f"got {type(raw).__name__}"
        )

    cluster = dict(raw.get("cluster") or {})
    defaults = dict(raw.get("defaults") or {})
    groups = list(raw.get("groups") or [])
    atoms = list(raw.get("atoms") or [])

    for key in cluster:
        if key not in _CLUSTER_META_KEYS:
            raise ValueError(f"Unsupported cluster_info cluster key: {key!r}")

    n_atoms = len(elements) if elements is not None else None
    annotations = _initialize_annotations(elements)
    _apply_group_rules(annotations, groups, n_atoms=n_atoms)
    _apply_atom_overrides(annotations, atoms, n_atoms=n_atoms)
    _apply_role_defaults(annotations, defaults)
    _validate_annotations(annotations, elements)

    return {
        "cluster": cluster,
        "defaults": defaults,
        "atom_annotations": annotations,
        "source_path": os.path.abspath(filepath),
    }


def _initialize_annotations(elements: list[str] | None) -> dict[int, dict]:
    annotations: dict[int, dict] = {}
    if elements is None:
        return annotations
    for idx, elem in enumerate(elements):
        annotations[idx] = {
            "atom_index": idx,
            "element": elem,
        }
    return annotations


def _apply_group_rules(annotations: dict[int, dict], groups: list[dict], *, n_atoms: int | None):
    for rule in groups:
        if not isinstance(rule, dict):
            raise ValueError("Each cluster_info group rule must be a mapping")
        selector = dict(rule.get("selector") or {})
        assign = dict(rule.get("assign") or {})
        if not assign:
            continue

        for atom_idx, annotation in annotations.items():
            if _matches_selector(annotation, selector, n_atoms=n_atoms):
                annotation.update(deepcopy(assign))


def _apply_atom_overrides(annotations: dict[int, dict], atoms: list[dict], *, n_atoms: int | None):
    for atom in atoms:
        if not isinstance(atom, dict):
            raise ValueError("Each cluster_info atoms entry must be a mapping")
        if "atom_index" not in atom:
            raise ValueError("cluster_info atom entry is missing 'atom_index'")
        idx = int(atom["atom_index"])
        if n_atoms is not None and not (0 <= idx < n_atoms):
            raise ValueError(
                f"cluster_info atom_index {idx} out of range for structure with {n_atoms} atoms"
            )
        if idx not in annotations:
            annotations[idx] = {"atom_index": idx}
        annotations[idx].update(deepcopy(atom))


def _apply_role_defaults(annotations: dict[int, dict], defaults: dict):
    charge_by_role = dict(defaults.get("charge_by_role") or {})
    projection_role_by_role = dict(defaults.get("projection_role_by_role") or {})

    for annotation in annotations.values():
        role = annotation.get("role")
        if not role:
            continue
        if "charge" not in annotation and role in charge_by_role:
            annotation["charge"] = charge_by_role[role]
        if "projection_role" not in annotation and role in projection_role_by_role:
            annotation["projection_role"] = projection_role_by_role[role]


def _matches_selector(annotation: dict, selector: dict, *, n_atoms: int | None) -> bool:
    if not selector:
        return False

    if "atom_index" in selector and int(selector["atom_index"]) != int(annotation.get("atom_index", -1)):
        return False

    if "atom_indices" in selector:
        atom_indices = {int(v) for v in selector["atom_indices"]}
        if int(annotation.get("atom_index", -1)) not in atom_indices:
            return False

    if "element" in selector and selector["element"] != annotation.get("element"):
        return False

    if "role" in selector and selector["role"] != annotation.get("role"):
        return False

    if "label" in selector and selector["label"] != annotation.get("label"):
        return False

    return True


def _validate_annotations(annotations: dict[int, dict], elements: list[str] | None):
    if elements is None:
        return

    for idx, annotation in annotations.items():
        expected = elements[idx]
        actual = annotation.get("element", expected)
        if actual != expected:
            raise ValueError(
                f"cluster_info element mismatch at atom {idx}: "
                f"expected {expected!r} from structure, got {actual!r}"
            )


def resolve_cluster_metadata(
    cluster_payload: dict | None,
    *,
    total_charge: int,
    target_spin: float,
    symmetry_group: str,
    reduction_symmetry: str | None,
    family_scheme: str,
    benchmark_profile: str,
    config_reduction_mode: str,
) -> dict:
    """Merge explicit cluster metadata over caller-provided values."""
    payload = dict(cluster_payload or {})
    return {
        "total_charge": payload.get("total_charge", total_charge),
        "target_spin": payload.get("target_spin", target_spin),
        "symmetry_group": payload.get("symmetry_group", symmetry_group),
        "reduction_symmetry": payload.get("reduction_symmetry", reduction_symmetry),
        "family_scheme": payload.get("family_scheme", family_scheme),
        "benchmark_profile": payload.get("benchmark_profile", benchmark_profile),
        "config_reduction_mode": payload.get("config_reduction_mode", config_reduction_mode),
    }
