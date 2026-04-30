"""Shared chemistry knowledge access for APEX.

This module centralizes:
- YAML-backed knowledge-base loading
- transition-metal oxidation / spin / d-electron lookup
- cluster template matching
- small chemistry helpers reused across packages

It is intentionally separate from ``shared.element_data``:
- ``element_data`` holds static element metadata and shell heuristics
- ``chem_knowledge`` holds YAML/template-driven chemistry knowledge
"""

from __future__ import annotations

import os
import re

import yaml

from .element_data import get_metal_row
from .models import ClusterInfo

_SHARED_KB = os.path.abspath(os.path.join(os.path.dirname(__file__), "knowledge_base"))

_metals_db = None
_ligands_db = None
_clusters_db = None


def _data_file(filename: str) -> str:
    return os.path.join(_SHARED_KB, filename)


def _load_yaml(filename: str):
    path = _data_file(filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_metals_db():
    global _metals_db
    if _metals_db is None:
        _metals_db = _load_yaml("transition_metals.yaml")
    return _metals_db


def get_ligands_db():
    global _ligands_db
    if _ligands_db is None:
        _ligands_db = _load_yaml("ligand_database.yaml")
    return _ligands_db


def get_cluster_templates_db():
    global _clusters_db
    if _clusters_db is None:
        _clusters_db = _load_yaml("cluster_templates.yaml")
    return _clusters_db


def get_local_spin(element: str, oxidation_state: int) -> float:
    """Return the high-spin S value for an element/oxidation-state pair."""
    db = get_metals_db()
    if element not in db:
        return 0.0
    key = f"{element}{abs(oxidation_state)}+" if oxidation_state > 0 else f"{element}0"
    states = db[element].get("high_spin_states", {})
    if key in states:
        return states[key]["S"]
    return 0.0


def get_common_oxidation_states(element: str) -> list:
    """Return common oxidation states for a transition metal."""
    db = get_metals_db()
    if element not in db:
        return []
    return db[element].get("common_oxidation_states", [])


def get_d_electron_count(element: str, oxidation_state: int) -> int:
    """Return the d-electron count for an element/oxidation-state pair."""
    db = get_metals_db()
    if element not in db:
        return 0
    key = f"{element}{abs(oxidation_state)}+" if oxidation_state > 0 else f"{element}0"
    states = db[element].get("high_spin_states", {})
    if key in states:
        return states[key]["d_count"]
    return 0


def get_n_active_orbitals(element: str) -> int:
    """Return the default number of active valence orbitals for a metal."""
    db = get_metals_db()
    if element not in db:
        return 0
    return db[element].get("n_active_orbitals", 5)


_VALENCE_S_SHELL = {
    "3d": "4s",
    "4d": "5s",
    "5d": "6s",
}


def get_valence_s_orbital(element: str) -> str | None:
    """Return the valence s-shell label for a transition metal."""
    db = get_metals_db()
    row = ""
    if element in db:
        row = db[element].get("row", "")
    if not row:
        row = get_metal_row(element)
    return _VALENCE_S_SHELL.get(row)


def _parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        counts[elem] = counts.get(elem, 0) + count
    return counts


def match_cluster_template(cluster_info: ClusterInfo) -> dict | None:
    """Return the first cluster template matching formula/charge/spin metadata."""
    clusters_db = get_cluster_templates_db()
    formula = cluster_info.formula

    for _, data in clusters_db.items():
        core = data.get("core_formula", "")
        if core and core in formula:
            return data
        if (
            data.get("total_charge") == cluster_info.total_charge
            and data.get("ground_state_S") == cluster_info.target_spin
        ):
            core_elems = _parse_formula(core)
            formula_elems = _parse_formula(formula)
            if all(formula_elems.get(e, 0) >= core_elems.get(e, 0) for e in core_elems):
                return data

    return None
