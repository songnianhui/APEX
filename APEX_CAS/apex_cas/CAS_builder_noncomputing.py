"""CAS builder based on noncomputing methods

Rule-based CAS determination from ClusterInfo.
Determines which orbitals enter CAS based on metal identity, ligand types, and domain knowledge from the knowledge base.
"""

import os
from itertools import product

import numpy as np
import yaml

from . import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    NonComputingMethod,
    OrbitalGroup,
    NonComputingMethodConfig,
)
from ._paths import data_file as _data_file

# ──────────────────────────────────────────────────────────────────
# Knowledge base loading
# ──────────────────────────────────────────────────────────────────

_metals_db = None
_ligands_db = None
_clusters_db = None


def _load_yaml(filename):
    path = _data_file(filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_metals_db():
    global _metals_db
    if _metals_db is None:
        _metals_db = _load_yaml("transition_metals.yaml")
    return _metals_db


def _get_ligands_db():
    global _ligands_db
    if _ligands_db is None:
        _ligands_db = _load_yaml("ligand_database.yaml")
    return _ligands_db


def _get_clusters_db():
    global _clusters_db
    if _clusters_db is None:
        _clusters_db = _load_yaml("cluster_templates.yaml")
    return _clusters_db

# ──────────────────────────────────────────────────────────────────
# Unified non-computing CAS entry point
# ──────────────────────────────────────────────────────────────────


def build_NC_CAS(
    cluster_info: ClusterInfo,
    level: ActiveSpaceLevel = ActiveSpaceLevel.STANDARD,
    oxidation_states: dict = None,
    non_computing_type: list[NonComputingMethod] = None,
    template_name: str = None,
) -> tuple[dict[str, CAS], list[dict]]:
    """Build CAS using non-computing (rule/data-driven) methods.

    Unified entry point for all non-computing CAS construction strategies.
    Supports running one, two, or all three methods and optionally combining
    the results.

    Args:
        cluster_info: ClusterInfo from structure analysis.
        level: Desired level of active space construction.
        oxidation_states: Optional dict {metal_idx: oxidation_state}.
        non_computing_type: List of NonComputingMethod values to use. Defaults to all three
            (RULE, TOPOLOGY, KNOWLEDGE_BASE).
        template_name: Optional specific template name for knowledge_base method.

    Returns:
        Tuple of (cases dict, expected_types):
        - cases: Dict mapping method name ("rule", "topology", "knowledge_base")
          to its CAS result. If multiple methods are requested, a "combined" entry
          is also included.
        - expected_types: Combined expected orbital types list. If multiple methods
          are used, this is the merged/combined list with "source" tracking.
    """
    if non_computing_type is None:
        non_computing_type = [NonComputingMethod.RULE, NonComputingMethod.TOPOLOGY, NonComputingMethod.KNOWLEDGE_BASE]

    cases: dict[str, CAS] = {}
    expected_types_map: dict[str, list[dict]] = {}

    if NonComputingMethod.RULE in non_computing_type:
        cases["rule"] = _build_from_rule(cluster_info, level, oxidation_states)
        expected_types_map["rule"] = _get_expected_orbital_types(cluster_info, level)

    if NonComputingMethod.TOPOLOGY in non_computing_type:
        cases["topology"] = _build_from_topology(cluster_info, level, oxidation_states)
        expected_types_map["topology"] = _get_expected_orbital_types(
            cluster_info, level
        )

    if NonComputingMethod.KNOWLEDGE_BASE in non_computing_type:
        cases["knowledge_base"] = _build_from_knowledge_base(
            cluster_info,
            level,
            template_name=template_name,
            oxidation_states=oxidation_states,
        )
        expected_types_map["knowledge_base"] = _get_expected_orbital_types(
            cluster_info, level
        )

    # Combine results if multiple methods were requested
    if len(cases) > 1:
        combined_cas, combined_et = _build_combined(cases, expected_types_map, level)
        cases["combined"] = combined_cas
        expected_types = combined_et
    else:
        expected_types = list(expected_types_map.values())[0]

    return cases, expected_types


# ──────────────────────────────────────────────────────────────────
# Oxidation state and electron count helpers
# ──────────────────────────────────────────────────────────────────

def get_local_spin(element: str, oxidation_state: int) -> float:
    """Get the high-spin S value for a given element and oxidation state.

    Args:
        element: Element symbol, e.g. "Fe".
        oxidation_state: Oxidation state, e.g. +2.
    Returns:
        Spin quantum number S.
    """
    db = _get_metals_db()
    if element not in db:
        return 0.0
    key = f"{element}{abs(oxidation_state)}+" if oxidation_state > 0 else f"{element}0"
    states = db[element].get("high_spin_states", {})
    if key in states:
        return states[key]["S"]
    return 0.0

def get_common_oxidation_states(element: str) -> list:
    """Get common oxidation states for a transition metal."""
    db = _get_metals_db()
    if element not in db:
        return []
    return db[element].get("common_oxidation_states", [])

def get_d_electron_count(element: str, oxidation_state: int) -> int:
    """Get the d-electron count for a given element and oxidation state.

    Args:
        element: Element symbol.
        oxidation_state: Oxidation state.

    Returns:
        Number of d electrons.
    """
    db = _get_metals_db()
    if element not in db:
        return 0
    key = f"{element}{abs(oxidation_state)}+" if oxidation_state > 0 else f"{element}0"
    states = db[element].get("high_spin_states", {})
    if key in states:
        return states[key]["d_count"]
    return 0

def get_n_active_orbitals(element: str) -> int:
    """Get number of active orbitals for a transition metal (always 5 for d)."""
    db = _get_metals_db()
    if element not in db:
        return 0
    return db[element].get("n_active_orbitals", 5)

# ──────────────────────────────────────────────────────────────────
# Internal implementation
# ──────────────────────────────────────────────────────────────────


def _build_from_rule(
    cluster_info: ClusterInfo, level: ActiveSpaceLevel, oxidation_states: dict = None
) -> CAS:
    """Core active space builder."""
    # Try to match a known cluster template first
    template = _match_cluster_template(cluster_info)

    if template and level == ActiveSpaceLevel.STANDARD:
        # Use validated data from template if available
        template_as = (
            template.get("active_space", {}).get("LLDUC_model")
            or template.get("active_space", {}).get("minimal")
            or template.get("active_space", {}).get("standard")
        )
        if template_as:
            return _build_from_template(cluster_info, template, template_as, level)

    orbital_groups = []
    n_electrons = 0
    n_orbitals = 0

    # Default oxidation states if not provided
    if oxidation_states is None:
        oxidation_states = _infer_default_oxidation_states(cluster_info)

    # 1. Metal d orbitals
    for k, metal in enumerate(cluster_info.metals):
        ox = oxidation_states.get(k, _default_oxidation(metal.element))
        d_count = get_d_electron_count(metal.element, ox)
        d_orbitals = get_n_active_orbitals(metal.element)

        orbital_groups.append(
            OrbitalGroup(
                atom_label=metal.label,
                orbital_type=_get_orbital_type(metal.element),
                n_orbitals=d_orbitals,
                n_electrons=d_count,
            )
        )
        n_electrons += d_count
        n_orbitals += d_orbitals

    # 2. Bridging atom orbitals (for STANDARD and EXTENDED)
    if level in (ActiveSpaceLevel.STANDARD, ActiveSpaceLevel.EXTENDED):
        ligands_db = _get_ligands_db()
        for bridge in cluster_info.bridging_atoms:
            elem = bridge.element
            if elem in ligands_db:
                elem_data = ligands_db[elem]
                n_orb = elem_data.get(
                    "n_bridging_orbitals", elem_data.get("n_active_orbitals", 3)
                )
                # Estimate electrons in these orbitals
                # S(2-): 6e in 3p, O(2-): 6e in 2p, C(4-): 4e (2s+2p)
                n_elec = _bridging_electron_count(elem, bridge.role)
                orb_type = elem_data.get(
                    "bridging_orbitals", elem_data.get("active_orbitals", ["2p"])
                )

                orbital_groups.append(
                    OrbitalGroup(
                        atom_label=f"{elem}{bridge.index}",
                        orbital_type="+".join(orb_type)
                        if isinstance(orb_type, list)
                        else orb_type,
                        n_orbitals=n_orb,
                        n_electrons=n_elec,
                    )
                )
                n_electrons += n_elec
                n_orbitals += n_orb

    # 3. Terminal ligand donor orbitals (EXTENDED only)
    if level == ActiveSpaceLevel.EXTENDED:
        for ligand in cluster_info.terminal_ligands:
            # Terminal ligands typically contribute 2 electrons from lone pairs
            if ligand.charge != 0 or ligand.name in _get_ligands_db().get(
                "terminal_ligands", {}
            ):
                lig_data = (
                    _get_ligands_db().get("terminal_ligands", {}).get(ligand.name)
                )
                if lig_data:
                    n_elec = lig_data.get("n_donor_electrons", 2)
                    orbital_groups.append(
                        OrbitalGroup(
                            atom_label=ligand.name,
                            orbital_type="donor",
                            n_orbitals=1,
                            n_electrons=n_elec,
                        )
                    )
                    n_electrons += n_elec
                    n_orbitals += 1

    # Build description
    level_desc = {
        ActiveSpaceLevel.MINIMAL: "minimal (metal d only)",
        ActiveSpaceLevel.STANDARD: "standard (metal d + bridging)",
        ActiveSpaceLevel.EXTENDED: "extended (+ ligand donors)",
    }
    desc = f"({n_electrons}e, {n_orbitals}o) {level_desc[level]}"

    return CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=level,
        description=desc,
    )


def _bridging_electron_count(element: str, role: str) -> int:
    """Estimate electron contribution of a bridging/interstitial atom."""
    if element == "S":
        return 6  # S(2-): 3p^6
    elif element == "O":
        return 6  # O(2-): 2p^6
    elif element == "N" and role == "bridging":
        return 6  # N(3-): 2p^6 (as nitride)
    elif element == "N":
        return 4  # amide/donor N
    elif element == "C" and role == "interstitial":
        return 4  # C(4-): 2s^2 2p^2 → 4 electrons in active orbitals
    elif element == "C":
        return 4
    elif element == "Se":
        return 6
    elif element == "Cl":
        return 6
    elif element == "P":
        return 3  # P(3-): 3p^3 → 3 electrons
    elif element == "F":
        return 1  # F(-): typically donates 1 electron to active space
    elif element == "Br":
        return 1
    elif element == "I":
        return 1
    elif element == "H":
        return 1  # hydride bridging: 1 electron
    else:
        return 4  # default


def _infer_default_oxidation_states(cluster_info: ClusterInfo) -> dict:
    """Infer default oxidation states from charge balance.

    Uses the cluster template if available, otherwise attempts charge balance.
    """
    template = _match_cluster_template(cluster_info)
    if template:
        ox_data = template.get("oxidation_states", {})
        if ox_data:
            # Apply template oxidation states
            result = {}
            for k, metal in enumerate(cluster_info.metals):
                if metal.element in ox_data:
                    states = ox_data[metal.element]
                    if isinstance(states, list):
                        # Distribute oxidation states to satisfy charge
                        return _balance_oxidation_states(cluster_info, states)
                    else:
                        result[k] = states
            if result:
                return result

    # Fall back to charge balance
    return _balance_oxidation_states(cluster_info)


def _balance_oxidation_states(
    cluster_info: ClusterInfo, allowed_states: list = None
) -> dict:
    """Determine oxidation states that satisfy charge balance.

    Uses a constraint-satisfaction approach: find combinations of oxidation
    states that make Σ(metal_ox) + Σ(ligand_charges) = total_charge.
    """
    metals = cluster_info.metals
    n_metals = len(metals)

    if n_metals == 0:
        return {}

    # Estimate ligand charge contribution
    ligand_charge = _estimate_ligand_charge(cluster_info)

    # Required sum of metal oxidation states
    target_metal_charge_sum = cluster_info.total_charge - ligand_charge

    # Get allowed oxidation states for each metal
    metals_db = _get_metals_db()
    options = []
    for metal in metals:
        if allowed_states and metal.element in [m.element for m in metals]:
            opts = [
                s
                for s in get_common_oxidation_states(metal.element)
                if s in allowed_states
            ]
        else:
            opts = get_common_oxidation_states(metal.element)
        if not opts:
            # Fallback for truly unknown metals (not in knowledge base)
            opts = [2, 3]
        options.append(opts)

    # For small systems, enumerate; for large, use heuristics
    if n_metals <= 8:
        return _solve_oxidation_csp(options, target_metal_charge_sum, metals)
    else:
        return _heuristic_oxidation(options, target_metal_charge_sum, metals)


def _solve_oxidation_csp(options, target_sum, metals):
    """Solve oxidation state assignment as constraint satisfaction problem."""
    # Try all combinations for small systems
    best = None
    best_dev = float("inf")

    for combo in product(*options):
        s = sum(combo)
        dev = abs(s - target_sum)
        if dev < best_dev:
            best_dev = dev
            best = combo
            if dev == 0:
                break

    if best is None:
        return {k: options[k][0] for k in range(len(metals))}

    return {k: best[k] for k in range(len(metals))}


def _heuristic_oxidation(options, target_sum, metals):
    """Heuristic oxidation state assignment for large systems."""
    n = len(metals)
    result = {}
    remaining = target_sum

    for k, metal in enumerate(metals):
        opts = sorted(options[k])
        if k < n - 1:
            # Pick the most common oxidation state
            result[k] = opts[len(opts) // 2]
            remaining -= result[k]
        else:
            # Last metal: pick closest to remaining
            result[k] = min(opts, key=lambda x: abs(x - remaining))

    return result


def _estimate_ligand_charge(cluster_info: ClusterInfo) -> int:
    """Estimate the total charge contributed by ligands."""
    ligands_db = _get_ligands_db()
    charge = 0

    # Bridging atoms
    for bridge in cluster_info.bridging_atoms:
        elem = bridge.element
        if elem in ligands_db:
            charges = ligands_db[elem].get("common_charges", [-2])
            charge += charges[0]  # most common charge

    # Terminal ligands
    for lig in cluster_info.terminal_ligands:
        charge += lig.charge

    return charge


def _default_oxidation(element: str) -> int:
    """Return the most common oxidation state for an element."""
    states = get_common_oxidation_states(element)
    return states[0] if states else 2


def _match_cluster_template(cluster_info: ClusterInfo) -> dict:
    """Try to match the cluster to a known template in the knowledge base."""
    clusters_db = _get_clusters_db()
    formula = cluster_info.formula

    for name, data in clusters_db.items():
        core = data.get("core_formula", "")
        if core and core in formula:
            return data
        # Also check charge and spin match
        if (
            data.get("total_charge") == cluster_info.total_charge
            and data.get("ground_state_S") == cluster_info.target_spin
        ):
            core_elems = _parse_formula(core)
            formula_elems = _parse_formula(formula)
            if all(formula_elems.get(e, 0) >= core_elems.get(e, 0) for e in core_elems):
                return data

    return None


def _parse_formula(formula: str) -> dict:
    """Parse a chemical formula into {element: count}."""
    import re

    counts = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        counts[elem] = counts.get(elem, 0) + count
    return counts


def _build_from_template(cluster_info, template, template_as, level):
    """Build CAS from a template's validated data."""
    n_elec = template_as["n_electrons"]
    n_orb = template_as["n_orbitals"]

    # Build orbital groups from metals and bridging atoms
    orbital_groups = []

    for metal in cluster_info.metals:
        ox_states = template.get("oxidation_states", {})
        if metal.element in ox_states:
            states = ox_states[metal.element]
            if isinstance(states, list):
                ox = states[0]
            else:
                ox = states
        else:
            ox = _default_oxidation(metal.element)

        d_count = get_d_electron_count(metal.element, ox)
        orbital_groups.append(
            OrbitalGroup(
                atom_label=metal.label,
                orbital_type=_get_orbital_type(metal.element),
                n_orbitals=5,
                n_electrons=d_count,
            )
        )

    for bridge in cluster_info.bridging_atoms:
        n_elec_b = _bridging_electron_count(bridge.element, bridge.role)
        ligands_db = _get_ligands_db()
        elem_data = ligands_db.get(bridge.element, {})
        orb_type = elem_data.get("bridging_orbitals", ["3p"])
        orbital_groups.append(
            OrbitalGroup(
                atom_label=f"{bridge.element}{bridge.index}",
                orbital_type="+".join(orb_type)
                if isinstance(orb_type, list)
                else orb_type,
                n_orbitals=elem_data.get(
                    "n_bridging_orbitals", elem_data.get("n_active_orbitals", 3)
                ),
                n_electrons=n_elec_b,
            )
        )

    desc = template_as.get("description", f"({n_elec}e, {n_orb}o)")
    ref = template_as.get("reference", "")
    if ref:
        desc += f" [{ref}]"

    return CAS(
        n_electrons=n_elec,
        n_orbitals=n_orb,
        orbital_groups=orbital_groups,
        level=level,
        description=desc,
    )


def _get_3d_metals():
    return {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}


def _get_4d_metals():
    return {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}


def _get_5d_metals():
    return {"La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}


def _get_orbital_type(element: str) -> str:
    """Return the active orbital type label (3d, 4d, or 5d) for an element."""
    if element in _get_3d_metals():
        return "3d"
    elif element in _get_4d_metals():
        return "4d"
    elif element in _get_5d_metals():
        return "5d"
    else:
        # Unknown element: try to infer from knowledge base
        db = _get_metals_db()
        if element in db:
            row = db[element].get("row", "")
            if "5d" in row:
                return "5d"
            elif "4d" in row:
                return "4d"
        return "4d"  # fallback


# ──────────────────────────────────────────────────────────────────
# Stage 1 extended methods
# ──────────────────────────────────────────────────────────────────


def _get_expected_orbital_types(
    cluster_info: ClusterInfo, level: ActiveSpaceLevel
) -> list[dict]:
    """Return a list of expected orbital type dicts for Stage 2 chemical character selection.

    Each dict has keys:
        atom_label: str   (e.g., "Fe1")
        element: str      (e.g., "Fe")
        ao_type: str      (e.g., "3d", "3p", "2s2p")
        n_expected: int
        priority: str     ("required" or "supplementary")

    Args:
        cluster_info: ClusterInfo from structure analysis.
        level: Active space level being built.

    Returns:
        List of orbital type descriptor dicts.
    """
    expected: list[dict] = []
    ligands_db = _get_ligands_db()

    # 1. Metals: always required, 5 d orbitals each
    for metal in cluster_info.metals:
        expected.append(
            {
                "atom_label": metal.label,
                "element": metal.element,
                "ao_type": _get_orbital_type(metal.element),
                "n_expected": 5,
                "priority": "required",
            }
        )

    # 2. Bridging atoms: required for STANDARD and EXTENDED
    if level in (ActiveSpaceLevel.STANDARD, ActiveSpaceLevel.EXTENDED):
        for bridge in cluster_info.bridging_atoms:
            elem = bridge.element

            if bridge.role == "interstitial":
                # Interstitial atoms: add both s and p orbitals as supplementary
                elem_data = ligands_db.get(elem, {})
                active_orbs = elem_data.get("active_orbitals", ["2s", "2p"])
                n_active = elem_data.get("n_active_orbitals", 4)

                ao_type = (
                    "+".join(active_orbs)
                    if isinstance(active_orbs, list)
                    else active_orbs
                )
                expected.append(
                    {
                        "atom_label": f"{elem}{bridge.index}",
                        "element": elem,
                        "ao_type": ao_type,
                        "n_expected": n_active,
                        "priority": "supplementary",
                    }
                )
            else:
                # Regular bridging atoms: required
                elem_data = ligands_db.get(elem, {})
                bridging_orbs = elem_data.get("bridging_orbitals", ["3p"])
                if elem in ("S", "Se", "Cl"):
                    default_ao = "3p"
                elif elem in ("O", "N"):
                    default_ao = "2p"
                elif elem in ("P",):
                    default_ao = "3p"
                else:
                    default_ao = "3p"
                ao_type = (
                    "+".join(bridging_orbs)
                    if isinstance(bridging_orbs, list)
                    else bridging_orbs
                )
                n_bridging = elem_data.get(
                    "n_bridging_orbitals", elem_data.get("n_active_orbitals", 3)
                )
                expected.append(
                    {
                        "atom_label": f"{elem}{bridge.index}",
                        "element": elem,
                        "ao_type": ao_type,
                        "n_expected": n_bridging,
                        "priority": "required",
                    }
                )

    # 3. Terminal ligands: supplementary for EXTENDED only
    if level == ActiveSpaceLevel.EXTENDED:
        for ligand in cluster_info.terminal_ligands:
            lig_data = ligands_db.get("terminal_ligands", {}).get(ligand.name)
            if lig_data:
                donor = lig_data.get("donor_atom", "X")
                expected.append(
                    {
                        "atom_label": ligand.name,
                        "element": donor,
                        "ao_type": "donor",
                        "n_expected": 1,
                        "priority": "supplementary",
                    }
                )

    return expected


def _build_from_topology(
    cluster_info: ClusterInfo, level: ActiveSpaceLevel, oxidation_states: dict = None
) -> CAS:
    """Topology-aware active space construction.

    Starts from the rule-based result and adjusts based on coordination
    geometry and bond-length covalency analysis.

    Args:
        cluster_info: ClusterInfo from structure analysis.
        level: Desired level of active space construction.
        oxidation_states: Optional dict {metal_idx: oxidation_state}.

    Returns:
        CAS with topology-aware orbital groups.
    """
    # 1. Start from the rule-based result as baseline
    baseline = _build_from_rule(cluster_info, level, oxidation_states)

    # Default oxidation states if not provided
    if oxidation_states is None:
        oxidation_states = _infer_default_oxidation_states(cluster_info)

    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()

    orbital_groups = []
    n_electrons = 0
    n_orbitals = 0
    geometry_flags: list[str] = []

    # 2. Metal d orbitals with topology-aware adjustments
    for k, metal in enumerate(cluster_info.metals):
        ox = oxidation_states.get(k, _default_oxidation(metal.element))
        d_count = get_d_electron_count(metal.element, ox)
        d_orbitals = get_n_active_orbitals(metal.element)

        coord = metal.coordination
        if coord == 4:
            # Tetrahedral: all 5 d orbitals active (no splitting adjustment)
            pass
        elif coord == 6:
            # Octahedral: crystal field splitting, still all 5 d orbitals
            pass
        elif coord in (3, 5):
            # Unusual geometry
            geometry_flags.append(f"{metal.label}: unusual coordination {coord}")

        orbital_groups.append(
            OrbitalGroup(
                atom_label=metal.label,
                orbital_type=_get_orbital_type(metal.element),
                n_orbitals=d_orbitals,
                n_electrons=d_count,
            )
        )
        n_electrons += d_count
        n_orbitals += d_orbitals

    # 3. Bridging atoms with covalency estimate from distance
    if level in (ActiveSpaceLevel.STANDARD, ActiveSpaceLevel.EXTENDED):
        for bridge in cluster_info.bridging_atoms:
            elem = bridge.element
            elem_data = ligands_db.get(elem, {})
            n_orb = elem_data.get(
                "n_bridging_orbitals", elem_data.get("n_active_orbitals", 3)
            )
            n_elec = _bridging_electron_count(elem, bridge.role)
            orb_type = elem_data.get(
                "bridging_orbitals", elem_data.get("active_orbitals", ["2p"])
            )

            # Estimate covalency from distance to bridged metals
            bridge_pos = bridge.position
            metal_cov_radius = 1.32  # default fallback
            ligand_cov_radius = elem_data.get("covalent_radius_angstrom", 1.0)

            strongly_covalent = False
            for metal_idx in bridge.bridged_metals:
                if metal_idx < len(cluster_info.metals):
                    metal = cluster_info.metals[metal_idx]
                    metal_pos = metal.position
                    dist = np.linalg.norm(bridge_pos - metal_pos)

                    # Get metal covalent radius from knowledge base
                    m_data = metals_db.get(metal.element, {})
                    metal_cov_radius = m_data.get(
                        "covalent_radius_angstrom", metal_cov_radius
                    )

                    sum_radii = metal_cov_radius + ligand_cov_radius
                    tolerance = NonComputingMethodConfig().bond_tolerance
                    threshold = sum_radii * tolerance

                    if dist < threshold:
                        strongly_covalent = True

            # Strongly covalent bonds may need additional correlating orbitals
            if strongly_covalent:
                # Add one extra correlating orbital for strong covalency
                n_orb += 1
                n_elec += 1

            orbital_groups.append(
                OrbitalGroup(
                    atom_label=f"{elem}{bridge.index}",
                    orbital_type="+".join(orb_type)
                    if isinstance(orb_type, list)
                    else orb_type,
                    n_orbitals=n_orb,
                    n_electrons=n_elec,
                )
            )
            n_electrons += n_elec
            n_orbitals += n_orb

    # 4. Terminal ligand donor orbitals (EXTENDED only)
    if level == ActiveSpaceLevel.EXTENDED:
        for ligand in cluster_info.terminal_ligands:
            lig_data = ligands_db.get("terminal_ligands", {}).get(ligand.name)
            if lig_data:
                n_elec = lig_data.get("n_donor_electrons", 2)
                orbital_groups.append(
                    OrbitalGroup(
                        atom_label=ligand.name,
                        orbital_type="donor",
                        n_orbitals=1,
                        n_electrons=n_elec,
                    )
                )
                n_electrons += n_elec
                n_orbitals += 1

    # 5. Build CAS with topology-adjusted parameters
    level_desc = {
        ActiveSpaceLevel.MINIMAL: "minimal (topology-aware)",
        ActiveSpaceLevel.STANDARD: "standard (topology-aware)",
        ActiveSpaceLevel.EXTENDED: "extended (topology-aware)",
    }
    desc = f"({n_electrons}e, {n_orbitals}o) {level_desc[level]}"
    if geometry_flags:
        desc += f" [flags: {'; '.join(geometry_flags)}]"

    active_space = CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=level,
        description=desc,
    )

    return active_space


def _build_from_knowledge_base(
    cluster_info: ClusterInfo,
    level: ActiveSpaceLevel,
    template_name: str = None,
    oxidation_states: dict = None,
) -> CAS:
    """Knowledge-base-driven active space construction.

    Uses transition_metals.yaml, ligand_database.yaml, and cluster_templates.yaml
    to build the active space from curated data.

    Args:
        cluster_info: ClusterInfo from structure analysis.
        level: Desired level of active space construction.
        template_name: Optional specific template name to look up directly.
        oxidation_states: Optional dict {metal_idx: oxidation_state}.

    Returns:
        CAS with knowledge-base-driven orbital groups.
    """
    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()
    clusters_db = _get_clusters_db()

    # Default oxidation states if not provided
    if oxidation_states is None:
        oxidation_states = _infer_default_oxidation_states(cluster_info)

    # 1. Try to match a cluster template
    template = None
    if template_name is not None:
        template = clusters_db.get(template_name)

    if template is None:
        template = _match_cluster_template(cluster_info)

    orbital_groups = []
    n_electrons = 0
    n_orbitals = 0

    # 2. Metal d orbitals from knowledge base
    for k, metal in enumerate(cluster_info.metals):
        elem = metal.element
        ox = oxidation_states.get(k, _default_oxidation(elem))
        d_count = get_d_electron_count(elem, ox)

        # Query transition_metals.yaml for active_orbitals field
        elem_data = metals_db.get(elem, {})
        active_orbs = elem_data.get("active_orbitals", [_get_orbital_type(elem)])
        d_orbitals = elem_data.get("n_active_orbitals", 5)

        orbital_groups.append(
            OrbitalGroup(
                atom_label=metal.label,
                orbital_type="+".join(active_orbs)
                if isinstance(active_orbs, list)
                else active_orbs,
                n_orbitals=d_orbitals,
                n_electrons=d_count,
            )
        )
        n_electrons += d_count
        n_orbitals += d_orbitals

    # 3. Bridging atoms from knowledge base
    if level in (ActiveSpaceLevel.STANDARD, ActiveSpaceLevel.EXTENDED):
        for bridge in cluster_info.bridging_atoms:
            elem = bridge.element
            n_elec = _bridging_electron_count(elem, bridge.role)

            # Query ligand_database.yaml for bridging_orbitals
            elem_data = ligands_db.get(elem, {})
            orb_type = elem_data.get(
                "bridging_orbitals", elem_data.get("active_orbitals", ["2p"])
            )
            n_orb = elem_data.get(
                "n_bridging_orbitals", elem_data.get("n_active_orbitals", 3)
            )

            orbital_groups.append(
                OrbitalGroup(
                    atom_label=f"{elem}{bridge.index}",
                    orbital_type="+".join(orb_type)
                    if isinstance(orb_type, list)
                    else orb_type,
                    n_orbitals=n_orb,
                    n_electrons=n_elec,
                )
            )
            n_electrons += n_elec
            n_orbitals += n_orb

    # 4. Terminal ligand donor orbitals (EXTENDED only)
    if level == ActiveSpaceLevel.EXTENDED:
        for ligand in cluster_info.terminal_ligands:
            lig_data = ligands_db.get("terminal_ligands", {}).get(ligand.name)
            if lig_data:
                n_elec = lig_data.get("n_donor_electrons", 2)
                orbital_groups.append(
                    OrbitalGroup(
                        atom_label=ligand.name,
                        orbital_type="donor",
                        n_orbitals=1,
                        n_electrons=n_elec,
                    )
                )
                n_electrons += n_elec
                n_orbitals += 1

    # 5. If a template match with validated CAS parameters, use those counts
    desc_suffix = "knowledge-base"
    if template:
        # Determine which active space entry to use based on level
        as_data = template.get("active_space", {})
        template_as = None
        if level == ActiveSpaceLevel.MINIMAL:
            template_as = as_data.get("minimal")
        elif level == ActiveSpaceLevel.STANDARD:
            template_as = (
                as_data.get("LLDUC_model")
                or as_data.get("with_sulfur")
                or as_data.get("with_oxo")
                or as_data.get("standard")
            )
        elif level == ActiveSpaceLevel.EXTENDED:
            template_as = as_data.get("extended_model") or as_data.get("extended")

        if template_as:
            n_electrons = template_as["n_electrons"]
            n_orbitals = template_as["n_orbitals"]
            desc_suffix = (
                f"knowledge-base [template: {template_as.get('description', '')}]"
            )

    level_desc = {
        ActiveSpaceLevel.MINIMAL: "minimal",
        ActiveSpaceLevel.STANDARD: "standard",
        ActiveSpaceLevel.EXTENDED: "extended",
    }
    desc = f"({n_electrons}e, {n_orbitals}o) {level_desc[level]} ({desc_suffix})"

    active_space = CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=level,
        description=desc,
    )

    return active_space


def _build_combined(
    cases: dict[str, CAS],
    expected_types_map: dict[str, list[dict]],
    level: ActiveSpaceLevel,
) -> tuple[CAS, list[dict]]:
    """Combine multiple non-computing CAS results via union approach.

    Takes maximum n_orbitals and n_electrons across all methods, and merges
    expected_types lists with deduplication.

    Args:
        cases: Dict mapping method name to its CAS result.
            e.g. {"rule": CAS(...), "topology": CAS(...), "knowledge_base": CAS(...)}
        expected_types_map: Dict mapping method name to its expected orbital types.
        level: Active space level.

    Returns:
        Tuple of (combined CAS, combined expected_types) where each expected_type
        entry has an additional "source" field noting which method contributed it.
    """
    # Take maximum n_orbitals and n_electrons across all methods (union)
    all_cases = list(cases.values())
    n_electrons = max(cas.n_electrons for cas in all_cases)
    n_orbitals = max(cas.n_orbitals for cas in all_cases)

    # Merge orbital groups: collect unique groups by (atom_label, orbital_type)
    merged_groups: dict[tuple, OrbitalGroup] = {}
    for cas_obj in all_cases:
        for og in cas_obj.orbital_groups:
            key = (og.atom_label, og.orbital_type)
            if key in merged_groups:
                existing = merged_groups[key]
                existing.n_orbitals = max(existing.n_orbitals, og.n_orbitals)
                existing.n_electrons = max(existing.n_electrons, og.n_electrons)
            else:
                merged_groups[key] = OrbitalGroup(
                    atom_label=og.atom_label,
                    orbital_type=og.orbital_type,
                    n_orbitals=og.n_orbitals,
                    n_electrons=og.n_electrons,
                )

    orbital_groups = list(merged_groups.values())

    # Merge expected_types, deduplicating by (atom_label, ao_type)
    # and adding a "source" field
    merged_expected: dict[tuple, dict] = {}

    for method_name, entries in expected_types_map.items():
        for entry in entries:
            key = (entry["atom_label"], entry["ao_type"])
            if key not in merged_expected:
                merged_expected[key] = dict(entry)
                merged_expected[key]["source"] = [method_name]
            else:
                if method_name not in merged_expected[key]["source"]:
                    merged_expected[key]["source"].append(method_name)

    combined_expected = list(merged_expected.values())

    # Build the merged CAS
    level_desc = {
        ActiveSpaceLevel.MINIMAL: "minimal",
        ActiveSpaceLevel.STANDARD: "standard",
        ActiveSpaceLevel.EXTENDED: "extended",
    }

    # Build description with per-method breakdown
    method_summaries = ", ".join(
        f"{name}={cas.n_electrons}e/{cas.n_orbitals}o" for name, cas in cases.items()
    )
    desc = f"({n_electrons}e, {n_orbitals}o) {level_desc[level]} combined [{method_summaries}]"

    active_space = CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=level,
        description=desc,
    )

    return active_space, combined_expected
