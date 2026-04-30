"""Module 4: Electronic Configuration Generator

For each spin isomer, enumerate oxidation state assignments and d-orbital
occupancies. The Cartesian product of spin isomers × oxidation assignments ×
d-orbital choices gives the full set of electronic configurations.

Reference: FeMo-co yields 35 spin isomers × 18 Fe(II)/Fe(III) assignments
× 5 d-orbital choices per Fe(II) = 78,750 total UHF initial guesses.
"""

import numpy as np
from itertools import combinations, permutations, product

from .models import (
    ClusterInfo,
    ElectronicConfig,
    OxidationAssignment,
    SpinIsomer,
)
from apex_cas.CAS_builder_noncomputing import (
    get_common_oxidation_states,
    get_d_electron_count,
    get_local_spin,
)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def enumerate_oxidation_assignments(cluster_info: ClusterInfo,
                                     spin_isomer: SpinIsomer = None,
                                     allowed_oxidations: dict = None) -> list[OxidationAssignment]:
    """Enumerate valid oxidation state assignments for the metal centers.

    Constraints:
    - sum(metal_oxidation_states) + ligand_charge = total_charge
    - Each metal uses one of its common oxidation states
    - If spin_isomer is given, the spin assignment constrains which
      oxidation states are compatible (e.g., minority-spin Fe(III) in a
      high-spin context)

    Args:
        cluster_info: Cluster description.
        spin_isomer: Optional spin isomer for context.
        allowed_oxidations: Optional dict {element: [allowed_states]} to override KB.

    Returns:
        List of OxidationAssignment objects.
    """
    metals = cluster_info.metals
    n_metals = len(metals)
    if n_metals == 0:
        return []

    # Ligand charge contribution
    ligand_charge = _estimate_ligand_charge(cluster_info)

    # Target sum of metal oxidation states
    target_sum = cluster_info.total_charge - ligand_charge

    # Get allowed oxidation states per metal
    options = []
    for metal in metals:
        if allowed_oxidations and metal.element in allowed_oxidations:
            opts = allowed_oxidations[metal.element]
        else:
            opts = get_common_oxidation_states(metal.element)
        if not opts:
            opts = [2, 3]
        options.append(opts)

    # Enumerate all combinations and sort by distance to target_sum
    candidates = []
    for combo in product(*options):
        distance = abs(sum(combo) - target_sum)
        desc = _describe_oxidation_assignment(combo, metals)
        candidates.append((distance, OxidationAssignment(
            assignments={k: combo[k] for k in range(n_metals)},
            description=desc,
        )))

    if not candidates:
        return []

    # Exact matches first
    exact = [oa for d, oa in candidates if d < 1e-6]
    if exact:
        return exact

    # Fall back to closest matches sorted by distance (ascending)
    candidates.sort(key=lambda x: x[0])
    min_dist = candidates[0][0]

    # Return all matches at the minimum distance
    closest = [oa for d, oa in candidates if abs(d - min_dist) < 1e-6]

    if min_dist > 2:
        import logging
        logging.getLogger(__name__).warning(
            "No exact oxidation state match found. "
            "Closest match has distance %.1f from target charge balance. "
            "Returning %d closest assignment(s).",
            min_dist, len(closest)
        )

    return closest


def enumerate_d_orbital_configs(metal_element: str,
                                 oxidation_state: int,
                                 spin_direction: int,
                                 n_d_orbitals: int = 5) -> list[int]:
    """Enumerate which d-orbital hosts the extra electron for partially-filled shells.

    For a high-spin d6 configuration (Fe(II)), 5 orbitals are singly occupied
    and the 6th electron has 5 choices of which orbital to doubly occupy.

    Args:
        metal_element: Element symbol.
        oxidation_state: Oxidation state.
        spin_direction: +1 (majority) or -1 (minority).
        n_d_orbitals: Number of d orbitals (typically 5).

    Returns:
        List of d-orbital indices (0-based) where extra electron can go.
        Empty list if no choice (fully determined by high-spin filling).
    """
    d_count = get_d_electron_count(metal_element, oxidation_state)
    S = get_local_spin(metal_element, oxidation_state)
    n_unpaired = int(2 * S)

    if d_count == 0 or d_count == 10:
        # Empty or full shell — no choice
        return []

    # High-spin filling: first fill all orbitals singly (min(n_d, d_count)),
    # then pair from the first orbital
    n_singly_occupied = min(n_d_orbitals, d_count)
    n_pairs = d_count - n_singly_occupied  # number of doubly-occupied orbitals

    if n_pairs == 0:
        # All electrons are unpaired, no orbital choice
        return []

    # The "extra" electron (relative to half-filling) goes into one of
    # the singly-occupied orbitals. For d6 (Fe(II)): 5 singly occupied + 1 pair
    # The pair can be in any of the 5 orbitals → 5 choices
    n_choices = n_singly_occupied

    if spin_direction == -1:
        # Minority spin: the extra electron is in the beta channel
        # It pairs with one of the alpha electrons
        return list(range(n_choices))
    else:
        # Majority spin direction: same logic
        return list(range(n_choices))


def generate_all_configs(spin_isomers: list[SpinIsomer],
                          cluster_info: ClusterInfo,
                          max_configs: int = None) -> list[ElectronicConfig]:
    """Generate all electronic configurations (Cartesian product).

    For each spin isomer × oxidation assignment × d-orbital choice,
    produce one ElectronicConfig.

    Args:
        spin_isomers: List of SpinIsomer objects.
        cluster_info: Cluster description.
        max_configs: Optional limit on total configs (truncates).

    Returns:
        List of ElectronicConfig objects.
    """
    all_configs = []
    config_id = 0

    for iso in spin_isomers:
        # Get oxidation assignments for this spin isomer
        ox_assignments = enumerate_oxidation_assignments(cluster_info, iso)

        for ox in ox_assignments:
            # Determine d-orbital choices for metals with partially filled shells
            d_choices = _get_d_orbital_choices_for_cluster(
                cluster_info, iso, ox
            )

            if not d_choices:
                # No orbital choices — single configuration
                cfg = ElectronicConfig(
                    spin_isomer=iso,
                    oxidation=ox,
                    d_orbital_assignments={},
                    minority_spin_sites=sorted(
                        k for k, v in iso.spin_assignment.items() if v == -1
                    ),
                    spin_assignment=iso.spin_assignment.copy(),
                    config_id=config_id,
                    label=f"{iso.label}|{ox.description}|d0",
                )
                all_configs.append(cfg)
                config_id += 1
            else:
                # Cartesian product of d-orbital choices
                metal_indices = sorted(d_choices.keys())
                choice_lists = [d_choices[m] for m in metal_indices]

                for combo in product(*choice_lists):
                    d_assign = {metal_indices[k]: combo[k]
                                for k in range(len(metal_indices))}

                    d_label = "d" + "".join(
                        f"{m}:{v}" for m, v in sorted(d_assign.items())
                    )

                    cfg = ElectronicConfig(
                        spin_isomer=iso,
                        oxidation=ox,
                        d_orbital_assignments=d_assign,
                        minority_spin_sites=sorted(
                            k for k, v in iso.spin_assignment.items() if v == -1
                        ),
                        spin_assignment=iso.spin_assignment.copy(),
                        config_id=config_id,
                        label=f"{iso.label}|{ox.description}|{d_label}",
                    )
                    all_configs.append(cfg)
                    config_id += 1

                    if max_configs and config_id >= max_configs:
                        return all_configs

    return all_configs


def generate_all_configs_v2(cluster_info: ClusterInfo,
                             max_configs: int = None,
                             forced_oxidation: dict = None) -> list[ElectronicConfig]:
    """Generate physically consistent electronic configurations.

    Unlike ``generate_all_configs``, this enumerates oxidation assignments
    **first**, then for each assignment generates spin isomers using the
    **actual per-metal local spins** derived from that assignment.  This
    ensures the total-Sz constraint is validated against the real S_i values,
    filtering out oxidation-spin combinations that are physically impossible.

    Flow::

        for oxidation_assignment:
            per_metal_S = {i: get_local_spin(elem, ox_i)}
            spin_isomers = enumerate_spin_isomers(..., oxidation_states=ox)
            for spin_isomer:
                for d_orbital_combo:
                    yield ElectronicConfig

    Args:
        cluster_info: Cluster description.
        max_configs: Optional cap on total configs.
        forced_oxidation: Optional dict {site_idx: ox_state} to override
            automatic oxidation enumeration.

    Returns:
        List of physically consistent ElectronicConfig objects.
    """
    from .spin_config import enumerate_spin_isomers as _enum_spin

    all_configs = []
    config_id = 0

    # Step 1: enumerate oxidation assignments
    if forced_oxidation:
        ox_assignments = [OxidationAssignment(
            assignments=forced_oxidation,
            description=_describe_oxidation_assignment(
                [forced_oxidation.get(k, 2) for k in range(len(cluster_info.metals))],
                cluster_info.metals,
            ),
        )]
    else:
        ox_assignments = enumerate_oxidation_assignments(cluster_info)

    for ox in ox_assignments:
        # Step 2: enumerate spin isomers with *correct* per-metal local spins
        isomers = _enum_spin(
            cluster_info,
            oxidation_states=ox.assignments,
        )

        for iso in isomers:
            # Step 3: enumerate d-orbital choices
            d_choices = _get_d_orbital_choices_for_cluster(
                cluster_info, iso, ox,
            )
            if not d_choices:
                cfg = ElectronicConfig(
                    spin_isomer=iso,
                    oxidation=ox,
                    d_orbital_assignments={},
                    minority_spin_sites=sorted(
                        k for k, v in iso.spin_assignment.items() if v == -1
                    ),
                    spin_assignment=iso.spin_assignment.copy(),
                    config_id=config_id,
                    label=f"{iso.label}|{ox.description}|d0",
                )
                all_configs.append(cfg)
                config_id += 1
            else:
                metal_indices = sorted(d_choices.keys())
                choice_lists = [d_choices[m] for m in metal_indices]

                for combo in product(*choice_lists):
                    d_assign = {metal_indices[k]: combo[k]
                                for k in range(len(metal_indices))}
                    d_label = "d" + "".join(
                        f"{m}:{v}" for m, v in sorted(d_assign.items())
                    )

                    cfg = ElectronicConfig(
                        spin_isomer=iso,
                        oxidation=ox,
                        d_orbital_assignments=d_assign,
                        minority_spin_sites=sorted(
                            k for k, v in iso.spin_assignment.items() if v == -1
                        ),
                        spin_assignment=iso.spin_assignment.copy(),
                        config_id=config_id,
                        label=f"{iso.label}|{ox.description}|{d_label}",
                    )
                    all_configs.append(cfg)
                    config_id += 1

                    if max_configs and config_id >= max_configs:
                        return all_configs

    return all_configs


def reduce_configs_by_symmetry(configs: list[ElectronicConfig],
                                cluster_info: ClusterInfo) -> list[ElectronicConfig]:
    """Remove symmetry-equivalent electronic configurations.

    Detects equivalent metal sites (same element, similar geometry and
    coordination environment), then keeps only one representative per
    group of configs that differ only by permuting equivalent sites.

    Args:
        configs: List of ElectronicConfig objects.
        cluster_info: Cluster description with metal geometry.

    Returns:
        Deduplicated list of ElectronicConfig objects.
    """
    if len(configs) <= 1 or len(cluster_info.metals) <= 1:
        return configs

    equiv_maps = _build_equivalence_maps(cluster_info)
    if not equiv_maps:
        return configs

    # Group configs: keep first representative of each equivalence class
    visited_keys = set()
    unique_configs = []

    for cfg in configs:
        # Generate all keys for this config (original + permuted)
        keys = set()
        keys.add(_config_key(cfg))
        for perm in equiv_maps:
            keys.add(_apply_permutation_key(cfg, perm))

        # If none of the keys have been seen, this is a new equivalence class
        if not keys.intersection(visited_keys):
            unique_configs.append(cfg)
            visited_keys.update(keys)

    return unique_configs


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────


def _build_equivalence_maps(cluster_info: ClusterInfo) -> list[dict]:
    """Build metal index permutation maps from equivalent site detection.

    Two metals are equivalent if they share: element type, distance to
    centroid (within 0.5 A), coordination number, and sorted neighbor
    element types.

    Returns:
        List of non-trivial permutation dicts, e.g. [{0:1, 1:0}].
        Empty list if no equivalent sites found.
    """
    metals = cluster_info.metals
    n = len(metals)
    if n <= 1:
        return []

    positions = np.array([m.position for m in metals])
    centroid = positions.mean(axis=0)
    dist_to_center = np.linalg.norm(positions - centroid, axis=1)

    # Build neighbor element signature for each metal
    # (sorted tuple of neighbor element symbols)
    all_elements = cluster_info.all_elements
    neighbor_sigs = []
    for m in metals:
        sig = tuple(sorted(all_elements[nb] for nb in m.neighbors))
        neighbor_sigs.append(sig)

    # Group metals by equivalence: (element, coordination, neighbor_sig)
    # then check distance consistency within each group
    groups = {}
    for i, m in enumerate(metals):
        key = (m.element, m.coordination, neighbor_sigs[i])
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # Sub-split groups by distance to centroid (within tolerance)
    dist_tol = 0.5
    equiv_groups = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        # Cluster by distance
        dists = [dist_to_center[i] for i in indices]
        sorted_pairs = sorted(zip(dists, indices))
        subgroups = []
        current = [sorted_pairs[0][1]]
        for k in range(1, len(sorted_pairs)):
            if abs(sorted_pairs[k][0] - sorted_pairs[k-1][0]) < dist_tol:
                current.append(sorted_pairs[k][1])
            else:
                if len(current) >= 2:
                    subgroups.append(current)
                current = [sorted_pairs[k][1]]
        if len(current) >= 2:
            subgroups.append(current)
        equiv_groups.extend(subgroups)

    if not equiv_groups:
        return []

    # Generate permutation maps for each group, then Cartesian product
    group_perms = []
    for group in equiv_groups:
        perms = []
        for p in permutations(range(len(group))):
            if any(p[i] != i for i in range(len(p))):
                perm = {group[i]: group[p[i]] for i in range(len(group))}
                perms.append(perm)
        group_perms.append(perms)

    # Combine across groups
    if len(group_perms) == 1:
        return group_perms[0]

    combined = []
    for combo in product(*group_perms):
        merged = {}
        for p in combo:
            merged.update(p)
        combined.append(merged)
    return combined


def _config_key(cfg: ElectronicConfig) -> tuple:
    """Return a hashable canonical key for a config."""
    spin = tuple(sorted(cfg.spin_assignment.items()))
    if cfg.oxidation:
        ox = tuple(sorted(cfg.oxidation.assignments.items()))
    else:
        ox = ()
    d_orb = tuple(sorted(cfg.d_orbital_assignments.items()))
    return (spin, ox, d_orb)


def _apply_permutation_key(cfg: ElectronicConfig, perm: dict) -> tuple:
    """Apply a metal index permutation to a config, return canonical key."""
    new_spin = tuple(sorted(
        (perm.get(k, k), v) for k, v in cfg.spin_assignment.items()
    ))
    if cfg.oxidation:
        new_ox = tuple(sorted(
            (perm.get(k, k), v) for k, v in cfg.oxidation.assignments.items()
        ))
    else:
        new_ox = ()
    new_d = tuple(sorted(
        (perm.get(k, k), v) for k, v in cfg.d_orbital_assignments.items()
    ))
    return (new_spin, new_ox, new_d)


def estimate_computational_cost(n_configs: int,
                                 active_space_n_electrons: int,
                                 active_space_n_orbitals: int,
                                 method: str = "UHF") -> dict:
    """Estimate wall time / computational cost for a batch of calculations.

    Returns a dict with rough estimates (orders of magnitude only).
    These are not exact — they provide guidance for planning.

    Args:
        n_configs: Number of configurations.
        active_space_n_electrons: Number of active electrons.
        active_space_n_orbitals: Number of active orbitals.
        method: "UHF", "UCCSD", "UCCSDT", "DMRG".

    Returns:
        Dict with cost estimates.
    """
    n = active_space_n_orbitals
    e = active_space_n_electrons

    # Rough scaling estimates (relative units)
    cost_per_config = {
        "UHF": n ** 3,
        "UCCSD": n ** 6,
        "UCCSDT": n ** 8,
        "DMRG": n ** 3 * 5000,  # D * n^3 with D~5000
        "CCSDTQ": n ** 10,
    }

    per_config = cost_per_config.get(method, n ** 4)
    total = per_config * n_configs

    return {
        "n_configs": n_configs,
        "method": method,
        "cost_per_config_relative": per_config,
        "total_cost_relative": total,
        "active_space": f"({e}e, {n}o)",
        "recommendation": _cost_recommendation(n_configs, total, method),
    }


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _estimate_ligand_charge(cluster_info: ClusterInfo) -> int:
    """Estimate total charge from ligands (bridging + terminal).

    First tries to use a matched cluster template's ``total_metal_oxidation``
    field to derive the correct ligand charge.  Falls back to the element-based
    ligand database when no template matches.
    """
    # Try template-based estimate first
    from apex_cas.CAS_builder_noncomputing import _match_cluster_template

    template = _match_cluster_template(cluster_info)
    if template and "total_metal_oxidation" in template:
        metal_ox_data = template["total_metal_oxidation"]
        # Handle dict-type (charge-dependent) or integer-type
        if isinstance(metal_ox_data, dict):
            charge = cluster_info.total_charge
            # Build lookup key: charge_minus2, charge_minus1, charge_0, charge_plus1, etc.
            if charge < 0:
                key = f"charge_minus{abs(charge)}"
            elif charge > 0:
                key = f"charge_plus{charge}"
            else:
                key = "charge_0"
            metal_ox_sum = metal_ox_data.get(key)
            if metal_ox_sum is None:
                # Fallback: element-based estimate
                pass
            else:
                return cluster_info.total_charge - metal_ox_sum
        else:
            # Integer value (e.g., FeMo-co: 21)
            metal_ox_sum = metal_ox_data
            return cluster_info.total_charge - metal_ox_sum

    # Fallback: element-based estimate from ligand database
    import os
    import yaml

    from ._paths import data_file as _kb_file
    lig_path = _kb_file("ligand_database.yaml")
    with open(lig_path) as f:
        lig_db = yaml.safe_load(f)

    charge = 0
    for bridge in cluster_info.bridging_atoms:
        elem_data = lig_db.get(bridge.element, {})
        charges = elem_data.get("common_charges", [-2])
        charge += charges[0]

    for lig in cluster_info.terminal_ligands:
        charge += lig.charge

    return charge


def _describe_oxidation_assignment(combo, metals) -> str:
    """Generate a human-readable description of oxidation state assignment.

    Lists each metal's oxidation state in site order so that permutations
    (e.g., V0=II,V1=IV vs V0=IV,V1=II) produce distinct descriptions.
    """
    parts = []
    counts = {}
    for k, ox in enumerate(combo):
        elem = metals[k].element
        roman = _to_roman(ox)
        key = (elem, roman)
        if key not in counts:
            counts[key] = []
        counts[key].append(k)

    # Build label: group by (element, oxidation) but append site index
    # to distinguish permutations
    for (elem, roman), indices in sorted(counts.items()):
        if len(indices) > 1:
            sites = ",".join(str(i) for i in indices)
            parts.append(f"{elem}{sites}({roman})")
        else:
            parts.append(f"{elem}{indices[0]}({roman})")

    return " + ".join(parts)


def _to_roman(n: int) -> str:
    """Convert integer to Roman numeral string."""
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = ""
    for val, sym in vals:
        while n >= val:
            result += sym
            n -= val
    return result


def _get_d_orbital_choices_for_cluster(cluster_info, spin_isomer, ox_assignment):
    """Get d-orbital choice indices for each metal that has a partially filled shell."""
    choices = {}
    for k, metal in enumerate(cluster_info.metals):
        ox = ox_assignment.assignments.get(k)
        if ox is None:
            continue
        spin_dir = spin_isomer.spin_assignment.get(k, +1)
        d_choices = enumerate_d_orbital_configs(metal.element, ox, spin_dir)
        if d_choices:
            choices[k] = d_choices
    return choices


def _cost_recommendation(n_configs, total_cost, method):
    """Generate a recommendation based on cost estimate."""
    if method == "UHF" and n_configs < 1000:
        return "Feasible — run all configurations."
    elif method == "UHF" and n_configs < 100000:
        return "Manageable — consider parallel execution."
    elif method == "UCCSD" and n_configs < 100:
        return "Feasible for coupled cluster."
    elif method == "UCCSDT" and n_configs < 50:
        return "Pushing limits — use FNO truncation."
    elif method == "DMRG" and n_configs < 50:
        return "DMRG feasible — ensure good orbital ordering."
    else:
        return "Very expensive — apply aggressive filtering."
