"""Spin isomer enumeration, oxidation state assignment, and electronic
configuration generation for transition metal clusters.

This module combines:
- internal spin-isomer enumeration primitives
- internal symmetry reduction of spin isomers
- internal oxidation-state assignment and d-orbital occupancy primitives
- public high-level electronic configuration generation
- public canonical relabeling of saved Step 2 configuration families

Reference: Zhai et al. 2026 — FeMo-co yields 35 collinear spin isomers
with target Sz=3/2, reduced to 10 BS families under C3 symmetry,
generating 78,750 total UHF initial guesses.
"""

import logging
from itertools import combinations as _combinations, permutations as _permutations, product as _product

import numpy as np
import yaml

from shared.cluster_info_labels import resolve_metal_site_label as _resolve_metal_site_label
from shared.knowledge_base import data_file as _knowledge_base_file
from shared.models import (
    ClusterInfo as _ClusterInfo,
    ElectronicConfig as _ElectronicConfig,
    OxidationAssignment as _OxidationAssignment,
    SpinIsomer as _SpinIsomer,
    SpinIsomerFamily as _SpinIsomerFamily,
)
from shared.roman import to_roman as _to_roman
from shared.chem_knowledge import (
    get_common_oxidation_states as _get_common_oxidation_states,
    get_d_electron_count as _get_d_electron_count,
    get_local_spin as _get_local_spin,
    match_cluster_template as _match_cluster_template,
)
# ══════════════════════════════════════════════════════════════════
# Spin Isomer Enumeration
# ══════════════════════════════════════════════════════════════════

def _enumerate_spin_isomers(cluster_info: _ClusterInfo,
                            target_Sz: float = None,
                            oxidation_states: dict = None) -> list[_SpinIsomer]:
    """Enumerate all collinear broken-symmetry spin isomers.

    Each metal site has a local spin Si that can point up (+1) or down (-1).
    Valid isomers satisfy the constraint: sum(±Si) = target_Sz.

    Args:
        cluster_info: Cluster description with metal centers.
        target_Sz: Target total Sz. If None, uses cluster_info.target_spin.
        oxidation_states: Optional dict {metal_idx: oxidation_state}.
            Needed to determine local spins. If None, inferred automatically.

    Returns:
        List of SpinIsomer objects satisfying sum(±Si) = target_Sz.
    """
    if target_Sz is None:
        target_Sz = cluster_info.target_spin

    metals = cluster_info.metals
    if not metals:
        return []

    # Determine local spins for each metal
    local_spins = _get_local_spins(metals, oxidation_states, cluster_info)

    # Enumerate all ±1 assignments
    n_metals = len(metals)
    isomers = []

    for signs in _product([+1, -1], repeat=n_metals):
        # Compute total Sz for this assignment
        total_Sz = sum(s * si for s, si in zip(signs, local_spins))

        if abs(total_Sz - target_Sz) < 1e-6:
            # Count minority-spin sites
            n_minority = sum(1 for s in signs if s == -1)

            spin_assignment = {k: signs[k] for k in range(n_metals)}
            minority_sites = sorted([k for k in range(n_metals) if signs[k] == -1])

            # Generate label
            family_label = f"BS{n_minority}"
            site_label = "".join(str(s + 1) for s in minority_sites) if minority_sites else "0"
            full_label = f"{family_label}-{site_label}"

            isomers.append(_SpinIsomer(
                label=full_label,
                spin_assignment=spin_assignment,
                n_minority=n_minority,
                family=family_label,
                Sz=total_Sz,
            ))

    return isomers


# ══════════════════════════════════════════════════════════════════
# Symmetry Reduction
# ══════════════════════════════════════════════════════════════════

def _apply_symmetry_reduction(isomers: list[_SpinIsomer],
                               symmetry_group: str = "C1",
                               metal_positions: np.ndarray = None) -> list[_SpinIsomerFamily]:
    """Group equivalent spin isomers under point group symmetry.

    Args:
        isomers: List of SpinIsomer objects.
        symmetry_group: Approximate point group (e.g., "C3", "C1").
        metal_positions: (n_metals, 3) array of metal positions.

    Returns:
        List of SpinIsomerFamily objects, each containing equivalent isomers.
    """
    if symmetry_group == "C1" or not isomers:
        # No symmetry reduction — each isomer is its own family
        families = []
        for iso in isomers:
            fam = _SpinIsomerFamily(
                label=iso.family,
                n_minority=iso.n_minority,
                isomers=[iso],
                representative=iso,
            )
            families.append(fam)
        return families

    # Generate symmetry operations
    fold = _parse_fold(symmetry_group)
    if fold <= 1 or metal_positions is None:
        return _group_by_minority_set(isomers)

    # Find rotation axis — assume it passes through the centroid of metal positions
    center = metal_positions.mean(axis=0)
    axis = _find_rotation_axis(metal_positions, center)

    # Generate equivalent atom index mappings under rotation
    equiv_maps = _generate_equivalence_maps(metal_positions, center, axis, fold)

    # Group isomers by their equivalence class
    return _group_by_symmetry(isomers, equiv_maps, fold)


def _label_isomers(families: list[_SpinIsomerFamily]) -> list[_SpinIsomerFamily]:
    """Label isomer families as BSn and individual isomers as BSn-ijk.

    Convention: ijk are 1-indexed positions of minority-spin metals,
    sorted in ascending order within each family.

    Args:
        families: List of SpinIsomerFamily to label.

    Returns:
        Same families with updated labels.
    """
    # Track family counts per n_minority
    family_counter = {}

    for fam in families:
        n = fam.n_minority
        family_counter[n] = family_counter.get(n, 0) + 1
        fam_idx = family_counter[n]
        fam.label = f"BS{n}_{fam_idx}"

        # Label individual isomers within family
        for i, iso in enumerate(fam.isomers):
            minority_sites = sorted(
                k for k, v in iso.spin_assignment.items() if v == -1
            )
            site_label = "".join(str(s + 1) for s in minority_sites)
            iso.family = fam.label
            iso.label = f"{fam.label}-{site_label}"

        # Set representative as the first isomer
        if fam.isomers:
            fam.representative = fam.isomers[0]

    return families


# ══════════════════════════════════════════════════════════════════
# Oxidation State Enumeration
# ══════════════════════════════════════════════════════════════════

def _enumerate_oxidation_assignments(cluster_info: _ClusterInfo,
                                      spin_isomer: _SpinIsomer = None,
                                      allowed_oxidations: dict = None) -> list[_OxidationAssignment]:
    """Enumerate valid oxidation state assignments for the metal centers.

    Constraints:
    - sum(metal_oxidation_states) + ligand_charge = total_charge
    - Each metal uses one of its common oxidation states
    - If spin_isomer is given, the spin assignment constrains which
      oxidation states are compatible

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
            opts = _get_common_oxidation_states(metal.element)
        if not opts:
            opts = [2, 3]
        options.append(opts)

    # Enumerate all combinations and sort by distance to target_sum
    candidates = []
    for combo in _product(*options):
        distance = abs(sum(combo) - target_sum)
        desc = _describe_oxidation_assignment(combo, metals)
        candidates.append((distance, _OxidationAssignment(
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
        logging.getLogger(__name__).warning(
            "No exact oxidation state match found. "
            "Closest match has distance %.1f from target charge balance. "
            "Returning %d closest assignment(s).",
            min_dist, len(closest)
        )

    return closest


# ══════════════════════════════════════════════════════════════════
# D-Orbital Configuration
# ══════════════════════════════════════════════════════════════════

def _enumerate_d_orbital_configs(metal_element: str,
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
    d_count = _get_d_electron_count(metal_element, oxidation_state)
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
    # the singly-occupied orbitals.
    n_choices = n_singly_occupied

    if spin_direction == -1:
        return list(range(n_choices))
    else:
        return list(range(n_choices))


# ══════════════════════════════════════════════════════════════════
# Electronic Configuration Generation
# ══════════════════════════════════════════════════════════════════

def generate_all_configs(cluster_info: _ClusterInfo,
                         target_Sz: float = None,
                         max_configs: int = None,
                         forced_oxidation: dict = None) -> list[_ElectronicConfig]:
    """Generate physically consistent electronic configurations.

    Enumerates oxidation assignments **first**, then for each assignment
    generates spin isomers using the **actual per-metal local spins** derived
    from that assignment.  This ensures the total-Sz constraint is validated
    against the real S_i values, filtering out oxidation-spin combinations
    that are physically impossible.

    Flow::

        for oxidation_assignment:
            per_metal_S = {i: _get_local_spin(elem, ox_i)}
            spin_isomers = enumerate_spin_isomers(..., oxidation_states=ox)
            for spin_isomer:
                for d_orbital_combo:
                    yield ElectronicConfig

    Args:
        cluster_info: Cluster description.
        target_Sz: Optional target total Sz override.
        max_configs: Optional cap on total configs.
        forced_oxidation: Optional dict {site_idx: ox_state} to override
            automatic oxidation enumeration.

    Returns:
        List of physically consistent ElectronicConfig objects.
    """
    profile = getattr(cluster_info, "benchmark_profile", "") or ""
    if profile == "chan_fe2s2_2017":
        return _generate_configs_chan_fe2s2(cluster_info, target_Sz=target_Sz)
    if profile == "chan_fe4s4_2017":
        return _generate_configs_chan_fe4s4(cluster_info)
    if profile == "chan_femoco_2026_llduc":
        return _generate_configs_chan_femoco(cluster_info)

    all_configs = []
    config_id = 0

    # Step 1: enumerate oxidation assignments
    if forced_oxidation:
        ox_assignments = [_OxidationAssignment(
            assignments=forced_oxidation,
            description=_describe_oxidation_assignment(
                [forced_oxidation.get(k, 2) for k in range(len(cluster_info.metals))],
                cluster_info.metals,
            ),
        )]
    else:
        ox_assignments = _enumerate_oxidation_assignments(cluster_info)

    for ox in ox_assignments:
        # Step 2: enumerate spin isomers with *correct* per-metal local spins
        isomers = _enumerate_spin_isomers(
            cluster_info,
            target_Sz=target_Sz,
            oxidation_states=ox.assignments,
        )

        for iso in isomers:
            # Step 3: enumerate d-orbital choices
            d_choices = _get_d_orbital_choices_for_cluster(
                cluster_info, iso, ox,
            )
            if not d_choices:
                cfg = _ElectronicConfig(
                    spin_isomer=iso,
                    oxidation=ox,
                    d_orbital_assignments={},
                    minority_spin_sites=sorted(
                        k for k, v in iso.spin_assignment.items() if v == -1
                    ),
                    spin_assignment=iso.spin_assignment.copy(),
                    config_id=config_id,
                    label=_build_config_label(cluster_info, iso, ox, {}),
                )
                all_configs.append(cfg)
                config_id += 1
            else:
                metal_indices = sorted(d_choices.keys())
                choice_lists = [d_choices[m] for m in metal_indices]

                for combo in _product(*choice_lists):
                    d_assign = {metal_indices[k]: combo[k]
                                for k in range(len(metal_indices))}
                    cfg = _ElectronicConfig(
                        spin_isomer=iso,
                        oxidation=ox,
                        d_orbital_assignments=d_assign,
                        minority_spin_sites=sorted(
                            k for k, v in iso.spin_assignment.items() if v == -1
                        ),
                        spin_assignment=iso.spin_assignment.copy(),
                        config_id=config_id,
                        label=_build_config_label(cluster_info, iso, ox, d_assign),
                    )
                    all_configs.append(cfg)
                    config_id += 1

                    if max_configs and config_id >= max_configs:
                        return all_configs

    return all_configs


def canonicalize_config_spin_labels(
    configs: list[_ElectronicConfig],
    cluster_info: _ClusterInfo,
) -> tuple[list[_ElectronicConfig], list[_SpinIsomer], list[_SpinIsomerFamily]]:
    """Relabel config spin isomers using one symmetry-reduced canonical map.

    This keeps the config list, the saved unique spin isomer list, and the
    family metadata on the same labeling scheme.
    """
    if not configs:
        return [], [], []

    profile = getattr(cluster_info, "benchmark_profile", "") or ""
    if profile == "chan_fe2s2_2017":
        return _canonicalize_chan_fe2s2(configs)
    if profile == "chan_fe4s4_2017":
        return _canonicalize_chan_fe4s4(configs)
    if profile == "chan_femoco_2026_llduc":
        return _canonicalize_chan_femoco(configs)
    family_scheme = getattr(cluster_info, "family_scheme", "") or ""
    if family_scheme == "literature_fe4s4_cubane":
        return _canonicalize_literature_fe4s4_cubane(configs)

    unique_isomers = []
    by_key = {}
    for cfg in configs:
        if cfg.spin_isomer is None:
            continue
        key = _spin_isomer_key(cfg.spin_isomer)
        if key not in by_key:
            iso = cfg.spin_isomer
            by_key[key] = _SpinIsomer(
                label=iso.label,
                spin_assignment=dict(iso.spin_assignment),
                n_minority=iso.n_minority,
                family=iso.family,
                Sz=iso.Sz,
                symmetry_equivalent=list(iso.symmetry_equivalent),
            )
            unique_isomers.append(by_key[key])

    if not unique_isomers:
        return configs, [], []

    metal_positions = np.array([m.position for m in cluster_info.metals]) if cluster_info.metals else None
    reduction_symmetry = getattr(cluster_info, "reduction_symmetry", None) or cluster_info.symmetry_group
    families = _apply_symmetry_reduction(unique_isomers, reduction_symmetry, metal_positions)
    families = _label_isomers(families)

    labeled_by_key = {}
    for fam in families:
        for iso in fam.isomers:
            labeled_by_key[_spin_isomer_key(iso)] = iso

    relabeled_configs = []
    for cfg in configs:
        if cfg.spin_isomer is None:
            relabeled_configs.append(cfg)
            continue

        labeled_iso = labeled_by_key[_spin_isomer_key(cfg.spin_isomer)]
        cfg.spin_isomer = _SpinIsomer(
            label=labeled_iso.label,
            spin_assignment=dict(labeled_iso.spin_assignment),
            n_minority=labeled_iso.n_minority,
            family=labeled_iso.family,
            Sz=labeled_iso.Sz,
            symmetry_equivalent=list(labeled_iso.symmetry_equivalent),
        )
        cfg.spin_assignment = dict(labeled_iso.spin_assignment)
        cfg.minority_spin_sites = sorted(
            site_idx for site_idx, sign in labeled_iso.spin_assignment.items() if sign == -1
        )
        if cfg.oxidation is not None:
            cfg.label = _build_config_label(cluster_info, cfg.spin_isomer, cfg.oxidation, cfg.d_orbital_assignments)
        relabeled_configs.append(cfg)

    return relabeled_configs, unique_isomers, families


# ══════════════════════════════════════════════════════════════════
# Symmetry Reduction for Electronic Configurations
# ══════════════════════════════════════════════════════════════════

def _reduce_configs_by_symmetry(configs: list[_ElectronicConfig],
                                cluster_info: _ClusterInfo) -> list[_ElectronicConfig]:
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
    reduction_mode = getattr(cluster_info, "config_reduction_mode", "none")
    if reduction_mode == "none":
        return configs

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


def _summarize_enumeration_layers(
    configs_before_reduction: list[_ElectronicConfig],
    configs_after_reduction: list[_ElectronicConfig] | None = None,
    spin_isomers: list[_SpinIsomer] | None = None,
    families: list[_SpinIsomerFamily] | None = None,
) -> dict:
    """Summarize enumeration using a fixed, cross-system counting vocabulary.

    The returned counts are intentionally layer-based rather than benchmark-
    specific, so Fe2S2 / Fe4S4 / Fe4S4H4 / FeMo-co can be compared without
    overloading the word "config".
    """
    before = configs_before_reduction or []
    after = configs_after_reduction if configs_after_reduction is not None else before

    def _spin_label(cfg: _ElectronicConfig) -> str:
        return cfg.spin_isomer.label if cfg.spin_isomer else ""

    def _family_label(cfg: _ElectronicConfig) -> str:
        return cfg.spin_isomer.family if cfg.spin_isomer else ""

    def _ox_key(cfg: _ElectronicConfig):
        if cfg.oxidation is None:
            return ()
        return tuple(sorted(cfg.oxidation.assignments.items()))

    def _d_key(cfg: _ElectronicConfig):
        return tuple(sorted(cfg.d_orbital_assignments.items()))

    raw_spin_patterns = len({_spin_label(cfg) for cfg in before if _spin_label(cfg)})
    spin_families = (
        len(families)
        if families is not None
        else len({_family_label(cfg) for cfg in before if _family_label(cfg)})
    )
    spin_x_oxidation = len({
        (_spin_label(cfg), _ox_key(cfg))
        for cfg in before
        if _spin_label(cfg)
    })
    spin_x_oxidation_x_d_before = len({
        (_spin_label(cfg), _ox_key(cfg), _d_key(cfg))
        for cfg in before
        if _spin_label(cfg)
    })

    return {
        "raw_spin_patterns": raw_spin_patterns,
        "spin_families": spin_families,
        "spin_x_oxidation": spin_x_oxidation,
        "spin_x_oxidation_x_d_before_reduction": spin_x_oxidation_x_d_before,
        "total_configs_after_reduction": len(after),
        "spin_isomer_count_reported": len(spin_isomers) if spin_isomers is not None else raw_spin_patterns,
        "family_count_reported": len(families) if families is not None else spin_families,
    }


# ══════════════════════════════════════════════════════════════════
# Internal Helpers — Spin Isomers
# ══════════════════════════════════════════════════════════════════

def _get_local_spins(metals, oxidation_states=None, cluster_info=None):
    """Determine local spin S for each metal center."""
    spins = []
    for k, metal in enumerate(metals):
        if oxidation_states and k in oxidation_states:
            ox = oxidation_states[k]
        else:
            ox = _infer_oxidation_state(metal.element, k, metals, cluster_info)
        S = _get_local_spin(metal.element, ox)
        spins.append(S)
    return spins


def _infer_oxidation_state(element, metal_idx, metals, cluster_info=None):
    """Infer the most likely oxidation state for a metal center."""
    states = _get_common_oxidation_states(element)
    if not states:
        return 2

    if cluster_info is None:
        return states[0]

    # Simple case: single metal — charge balance gives oxidation directly
    if len(metals) == 1:
        ligand_charge = _estimate_ligand_charge_simple(cluster_info)
        target_ox = cluster_info.total_charge - ligand_charge
        best = min(states, key=lambda s: abs(s - target_ox))
        return best

    return states[0]


def _estimate_ligand_charge_simple(cluster_info):
    """Simple element-based ligand charge estimate (for spin inference)."""
    total = 0
    for bridge in cluster_info.bridging_atoms:
        if getattr(bridge, "charge", None) is not None:
            total += bridge.charge
            continue
        if bridge.element in ("S", "Se"):
            total += -2
        elif bridge.element == "O":
            total += -2
        elif bridge.element == "N":
            total += -3
        elif bridge.element == "C":
            total += -4
        elif bridge.element == "Cl":
            total += -1
    for lig in cluster_info.terminal_ligands:
        total += lig.charge
    return total


def _parse_fold(symmetry_group: str) -> int:
    """Parse Cn symmetry to get fold n."""
    if symmetry_group.startswith("C") and symmetry_group[1:].isdigit():
        return int(symmetry_group[1:])
    return 1


def _find_rotation_axis(positions, center):
    """Find the dominant rotation axis from metal positions via PCA."""
    centered = positions - center
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, -1]


def _generate_equivalence_maps(positions, center, axis, fold):
    """Generate atom index equivalence maps under n-fold rotation."""
    n = len(positions)
    maps = []
    centered = positions - center

    for rot_idx in range(1, fold):
        angle = 2 * np.pi * rot_idx / fold
        rot_matrix = _rotation_matrix(axis, angle)
        rotated = (rot_matrix @ centered.T).T + center

        mapping = {}
        for i in range(n):
            dists = np.linalg.norm(positions - rotated[i], axis=1)
            mapping[i] = int(np.argmin(dists))
        maps.append(mapping)

    return maps


def _rotation_matrix(axis, angle):
    """Rotation matrix for rotation by angle around axis (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _group_by_symmetry(isomers, equiv_maps, fold):
    """Group isomers into families based on symmetry equivalence."""
    visited = set()
    families = []

    for i, iso in enumerate(isomers):
        if i in visited:
            continue

        group = [iso]
        visited.add(i)
        minority_set = frozenset(k for k, v in iso.spin_assignment.items() if v == -1)

        for j in range(i + 1, len(isomers)):
            if j in visited:
                continue
            other_minority = frozenset(k for k, v in isomers[j].spin_assignment.items()
                                       if v == -1)

            if _are_symmetry_equivalent(minority_set, other_minority, equiv_maps):
                group.append(isomers[j])
                visited.add(j)

        n_minority = iso.n_minority
        fam = _SpinIsomerFamily(
            label=f"BS{n_minority}",
            n_minority=n_minority,
            isomers=group,
            representative=iso,
        )
        families.append(fam)

    return families


def _are_symmetry_equivalent(set_a, set_b, equiv_maps):
    """Check if two minority-spin sets are equivalent under symmetry."""
    if len(set_a) != len(set_b):
        return False

    for mapping in equiv_maps:
        mapped_a = frozenset(mapping.get(s, s) for s in set_a)
        if mapped_a == set_b:
            return True
    return False


def _group_by_minority_set(isomers):
    """Group isomers by their minority-spin set (no symmetry)."""
    groups = {}
    for iso in isomers:
        key = frozenset(k for k, v in iso.spin_assignment.items() if v == -1)
        if key not in groups:
            groups[key] = []
        groups[key].append(iso)

    families = []
    for key, group in groups.items():
        n_minority = len(key)
        fam = _SpinIsomerFamily(
            label=f"BS{n_minority}",
            n_minority=n_minority,
            isomers=group,
            representative=group[0],
        )
        families.append(fam)

    return families


def _spin_isomer_key(isomer: _SpinIsomer) -> tuple:
    """Canonical key for deduplicating and relabeling spin isomers."""
    return tuple(sorted((int(site_idx), int(sign)) for site_idx, sign in isomer.spin_assignment.items()))


# ══════════════════════════════════════════════════════════════════
# Internal Helpers — Electronic Configurations
# ══════════════════════════════════════════════════════════════════

def _estimate_ligand_charge(cluster_info: _ClusterInfo) -> int:
    """Estimate total charge from ligands (bridging + terminal).

    When explicit ``cluster_info.yaml`` annotations are present, respect the
    annotated bridge/terminal donor charges directly. Otherwise:

    1. try a matched cluster template's ``total_metal_oxidation`` field
    2. fall back to the element-based ligand database
    """
    if getattr(cluster_info, "annotation_source", "") == "cluster_info_yaml":
        return _estimate_ligand_charge_simple(cluster_info)

    # Try template-based estimate first
    try:
        template = _match_cluster_template(cluster_info)
    except (ImportError, Exception):
        template = None

    if template and "total_metal_oxidation" in template:
        metal_ox_data = template["total_metal_oxidation"]
        if isinstance(metal_ox_data, dict):
            charge = cluster_info.total_charge
            if charge < 0:
                key = f"charge_minus{abs(charge)}"
            elif charge > 0:
                key = f"charge_plus{charge}"
            else:
                key = "charge_0"
            metal_ox_sum = metal_ox_data.get(key)
            if metal_ox_sum is not None:
                return cluster_info.total_charge - metal_ox_sum
        else:
            metal_ox_sum = metal_ox_data
            return cluster_info.total_charge - metal_ox_sum

    # Fallback: element-based estimate from ligand database
    try:
        lig_path = _knowledge_base_file("ligand_database.yaml")
        with open(lig_path) as f:
            lig_db = yaml.safe_load(f)
    except (FileNotFoundError, IOError):
        return _estimate_ligand_charge_simple(cluster_info)

    charge = 0
    for bridge in cluster_info.bridging_atoms:
        if getattr(bridge, "charge", None) is not None:
            charge += bridge.charge
        else:
            elem_data = lig_db.get(bridge.element, {})
            charges = elem_data.get("common_charges", [-2])
            charge += charges[0]

    for lig in cluster_info.terminal_ligands:
        charge += lig.charge

    return charge


def _describe_oxidation_assignment(combo, metals) -> str:
    """Generate a human-readable description of oxidation state assignment."""
    pseudo_cluster = type(
        "ClusterInfoLabels",
        (),
        {
            "metals": metals,
            "annotation_source": (
                "cluster_info_yaml" if all(getattr(m, "label", "") for m in metals) else "auto"
            ),
        },
    )()
    parts = []
    for k, ox in enumerate(combo):
        label = _resolve_metal_site_label(pseudo_cluster, k)
        parts.append(f"{label}({_to_roman(ox)})")
    return "+".join(parts)

def _get_d_orbital_choices_for_cluster(cluster_info, spin_isomer, ox_assignment):
    """Get d-orbital choice indices for each metal that has a partially filled shell."""
    choices = {}
    for k, metal in enumerate(cluster_info.metals):
        ox = ox_assignment.assignments.get(k)
        if ox is None:
            continue
        spin_dir = spin_isomer.spin_assignment.get(k, +1)
        d_choices = _enumerate_d_orbital_configs(metal.element, ox, spin_dir)
        if d_choices:
            choices[k] = d_choices
    return choices


def _metal_site_label(cluster_info: _ClusterInfo, site_idx: int) -> str:
    """Return the user-facing label for a metal site."""
    return _resolve_metal_site_label(cluster_info, site_idx)


def _format_spin_pattern(cluster_info: _ClusterInfo, spin_assignment: dict) -> str:
    """Render a user-facing spin pattern such as Fe1↓Fe2↑."""
    parts = []
    for site_idx in sorted(spin_assignment):
        label = _metal_site_label(cluster_info, site_idx)
        arrow = "↓" if spin_assignment[site_idx] == -1 else "↑"
        parts.append(f"{label}{arrow}")
    return "".join(parts)


def _format_d_assignments(cluster_info: _ClusterInfo, d_assign: dict) -> str:
    """Render user-facing d-assignment labels such as Fe1:d2."""
    if not d_assign:
        return "d:none"
    parts = []
    for site_idx, orb_idx in sorted(d_assign.items()):
        parts.append(f"{_metal_site_label(cluster_info, site_idx)}:d{int(orb_idx) + 1}")
    return ",".join(parts)


def _build_config_label(
    cluster_info: _ClusterInfo,
    spin_isomer: _SpinIsomer,
    oxidation: _OxidationAssignment,
    d_assign: dict,
) -> str:
    """Build the user-facing electronic-config label."""
    spin_label = _format_spin_pattern(cluster_info, spin_isomer.spin_assignment)
    oxidation_label = oxidation.description if oxidation else "ox:none"
    d_label = _format_d_assignments(cluster_info, d_assign)
    return f"{spin_label}|{oxidation_label}|{d_label}"


_FEMOCO_FAMILY_BY_SET = {
    "134": "BS3", "124": "BS3", "123": "BS3",
    "145": "BS9", "137": "BS9", "126": "BS9",
    "147": "BS10", "146": "BS10", "135": "BS10",
    "136": "BS10", "125": "BS10", "127": "BS10",
    "157": "BS6", "156": "BS6", "167": "BS6",
    "234": "BS2",
    "345": "BS8", "347": "BS8", "245": "BS8",
    "246": "BS8", "237": "BS8", "236": "BS8",
    "346": "BS7", "247": "BS7", "235": "BS7",
    "457": "BS5", "456": "BS5", "357": "BS5",
    "367": "BS5", "256": "BS5", "267": "BS5",
    "467": "BS4", "356": "BS4", "257": "BS4",
    "567": "BS1",
}

_FE4S4_CUBANE_FAMILY_BY_MINORITY = {
    frozenset({0, 1}): "BS1",
    frozenset({2, 3}): "BS1",
    frozenset({0, 2}): "BS2",
    frozenset({1, 3}): "BS2",
    frozenset({0, 3}): "BS3",
    frozenset({1, 2}): "BS3",
}


def _generate_configs_chan_fe2s2(cluster_info: _ClusterInfo, target_Sz=None) -> list[_ElectronicConfig]:
    """Reproduce the two initial singlet guesses used in Chan 2017 Fe2S2."""
    assignments = [
        {0: +1, 1: -1},
        {0: -1, 1: +1},
    ]
    ox = _OxidationAssignment(
        assignments={0: 3, 1: 3},
        description="2xFe(III)",
    )
    configs = []
    for idx, spin_assignment in enumerate(assignments):
        minority = sorted(k for k, v in spin_assignment.items() if v == -1)
        site_label = "".join(str(i + 1) for i in minority)
        iso = _SpinIsomer(
            label=f"BS1-{site_label}",
            spin_assignment=spin_assignment,
            n_minority=1,
            family=f"BS1_{idx + 1}",
            Sz=0.0 if target_Sz is None else target_Sz,
        )
        configs.append(_ElectronicConfig(
            spin_isomer=iso,
            oxidation=ox,
            d_orbital_assignments={},
            minority_spin_sites=minority,
            spin_assignment=dict(spin_assignment),
            config_id=idx,
            label=_build_config_label(cluster_info, iso, ox, {}),
        ))
    return configs


def _generate_configs_chan_fe4s4(cluster_info: _ClusterInfo) -> list[_ElectronicConfig]:
    """Reproduce the 24 physically meaningful initial guesses from Chan 2017."""
    pair_family = {
        frozenset({0, 1}): "BS1",
        frozenset({2, 3}): "BS1",
        frozenset({0, 2}): "BS2",
        frozenset({1, 3}): "BS2",
        frozenset({0, 3}): "BS3",
        frozenset({1, 2}): "BS3",
    }
    configs = []
    config_id = 0

    for minority in _combinations(range(4), 2):
        minority_set = frozenset(minority)
        family = pair_family[minority_set]
        spin_assignment = {i: (-1 if i in minority_set else +1) for i in range(4)}
        up_sites = [i for i in range(4) if spin_assignment[i] == +1]
        down_sites = [i for i in range(4) if spin_assignment[i] == -1]

        for feii_up in up_sites:
            for feii_down in down_sites:
                ox_map = {i: 3 for i in range(4)}
                ox_map[feii_up] = 2
                ox_map[feii_down] = 2
                ox = _OxidationAssignment(
                    assignments=ox_map,
                    description=_describe_oxidation_assignment(
                        [ox_map[i] for i in range(4)], cluster_info.metals
                    ),
                )
                site_label = "".join(str(i + 1) for i in sorted(minority_set))
                iso = _SpinIsomer(
                    label=f"{family}-{site_label}",
                    spin_assignment=dict(spin_assignment),
                    n_minority=2,
                    family=family,
                    Sz=0.0,
                )
                label = (
                    f"{_format_spin_pattern(cluster_info, spin_assignment)}|"
                    f"{ox.description}|"
                    f"FeII_up={_metal_site_label(cluster_info, feii_up)},"
                    f"FeII_down={_metal_site_label(cluster_info, feii_down)}"
                )
                configs.append(_ElectronicConfig(
                    spin_isomer=iso,
                    oxidation=ox,
                    d_orbital_assignments={},
                    minority_spin_sites=sorted(minority_set),
                    spin_assignment=dict(spin_assignment),
                    config_id=config_id,
                    label=label,
                ))
                config_id += 1

    return configs


def _generate_configs_chan_femoco(cluster_info: _ClusterInfo) -> list[_ElectronicConfig]:
    """Reproduce the 35 × 18 × 5^3 = 78750 LLDUC initial configurations."""
    fe_metals = cluster_info.metals[:7]
    configs = []
    config_id = 0

    for minority in _combinations(range(7), 3):
        minority_set = frozenset(minority)
        majority = [i for i in range(7) if i not in minority_set]
        site_label = "".join(str(i + 1) for i in sorted(minority_set))
        family = _FEMOCO_FAMILY_BY_SET[site_label]
        spin_assignment = {i: (-1 if i in minority_set else +1) for i in range(7)}
        iso = _SpinIsomer(
            label=f"{family}-{site_label}",
            spin_assignment=dict(spin_assignment),
            n_minority=3,
            family=family,
            Sz=cluster_info.target_spin,
        )

        # For target Sz=3/2, Fe(II) sites must satisfy 2 majority + 1 minority.
        for feii_majority in _combinations(majority, 2):
            for feii_minority in _combinations(minority_set, 1):
                feii_sites = set(feii_majority) | set(feii_minority)
                ox_map = {i: (2 if i in feii_sites else 3) for i in range(7)}
                ox = _OxidationAssignment(
                    assignments=ox_map,
                    description=_describe_oxidation_assignment(
                        [ox_map[i] for i in range(7)], fe_metals
                    ),
                )
                d_sites = sorted(feii_sites)
                for d_combo in _product(range(5), repeat=3):
                    d_assign = {d_sites[k]: d_combo[k] for k in range(3)}
                    configs.append(_ElectronicConfig(
                        spin_isomer=iso,
                        oxidation=ox,
                        d_orbital_assignments=d_assign,
                        minority_spin_sites=sorted(minority_set),
                        spin_assignment=dict(spin_assignment),
                        config_id=config_id,
                        label=_build_config_label(cluster_info, iso, ox, d_assign),
                    ))
                    config_id += 1

    return configs


def _canonicalize_chan_fe2s2(
    configs: list[_ElectronicConfig],
) -> tuple[list[_ElectronicConfig], list[_SpinIsomer], list[_SpinIsomerFamily]]:
    spin_isomers = []
    families = []
    by_family = {}
    for cfg in configs:
        fam = cfg.spin_isomer.family
        if fam not in by_family:
            iso = _SpinIsomer(
                label=cfg.spin_isomer.label,
                spin_assignment=dict(cfg.spin_isomer.spin_assignment),
                n_minority=cfg.spin_isomer.n_minority,
                family=fam,
                Sz=cfg.spin_isomer.Sz,
            )
            by_family[fam] = iso
            spin_isomers.append(iso)
            families.append(_SpinIsomerFamily(
                label=fam,
                n_minority=iso.n_minority,
                isomers=[iso],
                representative=iso,
            ))
    return configs, spin_isomers, families


def _canonicalize_chan_fe4s4(
    configs: list[_ElectronicConfig],
) -> tuple[list[_ElectronicConfig], list[_SpinIsomer], list[_SpinIsomerFamily]]:
    representatives = {
        "BS1": {0: +1, 1: +1, 2: -1, 3: -1},
        "BS2": {0: +1, 1: -1, 2: +1, 3: -1},
        "BS3": {0: +1, 1: -1, 2: -1, 3: +1},
    }
    spin_isomers = []
    families = []
    for family, spin_assignment in representatives.items():
        iso = _SpinIsomer(
            label=family,
            spin_assignment=spin_assignment,
            n_minority=2,
            family=family,
            Sz=0.0,
        )
        spin_isomers.append(iso)
        families.append(_SpinIsomerFamily(
            label=family,
            n_minority=2,
            isomers=[iso],
            representative=iso,
        ))
    return configs, spin_isomers, families


def _canonicalize_literature_fe4s4_cubane(
    configs: list[_ElectronicConfig],
) -> tuple[list[_ElectronicConfig], list[_SpinIsomer], list[_SpinIsomerFamily]]:
    """Apply the standard 4Fe-4S cubane three-pairing family definition.

    This keeps the raw site-labeled patterns in the config list, but relabels
    their family according to the literature cubane pairing classes BS1/BS2/BS3.
    """
    representatives = {
        "BS1": {0: +1, 1: +1, 2: -1, 3: -1},
        "BS2": {0: +1, 1: -1, 2: +1, 3: -1},
        "BS3": {0: +1, 1: -1, 2: -1, 3: +1},
    }
    spin_isomers = []
    families = []
    for family, spin_assignment in representatives.items():
        iso = _SpinIsomer(
            label=family,
            spin_assignment=spin_assignment,
            n_minority=2,
            family=family,
            Sz=0.0,
        )
        spin_isomers.append(iso)
        families.append(_SpinIsomerFamily(
            label=family,
            n_minority=2,
            isomers=[iso],
            representative=iso,
        ))

    relabeled_configs = []
    for cfg in configs:
        if cfg.spin_isomer is None:
            relabeled_configs.append(cfg)
            continue
        minority = frozenset(
            site_idx for site_idx, sign in cfg.spin_isomer.spin_assignment.items() if sign == -1
        )
        family = _FE4S4_CUBANE_FAMILY_BY_MINORITY[minority]
        old_label = cfg.spin_isomer.label
        cfg.spin_isomer = _SpinIsomer(
            label=family,
            spin_assignment=dict(cfg.spin_isomer.spin_assignment),
            n_minority=cfg.spin_isomer.n_minority,
            family=family,
            Sz=cfg.spin_isomer.Sz,
        )
        if cfg.label.startswith(old_label):
            cfg.label = family + cfg.label[len(old_label):]
        relabeled_configs.append(cfg)

    return relabeled_configs, spin_isomers, families


def _canonicalize_chan_femoco(
    configs: list[_ElectronicConfig],
) -> tuple[list[_ElectronicConfig], list[_SpinIsomer], list[_SpinIsomerFamily]]:
    by_label = {}
    for cfg in configs:
        iso = cfg.spin_isomer
        if iso.label not in by_label:
            by_label[iso.label] = _SpinIsomer(
                label=iso.label,
                spin_assignment=dict(iso.spin_assignment),
                n_minority=iso.n_minority,
                family=iso.family,
                Sz=iso.Sz,
            )
    spin_isomers = list(by_label.values())
    families = []
    fam_map = {}
    for iso in spin_isomers:
        fam_map.setdefault(iso.family, []).append(iso)
    for family in sorted(fam_map):
        members = sorted(fam_map[family], key=lambda x: x.label)
        families.append(_SpinIsomerFamily(
            label=family,
            n_minority=members[0].n_minority if members else 0,
            isomers=members,
            representative=members[0] if members else None,
        ))
    return configs, spin_isomers, families


def _build_equivalence_maps(cluster_info: _ClusterInfo) -> list[dict]:
    """Build metal index permutation maps from equivalent site detection."""
    metals = cluster_info.metals
    n = len(metals)
    if n <= 1:
        return []

    positions = np.array([m.position for m in metals])
    centroid = positions.mean(axis=0)
    dist_to_center = np.linalg.norm(positions - centroid, axis=1)

    all_elements = cluster_info.all_elements
    neighbor_sigs = []
    for m in metals:
        sig = tuple(sorted(all_elements[nb] for nb in m.neighbors))
        neighbor_sigs.append(sig)

    groups = {}
    for i, m in enumerate(metals):
        key = (m.element, m.coordination, neighbor_sigs[i])
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    dist_tol = 0.5
    equiv_groups = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
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

    group_perms = []
    for group in equiv_groups:
        perms = []
        for p in _permutations(range(len(group))):
            if any(p[i] != i for i in range(len(p))):
                perm = {group[i]: group[p[i]] for i in range(len(group))}
                perms.append(perm)
        group_perms.append(perms)

    if len(group_perms) == 1:
        return group_perms[0]

    combined = []
    for combo in _product(*group_perms):
        merged = {}
        for p in combo:
            merged.update(p)
        combined.append(merged)
    return combined


def _config_key(cfg: _ElectronicConfig) -> tuple:
    """Return a hashable canonical key for a config."""
    spin = tuple(sorted(cfg.spin_assignment.items()))
    if cfg.oxidation:
        ox = tuple(sorted(cfg.oxidation.assignments.items()))
    else:
        ox = ()
    d_orb = tuple(sorted(cfg.d_orbital_assignments.items()))
    return (spin, ox, d_orb)


def _apply_permutation_key(cfg: _ElectronicConfig, perm: dict) -> tuple:
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
