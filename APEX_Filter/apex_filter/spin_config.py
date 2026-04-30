"""Module 3: Spin Configuration Enumerator

Enumerate all collinear broken-symmetry spin isomers for a transition metal
cluster. Each metal site has a local spin Si that can point up (+1) or down (-1).
Valid isomers satisfy the constraint: sum(±Si) = target_Sz.

Reference: Zhai et al. 2026 — FeMo-co yields 35 collinear spin isomers
with target Sz=3/2, reduced to 10 BS families under C3 symmetry.
"""

from itertools import product

import numpy as np

from .models import (
    ClusterInfo,
    SpinIsomer,
    SpinIsomerFamily,
)
from apex_cas.CAS_builder_noncomputing import get_local_spin, get_common_oxidation_states


# ──────────────────────────────────────────────────────────────────
# Main enumeration
# ──────────────────────────────────────────────────────────────────

def enumerate_spin_isomers(cluster_info: ClusterInfo,
                           target_Sz: float = None,
                           oxidation_states: dict = None) -> list[SpinIsomer]:
    """Enumerate all collinear broken-symmetry spin isomers.

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

    for signs in product([+1, -1], repeat=n_metals):
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

            isomers.append(SpinIsomer(
                label=full_label,
                spin_assignment=spin_assignment,
                n_minority=n_minority,
                family=family_label,
                Sz=total_Sz,
            ))

    return isomers


def apply_symmetry_reduction(isomers: list[SpinIsomer],
                              symmetry_group: str = "C1",
                              metal_positions: np.ndarray = None) -> list[SpinIsomerFamily]:
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
            fam = SpinIsomerFamily(
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


def label_isomers(families: list[SpinIsomerFamily]) -> list[SpinIsomerFamily]:
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


def rank_by_heisenberg(isomers: list[SpinIsomer],
                       J_couplings: np.ndarray = None,
                       connectivity: list = None,
                       cluster_info: ClusterInfo = None,
                       oxidation_states: dict = None) -> list[SpinIsomer]:
    """Rank spin isomers by estimated Heisenberg exchange energy.

    E = -sum_{ij} J_ij * Si · Sj = -sum_{ij} J_ij * (sign_i * Si) * (sign_j * Sj)

    Lower (more negative) energy = more favorable.

    Args:
        isomers: List of SpinIsomer objects.
        J_couplings: (n_metals, n_metals) matrix of exchange couplings.
            If None, uses a simple AFM superexchange model.
        connectivity: List of (i, j) pairs indicating which metals are connected.
        cluster_info: ClusterInfo for looking up metal elements and oxidation states.
        oxidation_states: Optional dict {metal_idx: oxidation_state}.

    Returns:
        Isomers sorted by energy (lowest first).
    """
    if not isomers:
        return []

    n_metals = max(max(iso.spin_assignment.keys()) for iso in isomers) + 1

    if J_couplings is None:
        J_couplings = _default_J_model(n_metals, connectivity)

    # Build spin magnitude lookup from knowledge base
    spin_magnitudes = _load_spin_magnitudes(
        n_metals, cluster_info, oxidation_states
    )

    ranked = []
    for iso in isomers:
        E = 0.0
        for i in range(n_metals):
            for j in range(i + 1, n_metals):
                if abs(J_couplings[i, j]) > 1e-10:
                    Si = spin_magnitudes.get(i, 2.0)
                    Sj = spin_magnitudes.get(j, 2.0)
                    E -= J_couplings[i, j] * iso.spin_assignment[i] * iso.spin_assignment[j] * Si * Sj
        ranked.append((E, iso))

    ranked.sort(key=lambda x: x[0])
    return [iso for _, iso in ranked]


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _get_local_spins(metals, oxidation_states=None, cluster_info=None):
    """Determine local spin S for each metal center.

    If oxidation_states are not provided, attempts to infer them from
    cluster charge balance. Falls back to the first common oxidation state.
    """
    spins = []
    for k, metal in enumerate(metals):
        if oxidation_states and k in oxidation_states:
            ox = oxidation_states[k]
        else:
            ox = _infer_oxidation_state(metal.element, k, metals, cluster_info)
        S = get_local_spin(metal.element, ox)
        spins.append(S)
    return spins


def _infer_oxidation_state(element, metal_idx, metals, cluster_info=None):
    """Infer the most likely oxidation state for a metal center.

    Strategy:
    1. If cluster_info has bridging/terminal ligands with known charges,
       use charge balance to solve for oxidation states.
    2. If only 1 metal and cluster charge is known, compute directly.
    3. Fall back to first common oxidation state.
    """
    states = get_common_oxidation_states(element)
    if not states:
        return 2

    if cluster_info is None:
        return states[0]

    # Simple case: single metal — charge balance gives oxidation directly
    if len(metals) == 1:
        # total_charge = metal_ox + sum(ligand_charges)
        # ligand_charges estimated from bridging/terminal atoms
        ligand_charge = _estimate_ligand_charge(cluster_info)
        target_ox = cluster_info.total_charge - ligand_charge
        # Find the closest common oxidation state
        best = min(states, key=lambda s: abs(s - target_ox))
        return best

    # Multiple metals: use first common state (can be improved later)
    return states[0]


def _estimate_ligand_charge(cluster_info):
    """Estimate total ligand charge from bridging atoms and terminal ligands."""
    total = 0
    for bridge in cluster_info.bridging_atoms:
        if bridge.element in ("S", "Se"):
            total += -2  # S²⁻
        elif bridge.element == "O":
            total += -2  # O²⁻
        elif bridge.element == "N":
            total += -3  # N³⁻
        elif bridge.element == "C":
            total += -4  # C⁴⁻ (interstitial)
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
    """Find the dominant rotation axis from metal positions.

    Uses PCA: the axis with the most variance or that best separates
    equivalent positions.
    """
    centered = positions - center
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Return the axis with largest eigenvalue
    return eigenvectors[:, -1]


def _generate_equivalence_maps(positions, center, axis, fold):
    """Generate atom index equivalence maps under n-fold rotation.

    Returns a list of dicts, each mapping original_atom_idx -> rotated_atom_idx.
    """
    n = len(positions)
    maps = []
    centered = positions - center

    for rot_idx in range(1, fold):
        angle = 2 * np.pi * rot_idx / fold
        rot_matrix = _rotation_matrix(axis, angle)
        rotated = (rot_matrix @ centered.T).T + center

        # Map each original atom to the closest rotated position
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
    # Compute a canonical key for each isomer based on minority-spin sets
    # and check equivalence under the symmetry operations
    visited = set()
    families = []

    for i, iso in enumerate(isomers):
        if i in visited:
            continue

        # Find all isomers equivalent to this one
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

        # Label the family
        n_minority = iso.n_minority
        fam = SpinIsomerFamily(
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
        fam = SpinIsomerFamily(
            label=f"BS{n_minority}",
            n_minority=n_minority,
            isomers=group,
            representative=group[0],
        )
        families.append(fam)

    return families


def _default_J_model(n_metals, connectivity=None):
    """Default antiferromagnetic J coupling model.

    If connectivity is provided, only coupled pairs have nonzero J.
    Otherwise, a uniform AFM model is used.
    """
    J = np.zeros((n_metals, n_metals))
    J_val = -1.0  # AFM coupling (negative = antiferromagnetic)

    if connectivity:
        for i, j in connectivity:
            J[i, j] = J_val
            J[j, i] = J_val
    else:
        # Uniform AFM coupling between all pairs
        for i in range(n_metals):
            for j in range(i + 1, n_metals):
                J[i, j] = J_val
                J[j, i] = J_val

    return J


def _load_spin_magnitudes(n_metals, cluster_info=None, oxidation_states=None):
    """Load spin magnitudes Si for each metal from the knowledge base.

    Returns a dict {metal_idx: spin_magnitude}.
    Falls back to S=2.0 for metals without data.
    """
    magnitudes = {}
    if cluster_info is None:
        return magnitudes

    for k, metal in enumerate(cluster_info.metals):
        if k >= n_metals:
            break
        if oxidation_states and k in oxidation_states:
            ox = oxidation_states[k]
        else:
            states = get_common_oxidation_states(metal.element)
            ox = states[0] if states else 2
        S = get_local_spin(metal.element, ox)
        magnitudes[k] = S

    return magnitudes
