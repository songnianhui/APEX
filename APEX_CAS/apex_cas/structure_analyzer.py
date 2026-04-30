"""Module 1: Structure Analyzer

Parse molecular structure files, identify metal centers, bridging atoms,
terminal ligands, and detect approximate symmetry for transition metal clusters.
"""

import os
from collections import Counter
from itertools import combinations

import numpy as np

from . import (
    BridgingAtom,
    ClusterInfo,
    MetalCenter,
    TerminalLigand,
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

TRANSITION_METALS = {
    # 3d
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

# Covalent radii in Angstroms (Cordero et al. 2008)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
    "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 1.87, "Hf": 1.75, "Ta": 1.70,
    "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
}

BRIDGING_ELEMENTS = {"S", "O", "N", "Se", "Cl", "P", "F", "Br", "I", "H"}

# Bond detection parameters
BOND_TOLERANCE = 1.3
MAX_METAL_LIGAND_DIST = 3.0
MAX_METAL_METAL_DIST = 3.5


# ──────────────────────────────────────────────────────────────────
# Structure parsing
# ──────────────────────────────────────────────────────────────────

def parse_structure(filepath: str, charge: int = 0, target_spin: float = 0.0,
                    custom_metals: list = None,
                    bond_tolerance: float = BOND_TOLERANCE,
                    max_metal_metal_dist: float = MAX_METAL_METAL_DIST,
                    max_metal_ligand_dist: float = MAX_METAL_LIGAND_DIST) -> ClusterInfo:
    """Parse a structure file and return a ClusterInfo object.

    Args:
        filepath: Path to structure file (XYZ, PDB, or any format ASE supports).
        charge: Total charge of the cluster.
        target_spin: Target total spin S quantum number.
        custom_metals: Optional list of element symbols to treat as metals,
            supplementing automatic detection. Useful for metalloid clusters
            or unusual compositions.
        bond_tolerance: Multiplier for covalent radii sum in bond detection.
        max_metal_metal_dist: Maximum distance (A) for metal-metal bonding.
        max_metal_ligand_dist: Maximum distance (A) for metal-ligand bonding.

    Returns:
        ClusterInfo with identified metals, bridging atoms, and ligands.
    """
    from ase.io import read as ase_read

    atoms = ase_read(filepath)
    elements = list(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    # Identify metal centers
    metals = _identify_metal_centers(elements, positions, custom_metals=custom_metals)

    # Build connectivity graph
    connectivity = _build_connectivity(elements, positions, bond_tolerance=bond_tolerance)

    # Set metal neighbors from connectivity
    for metal in metals:
        metal.neighbors = connectivity.get(metal.index, [])
        metal.coordination = len(metal.neighbors)

    # Identify bridging atoms
    bridging_atoms = _identify_bridging_atoms(elements, positions, metals, connectivity)

    # Identify terminal ligands
    terminal_ligands = _identify_terminal_ligands(elements, positions, metals, connectivity)

    # Generate formula
    formula = _generate_formula(elements)

    # Detect approximate symmetry
    symmetry_group, sym_axis_atoms = _detect_symmetry(metals, positions)

    return ClusterInfo(
        metals=metals,
        bridging_atoms=bridging_atoms,
        terminal_ligands=terminal_ligands,
        all_elements=elements,
        all_positions=np.array(positions),
        formula=formula,
        total_charge=charge,
        target_spin=target_spin,
        symmetry_group=symmetry_group,
        symmetry_axis_atoms=sym_axis_atoms,
    )


def _identify_metal_centers(elements, positions, custom_metals=None):
    """Identify all transition metal centers in the structure.

    Args:
        elements: List of element symbols.
        positions: List of (x, y, z) positions.
        custom_metals: Optional list of element symbols to additionally treat
            as metals, supplementing the automatic TRANSITION_METALS detection.
    """
    metal_set = set(TRANSITION_METALS)
    if custom_metals:
        metal_set.update(custom_metals)

    metals = []
    metal_count = {}  # track count per element for labeling
    for i, elem in enumerate(elements):
        if elem in metal_set:
            metal_count[elem] = metal_count.get(elem, 0) + 1
            label = f"{elem}{metal_count[elem]}"
            metals.append(MetalCenter(
                element=elem,
                index=i,
                position=np.array(positions[i]),
                label=label,
            ))
    return metals


def _bond_distance(r1, r2, tol=BOND_TOLERANCE):
    """Check if two atoms are bonded based on covalent radii."""
    r_a = COVALENT_RADII.get(r1[0], 1.5)
    r_b = COVALENT_RADII.get(r2[0], 1.5)
    dist = np.linalg.norm(r1[1] - r2[1])
    return dist <= (r_a + r_b) * tol


def _build_connectivity(elements, positions, bond_tolerance=BOND_TOLERANCE):
    """Build a simple connectivity graph based on interatomic distances."""
    n = len(elements)
    connectivity = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if _bond_distance((elements[i], positions[i]),
                              (elements[j], positions[j]), tol=bond_tolerance):
                connectivity[i].append(j)
                connectivity[j].append(i)

    return connectivity


def _identify_bridging_atoms(elements, positions, metals, connectivity):
    """Identify atoms that bridge two or more metal centers."""
    metal_indices = {m.index for m in metals}
    metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
    bridging_atoms = []

    for i, elem in enumerate(elements):
        if i in metal_indices:
            continue
        if elem not in BRIDGING_ELEMENTS:
            continue

        # Check which metals this atom is bonded to
        bonded_metals = [j for j in connectivity.get(i, []) if j in metal_indices]

        if len(bonded_metals) >= 2:
            role = "bridging"
            # Check for interstitial (inside the metal cage)
            if elem == "C" and len(bonded_metals) >= 4:
                role = "interstitial"

            bridging_atoms.append(BridgingAtom(
                element=elem,
                index=i,
                position=np.array(positions[i]),
                bridged_metals=[metal_idx_to_metal_pos[j] for j in bonded_metals],
                role=role,
            ))

    return bridging_atoms


def _identify_terminal_ligands(elements, positions, metals, connectivity):
    """Identify terminal ligands attached to metal centers."""
    metal_indices = {m.index for m in metals}
    metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
    terminal_ligands = []

    # Common charges for terminal donor atoms
    LIGAND_ATOM_CHARGES = {
        "F": -1, "Cl": -1, "Br": -1, "I": -1,
        "O": -2, "S": -2, "Se": -2,
        "N": -3,
        "C": -4,
        "H": +1,
    }

    for metal in metals:
        for neighbor_idx in metal.neighbors:
            if neighbor_idx in metal_indices:
                continue
            elem = elements[neighbor_idx]

            # Check if this neighbor is already a bridging atom (bridges to another metal)
            other_metal_neighbors = [
                j for j in connectivity.get(neighbor_idx, [])
                if j in metal_indices and j != metal.index
            ]
            if other_metal_neighbors:
                continue  # It's a bridging atom, not terminal

            lig_charge = LIGAND_ATOM_CHARGES.get(elem, 0)

            terminal_ligands.append(TerminalLigand(
                name=elem,
                atom_indices=[neighbor_idx],
                donor_atom_index=neighbor_idx,
                charge=lig_charge,
                metal_index=metal_idx_to_metal_pos[metal.index],
            ))

    return terminal_ligands


def _generate_formula(elements):
    """Generate a Hill-order chemical formula."""
    counts = Counter(elements)
    # Put C first, then H, then alphabetical
    parts = []
    if "C" in counts:
        parts.append(f"C{counts['C']}" if counts['C'] > 1 else "C")
        del counts["C"]
    if "H" in counts:
        parts.append(f"H{counts['H']}" if counts['H'] > 1 else "H")
        del counts["H"]

    for elem in sorted(counts):
        c = counts[elem]
        parts.append(f"{elem}{c}" if c > 1 else elem)

    return "".join(parts)


def _detect_symmetry(metals, positions):
    """Detect approximate point group symmetry from metal positions.

    Returns (symmetry_group, axis_atom_indices).
    Currently detects C3 and C4 axes through pairs of metal atoms.
    """
    if len(metals) < 2:
        return "C1", []

    # Try to detect rotational symmetry by looking for near-equivalent
    # metal positions around an axis defined by two metals
    best_sym = "C1"
    best_axis = []

    for i, j in combinations(range(len(metals)), 2):
        pos_i = metals[i].position
        pos_j = metals[j].position
        axis = pos_j - pos_i
        axis_len = np.linalg.norm(axis)
        if axis_len < 0.1:
            continue
        axis_hat = axis / axis_len

        # Project other metal positions onto the plane perpendicular to the axis
        # and check for rotational symmetry
        other_metals = [m for k, m in enumerate(metals) if k != i and k != j]

        if len(other_metals) == 0:
            continue

        # Find metals close to the midpoint plane
        mid = (pos_i + pos_j) / 2
        projections = []
        for m in other_metals:
            vec = m.position - mid
            along = np.dot(vec, axis_hat)
            perp = vec - along * axis_hat
            perp_len = np.linalg.norm(perp)
            if perp_len > 0.5:  # must be off-axis
                angle = np.arctan2(perp[1], perp[0]) if len(perp) >= 2 else 0.0
                projections.append((perp_len, angle, m))

        if len(projections) < 2:
            continue

        # Check for C3 symmetry (120° apart, similar radii)
        for fold in [3, 4]:
            n_matched = _count_rotational_matches(projections, fold, tol_angle=0.3, tol_r=0.3)
            if n_matched >= fold:
                sym_label = f"C{fold}"
                # Higher symmetry wins
                if _symmetry_order(sym_label) > _symmetry_order(best_sym):
                    best_sym = sym_label
                    best_axis = [i, j]

    return best_sym, best_axis


def _count_rotational_matches(projections, fold, tol_angle=0.3, tol_r=0.3):
    """Count how many positions form a rotational pattern of given fold."""
    if len(projections) < fold:
        return 0

    target_angle = 2 * np.pi / fold
    radii = [p[0] for p in projections]
    angles = sorted([p[1] for p in projections])

    # Check if we have `fold` points at similar radii with regular angular spacing
    # Use the median radius as reference
    if len(radii) < fold:
        return 0

    # Cluster by similar radius
    sorted_r = sorted(radii)
    # Find the largest group with similar radii
    best_count = 0
    for start in range(len(sorted_r)):
        count = 1
        for k in range(start + 1, len(sorted_r)):
            if sorted_r[k] - sorted_r[start] < tol_r:
                count += 1
            else:
                break
        best_count = max(best_count, count)

    if best_count >= fold:
        return fold
    return 0


def _symmetry_order(sym_label):
    """Return numeric order for symmetry comparison."""
    if sym_label.startswith("C") and len(sym_label) > 1 and sym_label[1:].isdigit():
        return int(sym_label[1:])
    return 1
