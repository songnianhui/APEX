"""Module 4b: Broken-Symmetry Guess Builder

Convert ElectronicConfig data objects into PySCF density matrices encoding
the broken-symmetry spin pattern and d-orbital occupancy.

Algorithm:
1. Start from high-spin atomic guess density matrix
2. For minority-spin metals: swap alpha↔beta rows+columns in the DM
3. Encode d-orbital occupancy for Fe(II) sites (extra electron placement)
"""

import numpy as np

from .models import ClusterInfo, ElectronicConfig

try:
    from pyscf import gto, scf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def build_bs_dm(cluster_info: ClusterInfo,
                electronic_config: ElectronicConfig,
                mol=None,
                basis_set: str = "cc-pVDZ") -> np.ndarray:
    """Build a broken-symmetry initial density matrix.

    Args:
        cluster_info: Cluster description.
        electronic_config: Electronic configuration specifying spin pattern
            and d-orbital choices.
        mol: Optional pre-built PySCF Mole object. If None, built from cluster_info.
        basis_set: Basis set (used only if mol is None).

    Returns:
        Density matrix array of shape (2, nao, nao) for alpha and beta channels.
    """
    if not HAS_PYSCF:
        raise ImportError("PySCF is required for BS guess construction.")

    if mol is None:
        mol = _build_mol(cluster_info, basis_set)

    # Step 1: Get atomic guess density matrix
    mf = scf.UHF(mol)
    dm = mf.get_init_guess(key="atom")  # shape (2, nao, nao)
    dm_a = dm[0].copy()
    dm_b = dm[1].copy()

    # Step 2: Swap alpha↔beta for minority-spin metal sites
    if electronic_config.spin_assignment:
        for metal_idx, spin_dir in electronic_config.spin_assignment.items():
            if spin_dir == -1:
                atom_idx = cluster_info.metals[metal_idx].index
                dm_a, dm_b = _swap_spin_block(mol, dm_a, dm_b, atom_idx)

    # Step 3: Encode d-orbital occupancy choices
    if electronic_config.d_orbital_assignments:
        for metal_idx, d_choice in electronic_config.d_orbital_assignments.items():
            atom_idx = cluster_info.metals[metal_idx].index
            spin_dir = electronic_config.spin_assignment.get(metal_idx, +1)
            dm_a, dm_b = _encode_d_orbital_occupancy(
                mol, dm_a, dm_b, atom_idx, d_choice, spin_dir
            )

    return np.array([dm_a, dm_b])


def build_bs_dm_batch(cluster_info: ClusterInfo,
                       electronic_configs: list,
                       mol=None,
                       basis_set: str = "cc-pVDZ") -> list:
    """Build BS density matrices for a batch of electronic configurations.

    Reuses the Mole object and atomic guess for efficiency.

    Args:
        cluster_info: Cluster description.
        electronic_configs: List of ElectronicConfig objects.
        mol: Optional pre-built PySCF Mole object.
        basis_set: Basis set.

    Returns:
        List of density matrix arrays, each (2, nao, nao).
    """
    if not HAS_PYSCF:
        raise ImportError("PySCF is required.")

    if mol is None:
        mol = _build_mol(cluster_info, basis_set)

    # Get base atomic guess once
    mf = scf.UHF(mol)
    dm_base = mf.get_init_guess(key="atom")

    results = []
    for config in electronic_configs:
        dm_a = dm_base[0].copy()
        dm_b = dm_base[1].copy()

        # Apply spin flips
        for metal_idx, spin_dir in config.spin_assignment.items():
            if spin_dir == -1:
                atom_idx = cluster_info.metals[metal_idx].index
                dm_a, dm_b = _swap_spin_block(mol, dm_a, dm_b, atom_idx)

        # Apply d-orbital choices
        for metal_idx, d_choice in config.d_orbital_assignments.items():
            atom_idx = cluster_info.metals[metal_idx].index
            spin_dir = config.spin_assignment.get(metal_idx, +1)
            dm_a, dm_b = _encode_d_orbital_occupancy(
                mol, dm_a, dm_b, atom_idx, d_choice, spin_dir
            )

        results.append(np.array([dm_a, dm_b]))

    return results


# ──────────────────────────────────────────────────────────────────
# Spin block manipulation
# ──────────────────────────────────────────────────────────────────

def _swap_spin_block(mol, dm_a, dm_b, atom_idx):
    """Swap alpha↔beta density matrix elements for a specific atom.

    This flips the spin direction of the atom by exchanging all
    AO rows and columns associated with that atom between the
    alpha and beta density matrices.
    """
    ao_s, ao_e = _get_ao_slice(mol, atom_idx)

    # Swap full rows
    tmp_a_rows = dm_a[ao_s:ao_e, :].copy()
    tmp_b_rows = dm_b[ao_s:ao_e, :].copy()
    dm_a[ao_s:ao_e, :] = tmp_b_rows
    dm_b[ao_s:ao_e, :] = tmp_a_rows

    # Swap full columns
    tmp_a_cols = dm_a[:, ao_s:ao_e].copy()
    tmp_b_cols = dm_b[:, ao_s:ao_e].copy()
    dm_a[:, ao_s:ao_e] = tmp_b_cols
    dm_b[:, ao_s:ao_e] = tmp_a_cols

    return dm_a, dm_b


def _encode_d_orbital_occupancy(mol, dm_a, dm_b, atom_idx, d_choice, spin_dir):
    """Encode the choice of which d-orbital gets the extra electron.

    For a metal with a partially-filled d shell (e.g., Fe(II) d6 high-spin),
    the extra electron (beyond half-filling) goes into a specific d-orbital.

    Args:
        mol: PySCF Mole object.
        dm_a, dm_b: Alpha/beta density matrices.
        atom_idx: Atom index in the structure.
        d_choice: Which d-orbital (0-4) gets the extra electron.
        spin_dir: +1 (majority) or -1 (minority) for this metal.
    """
    d_indices = get_d_orbital_indices(mol, atom_idx)

    if not d_indices or d_choice >= len(d_indices):
        return dm_a, dm_b

    chosen_ao = d_indices[d_choice]

    # Add extra electron to the minority-spin channel at this d-orbital
    # For majority-spin metal: extra e- goes into beta
    # For minority-spin metal: extra e- goes into alpha (already swapped)
    if spin_dir == +1:
        dm_b[chosen_ao, chosen_ao] += 1.0
    else:
        dm_a[chosen_ao, chosen_ao] += 1.0

    return dm_a, dm_b


def get_d_orbital_indices(mol, atom_idx):
    """Find d-orbital AO indices for a specific atom.

    PySCF spherical harmonic ordering for l=2:
    dxy, dyz, dz2, dxz, dx2-y2 (5 functions)

    Args:
        mol: PySCF Mole object.
        atom_idx: Atom index (0-based).

    Returns:
        List of AO indices for the d-orbitals of this atom.
    """
    aoslices = mol.aoslice_by_atom()
    if atom_idx >= len(aoslices):
        return []

    _, _, ao_s, ao_e = aoslices[atom_idx]
    ao_labels = mol.ao_labels()

    d_indices = []
    for i in range(ao_s, ao_e):
        if i < len(ao_labels):
            label = ao_labels[i]
            parts = label.split()
            # PySCF label format: "atom_idx element orb_type"
            # Check for 'd' in the orbital type part
            if len(parts) >= 3 and 'd' in parts[-1].lower():
                d_indices.append(i)

    return d_indices


def get_p_orbital_indices(mol, atom_idx):
    """Find p-orbital AO indices for a specific atom.

    Args:
        mol: PySCF Mole object.
        atom_idx: Atom index (0-based).

    Returns:
        List of AO indices for the p-orbitals.
    """
    aoslices = mol.aoslice_by_atom()
    if atom_idx >= len(aoslices):
        return []

    _, _, ao_s, ao_e = aoslices[atom_idx]
    ao_labels = mol.ao_labels()

    p_indices = []
    for i in range(ao_s, ao_e):
        if i < len(ao_labels):
            label = ao_labels[i]
            parts = label.split()
            if len(parts) >= 3 and 'p' in parts[-1].lower():
                p_indices.append(i)

    return p_indices


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _build_mol(cluster_info, basis_set):
    """Build PySCF Mole object from ClusterInfo."""
    atoms = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        atoms.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    spin_2s = int(round(2 * cluster_info.target_spin))

    mol = gto.M(
        atom="\n".join(atoms),
        charge=cluster_info.total_charge,
        spin=spin_2s,
        basis=basis_set,
        symmetry=False,
        verbose=0,
    )
    mol.build()
    return mol


def _get_ao_slice(mol, atom_idx):
    """Get (ao_start, ao_end) for an atom."""
    aoslices = mol.aoslice_by_atom()
    _, _, ao_s, ao_e = aoslices[atom_idx]
    return ao_s, ao_e
