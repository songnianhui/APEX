"""Module: Orbital Optimizer

Advanced orbital optimization using UCCSD natural orbitals.
Provides an optional upgrade path from UNO-based to UCCSD-NO-based active spaces.
"""

import numpy as np

from . import CAS, ClusterInfo

from pyscf import ao2mo, cc, gto, lo, scf


def optimize_orbitals_uccsd_no(
    mol,
    mf,
    n_active_orbitals: int,
    cluster_info: ClusterInfo = None,
    frozen_core: bool = True,
) -> CAS:
    """Optimize active space orbitals using UCCSD natural orbitals.

    Runs UCCSD on top of the converged UHF/UKS, then constructs
    natural orbitals from the UCCSD one-particle density matrix.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object (UHF or UKS).
        n_active_orbitals: Number of active orbitals to select.
        cluster_info: Optional cluster info for labeling.
        frozen_core: Whether to freeze core orbitals.

    Returns:
        CAS with UCCSD-NO coefficients.
    """
    # Step 1: Run UCCSD
    nocc_a = mol.nelec[0]
    nocc_b = mol.nelec[1]
    nmo = mf.mo_coeff[0].shape[1]

    # Determine core orbitals to freeze
    ncore = 0
    if frozen_core:
        # Freeze all fully-occupied core (both alpha and beta)
        ncore = min(nocc_a, nocc_b)

    # Run UCCSD
    mycc = cc.UCCSD(mf, frozen=frozen_core)
    mycc.kernel()

    # Step 2: Get UCCSD one-particle density matrix
    dm1_a, dm1_b = mycc.make_rdm1()

    # Step 3: Construct natural orbitals from DM
    # For alpha:
    S = mol.intor_symmetric("int1e_ovlp")
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]

    # DM in MO basis -> diagonalize to get NOs
    # dm_mo = mo^T S dm_ao S mo, then diagonalize
    # For simplicity, use the total DM
    dm_total_ao = dm1_a + dm1_b

    # Generalized eigenvalue problem: S D S C = S C n
    SD = np.dot(S, dm_total_ao)
    SDS = np.dot(SD, S)
    # Solve standard eigenvalue decomposition of S D S
    SDS = 0.5 * (SDS + SDS.T)
    eigvals, eigvecs = np.linalg.eigh(SDS)

    # Sort by decreasing occupation
    idx = np.argsort(-eigvals)
    occupations = eigvals[idx]
    mo_coeff_no = eigvecs[:, idx]

    # Step 4: Select active subset
    n_active = n_active_orbitals
    # Select orbitals with occupations closest to 1.0
    dist_to_one = np.abs(occupations - 1.0)
    active_indices = np.argsort(dist_to_one)[:n_active]
    active_indices = sorted(active_indices)

    mo_active = mo_coeff_no[:, active_indices]
    occ_active = occupations[active_indices]
    n_electrons = int(round(sum(occ_active)))

    # Step 5: Generate labels
    labels = [f"UCCSD_NO_{i}" for i in range(len(active_indices))]
    if cluster_info is not None:
        # Try to assign chemical labels
        from .CAS_builder_computing import _assign_character_labels
        # Build a full coefficient matrix for labeling
        labels_placeholder = [""] * mo_coeff_no.shape[1]
        labels_full = _assign_character_labels(mol, mo_coeff_no, labels_placeholder, cluster_info)
        labels = [labels_full[i] for i in active_indices]

    return CAS(
        mo_coeff_alpha=mo_active.copy(),
        mo_coeff_beta=mo_active.copy(),
        occupations=occ_active,
        orbital_labels=labels,
        cpt_cas_type="uno",
        n_electrons=n_electrons,
        n_orbitals=len(active_indices),
        source_method="UCCSD-NO",
    )


def build_unrestricted_orbital_basis(
    mol,
    mf,
    n_active_orbitals: int,
    cluster_info: ClusterInfo = None,
    localization_method: str = "pm",
) -> CAS:
    """Build spin-unrestricted localized orbital basis for DMRG.

    Constructs alpha and beta localized orbital sets separately,
    suitable for spin-adapted DMRG calculations.

    Args:
        mol: PySCF Mole object.
        mf: Converged UHF object (must be UHF, not UKS).
        n_active_orbitals: Number of active orbitals to select.
        cluster_info: Optional cluster info for labeling.
        localization_method: "pm" or "boys".

    Returns:
        CAS with separate alpha/beta localized orbitals.
    """
    mo_alpha = mf.mo_coeff[0]
    mo_beta = mf.mo_coeff[1]
    n_alpha = mol.nelec[0]
    n_beta = mol.nelec[1]
    n_active = n_active_orbitals

    S = mol.intor_symmetric("int1e_ovlp")

    # Select localization function
    if localization_method == "pm":
        loc_fn = lo.PM
    else:
        loc_fn = lo.Boys

    # Localize occupied alpha
    def safe_localize(mol_obj, mo_block):
        if mo_block.shape[1] <= 1:
            return mo_block
        try:
            return loc_fn(mol_obj, mo_block).kernel()
        except Exception:
            try:
                fallback = lo.Boys if localization_method == "pm" else lo.PM
                return fallback(mol_obj, mo_block).kernel()
            except Exception:
                return mo_block

    loc_occ_a = safe_localize(mol, mo_alpha[:, :n_alpha])
    loc_vir_a = safe_localize(mol, mo_alpha[:, n_alpha:])
    loc_occ_b = safe_localize(mol, mo_beta[:, :n_beta])
    loc_vir_b = safe_localize(mol, mo_beta[:, n_beta:])

    all_loc_a = np.hstack([loc_occ_a, loc_vir_a])
    all_loc_b = np.hstack([loc_occ_b, loc_vir_b])

    # Select active subset by projection onto metal/bridging AOs
    if cluster_info is not None:
        from .CAS_builder_computing import _select_by_projection
        active_idx_a = _select_by_projection(mol, all_loc_a, n_active, cluster_info)
        active_idx_b = _select_by_projection(mol, all_loc_b, n_active, cluster_info)
    else:
        active_idx_a = list(range(min(n_active, all_loc_a.shape[1])))
        active_idx_b = list(range(min(n_active, all_loc_b.shape[1])))

    mo_active_a = all_loc_a[:, active_idx_a]
    mo_active_b = all_loc_b[:, active_idx_b]

    labels = [f"LUO_{i}" for i in range(len(active_idx_a))]

    return CAS(
        mo_coeff_alpha=mo_active_a.copy(),
        mo_coeff_beta=mo_active_b.copy(),
        occupations=None,
        orbital_labels=labels,
        cpt_cas_type="luo",
        n_electrons=mol.nelec[0] + mol.nelec[1],
        n_orbitals=len(active_idx_a),
        source_method=f"UHF/LUO-{localization_method}",
    )
