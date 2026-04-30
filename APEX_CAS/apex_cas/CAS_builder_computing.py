"""Computation-Based CAS Builder

Orchestrates all computation-driven CAS (Complete Active Space) construction:
  1. Build molecule with basis set and run high-spin SCF
  2. AVAS (Automated Valence Active Space) projection
  3. Compute unrestricted natural orbitals (UNOs)
  4. Split-localize the UNOs by occupation blocks
  5. Select active orbitals by character (projection onto AO subsets)
  6. (LUO variant) Localize alpha/beta orbitals separately for DMRG

Public API:
  - init_computing(): Build mol + run SCF + save chkfile
  - build_computed_CAS(): Unified entry point for all cpt_cas_types
"""

import os
import warnings

import numpy as np
import yaml
import pyscf
from scipy.linalg import eigh as _gen_eigh

from . import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ComputationSettings,
    OrbitalGroup,
    AVASConfig,
)

from pyscf import dft, gto, lo, scf

# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def build_computed_CAS(
    cluster_info: ClusterInfo,
    computation_settings: ComputationSettings = None,
    cpt_cas_type: str = "uno",
    localization_method: str = "pm",
    avas_config: AVASConfig = None,
    save_dir: str = ".",
) -> tuple:
    """Build CAS using computation-driven methods.

    Unified entry point for all computing-based CAS construction.
    Initializes the SCF calculation and dispatches to the appropriate
    orbital construction method.

    Args:
        cluster_info: Cluster description.
        computation_settings: SCF parameters. None produces defaults.
        cpt_cas_type: One of "uno", "luo", or "avas".
        localization_method: "pm" (Pipek-Mezey) or "boys".
        avas_config: Configuration for AVAS method (required if cpt_cas_type="avas").
        save_dir: Directory for checkpoint files.

    Returns:
        Tuple of (CAS, mol, mf, chkfile_path) where:
        - CAS: Active space result with MO coefficients and metadata
        - mol: PySCF Mole object
        - mf: Converged SCF object
        - chkfile_path: Path to saved checkpoint file
    """
    if computation_settings is None:
        computation_settings = ComputationSettings()

    # Initialize: build molecule + run SCF + save chkfile
    mol, mf, chkfile_path = init_computing(
        cluster_info, computation_settings, save_dir
    )

    # Build source method string from settings
    scf_method = computation_settings.scf_method.upper()
    xc = computation_settings.xc_functional
    source_prefix = f"UKS-{xc}" if scf_method == "UKS" else "UHF"

    # Dispatch by orbital type
    if cpt_cas_type == "avas":
        cas, expected_types = _construct_avas(mol, mf, cluster_info, avas_config)
    elif cpt_cas_type == "luo":
        cas = _construct_luo(
            mol, mf, cluster_info,
            localization_method=localization_method,
            source_prefix=source_prefix,
        )
    else:  # "uno" (default)
        cas = _construct_uno(mol, mf, cluster_info, source_prefix=source_prefix)

    return cas, mol, mf, chkfile_path

# ──────────────────────────────────────────────────────────────────
# Initialization: build molecule + run SCF + save checkpoint
# ──────────────────────────────────────────────────────────────────
def _make_chkfile_name(cluster_info: ClusterInfo,
                       settings: ComputationSettings) -> str:
    """Generate a descriptive checkpoint filename."""
    formula = getattr(cluster_info, "formula", "cluster")
    method = settings.scf_method
    xc = settings.xc_functional or "none"
    basis = settings.basis_set_default or "unknown"
    name = f"{formula}_{method}_{xc}_{basis}.chk"
    return name.replace(" ", "_").replace("|", "_")

def init_computing(
    cluster_info: ClusterInfo,
    computation_settings: ComputationSettings,
    save_dir: str = ".",
) -> tuple:
    """Build molecule, run high-spin SCF, and save checkpoint.

    This is the standard initialization step for all computing-based
    CAS construction methods. It builds the PySCF molecule object,
    runs a high-spin SCF calculation, and saves the results to a
    checkpoint file for later reuse.

    Args:
        cluster_info: Cluster description from structure analysis.
        computation_settings: SCF computation parameters.
        save_dir: Directory to save the checkpoint file (default: current directory).

    Returns:
        Tuple of (mol, mf, chkfile_path) where:
        - mol: PySCF Mole object
        - mf: Converged SCF object
        - chkfile_path: Path to the saved checkpoint file
    """
    
    pyscf.lib.num_threads(8)
    
    mol = build_mol_with_basis(cluster_info, computation_settings)

    # Set chkfile path before kernel so PySCF auto-saves
    filename = _make_chkfile_name(cluster_info, computation_settings)
    os.makedirs(save_dir, exist_ok=True)
    chkfile_path = os.path.join(save_dir, filename)

    # Run SCF
    mf = _run_high_spin_scf(mol, computation_settings, chkfile_path)

    return mol, mf, chkfile_path

# ──────────────────────────────────────────────────────────────────
# UNO pipeline
# ──────────────────────────────────────────────────────────────────
def _construct_uno(mol, mf, cluster_info, source_prefix="UHF"):
    """Restricted UNO pipeline: UKS → UNO → split-localize → select."""
    # Step 2: Compute UNOs from alpha+beta density matrix
    mo_coeff_uno, occ_uno = compute_unos(mol, mf)

    # Step 3: Split-localize by occupation blocks
    mo_coeff_loc, orbital_labels = split_localize(
        mol, mo_coeff_uno, occ_uno, cluster_info
    )

    # Step 4: Select active orbitals by NOON threshold
    occ_lo, occ_hi = 0.02, 1.98
    active_indices = [i for i in range(len(occ_uno))
                      if occ_lo <= occ_uno[i] <= occ_hi]

    # Extract active orbitals
    mo_active = mo_coeff_loc[:, active_indices]
    occ_active = occ_uno[active_indices]
    labels_active = [orbital_labels[i] for i in active_indices]

    n_electrons = int(round(sum(occ_active)))

    return CAS(
        mo_coeff_alpha=mo_active.copy(),
        mo_coeff_beta=mo_active.copy(),  # Same for restricted
        occupations=occ_active,
        orbital_labels=labels_active,
        cpt_cas_type="uno",
        n_electrons=n_electrons,
        n_orbitals=len(active_indices),
        source_method=f"{source_prefix}/UNO",
        mo_coeff_full=mo_coeff_loc,
        occupations_full=occ_uno,
        orbital_labels_full=orbital_labels,
    )

# ──────────────────────────────────────────────────────────────────
# LUO pipeline
# ──────────────────────────────────────────────────────────────────

def _construct_luo(mol, mf, cluster_info,
                    localization_method: str = "pm",
                    source_prefix="UHF"):
    """Unrestricted LUO pipeline: localize alpha and beta separately.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object (UHF or UKS).
        cluster_info: ClusterInfo for projection-based selection.
        localization_method: "pm" (Pipek-Mezey) or "boys".
    """
    S = mol.intor_symmetric("int1e_ovlp")

    # Get alpha and beta MOs
    mo_alpha = mf.mo_coeff[0]
    mo_beta = mf.mo_coeff[1]

    # Determine occupation
    n_alpha = mol.nelec[0]
    n_beta = mol.nelec[1]

    n_occ_a = n_alpha
    n_occ_b = n_beta

    n_tot = mo_alpha.shape[1]

    occ_alpha = np.zeros(n_tot)
    occ_alpha[:n_occ_a] = 1.0
    occ_beta = np.zeros(n_tot)
    occ_beta[:n_occ_b] = 1.0

    # Localize alpha occupied
    loc_a = _localize_orbitals(mol, mo_alpha[:, :n_occ_a], S, method=localization_method)
    # Localize alpha virtual
    loc_a_vir = _localize_orbitals(mol, mo_alpha[:, n_occ_a:], S, method=localization_method)

    # Localize beta occupied
    loc_b = _localize_orbitals(mol, mo_beta[:, :n_occ_b], S, method=localization_method)
    # Localize beta virtual
    loc_b_vir = _localize_orbitals(mol, mo_beta[:, n_occ_b:], S, method=localization_method)

    # Select active subset from localized orbitals
    # Combine occupied and virtual, then select by character
    all_loc_a = np.hstack([loc_a, loc_a_vir])
    all_loc_b = np.hstack([loc_b, loc_b_vir])

    # Select active orbitals by projection onto metal-d + bridging-p
    if cluster_info is not None:
        active_idx_a = _select_by_projection_threshold(
            mol, all_loc_a, cluster_info
        )
        active_idx_b = _select_by_projection_threshold(
            mol, all_loc_b, cluster_info
        )
    else:
        n_tot = all_loc_a.shape[1]
        active_idx_a = list(range(n_tot))
        active_idx_b = list(range(n_tot))

    mo_active_a = all_loc_a[:, active_idx_a]
    mo_active_b = all_loc_b[:, active_idx_b]

    # Generate labels
    labels_a = [f"LUO_a_{i}" for i in range(len(active_idx_a))]
    labels_b = [f"LUO_b_{i}" for i in range(len(active_idx_b))]
    labels = labels_a  # Use alpha labels as primary

    return CAS(
        mo_coeff_alpha=mo_active_a.copy(),
        mo_coeff_beta=mo_active_b.copy(),
        occupations=None,  # LUO doesn't have UNO occupations
        orbital_labels=labels,
        cpt_cas_type="luo",
        n_electrons=mol.nelec[0] + mol.nelec[1],
        n_orbitals=len(active_idx_a),
        source_method=f"{source_prefix}/LUO",
        mo_coeff_full=all_loc_a,
        occupations_full=occ_alpha,
        orbital_labels_full=labels,
    )

# ──────────────────────────────────────────────────────────────────
# Core computational functions
# ──────────────────────────────────────────────────────────────────

def compute_unos(mol, mf):
    """Compute unrestricted natural orbitals from UKS/UHF density matrix.

    Diagonalizes the total (alpha+beta) 1-RDM in the AO basis.

    Returns:
        (mo_coeff_uno, occupations) — natural orbital coefficients and
        occupation numbers, sorted by decreasing occupation.
    """
    dm_alpha, dm_beta = mf.make_rdm1()
    dm_total = dm_alpha + dm_beta

    S = mol.intor_symmetric("int1e_ovlp")

    # Solve generalized eigenvalue problem: S·D·S·C = S·C·n
    SD = np.dot(S, dm_total)
    SDS = np.dot(SD, S)

    eigvals, eigvecs = _gen_eigh(SDS, S)

    # Sort by decreasing occupation
    idx = np.argsort(-eigvals)
    occupations = eigvals[idx]
    mo_coeff_uno = eigvecs[:, idx]

    return mo_coeff_uno, occupations

def split_localize(mol, mo_coeff, occupations,
                    cluster_info=None,
                    occ_threshold_core: float = 1.98,
                    occ_threshold_virtual: float = 0.02,
                    method: str = "pm"):
    """Split-localize UNOs by occupation blocks.

    Partitions orbitals into core (occ > 1.98), active (0.02 < occ < 1.98),
    and virtual (occ < 0.02) blocks. Applies localization separately
    to each block.

    Args:
        mol: PySCF Mole object.
        mo_coeff: MO coefficient matrix.
        occupations: UNO occupation numbers.
        cluster_info: Optional cluster info for labeling.
        occ_threshold_core: Occupation threshold for core orbitals.
        occ_threshold_virtual: Occupation threshold for virtual orbitals.
        method: Localization method: "pm" (Pipek-Mezey, default) or
            "boys" (Boys).

    Returns:
        (localized_mo_coeff, orbital_labels)
    """
    S = mol.intor_symmetric("int1e_ovlp")
    n_orb = len(occupations)

    # Partition into blocks
    core_idx = [i for i in range(n_orb) if occupations[i] > occ_threshold_core]
    active_idx = [i for i in range(n_orb)
                  if occ_threshold_virtual <= occupations[i] <= occ_threshold_core]
    virtual_idx = [i for i in range(n_orb) if occupations[i] < occ_threshold_virtual]

    localized = mo_coeff.copy()
    labels = [""] * n_orb

    # Localize each block separately
    for block_name, block_idx in [("core", core_idx),
                                   ("active", active_idx),
                                   ("virtual", virtual_idx)]:
        if len(block_idx) <= 1:
            for i in block_idx:
                labels[i] = f"{block_name}_{i}"
            continue

        mo_block = mo_coeff[:, block_idx]
        loc_block = _localize_orbitals(mol, mo_block, S, method=method)
        localized[:, block_idx] = loc_block

        for k, i in enumerate(block_idx):
            labels[i] = f"{block_name}_{k}"

    # Add atomic character labels if cluster_info available
    if cluster_info is not None:
        labels = _assign_character_labels(mol, localized, labels, cluster_info)

    return localized, labels

def select_active_orbitals(mol, mo_coeff, occupations,
                            n_active: int, cluster_info=None):
    """Select n_active orbitals from localized UNOs.

    Selection strategy:
    1. All orbitals in the active occupation range (0.02–1.98) are included.
    2. If more are needed, add from core/virtual by projection character.
    3. If fewer are needed (shouldn't happen normally), trim by occupation.

    Args:
        mol: PySCF Mole object.
        mo_coeff: Localized MO coefficients.
        occupations: UNO occupation numbers.
        n_active: Target number of active orbitals.
        cluster_info: Optional cluster info for projection-based selection.

    Returns:
        List of orbital indices to include in the active space.
    """
    n_orb = len(occupations)

    # Active occupation range
    active_occ = [i for i in range(n_orb)
                  if 0.02 <= occupations[i] <= 1.98]

    if len(active_occ) == n_active:
        return active_occ

    if len(active_occ) > n_active:
        # Too many — keep those with occupation closest to 1.0
        active_occ.sort(key=lambda i: abs(occupations[i] - 1.0))
        return active_occ[:n_active]

    # Not enough — add from core/virtual by projection onto metal/bridging AOs
    if cluster_info is not None:
        return _select_by_projection(mol, mo_coeff, n_active, cluster_info)

    # Fallback: add by closest occupation to 1.0
    remaining = [i for i in range(n_orb) if i not in active_occ]
    remaining.sort(key=lambda i: abs(occupations[i] - 1.0))
    needed = n_active - len(active_occ)
    return active_occ + remaining[:needed]

# ──────────────────────────────────────────────────────────────────
# PySCF interface helpers
# ──────────────────────────────────────────────────────────────────

def _build_mol(cluster_info, basis_set):
    """Build PySCF Mole object from ClusterInfo."""
    atoms = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        atoms.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    # Determine spin (2*Sz) from target_spin
    spin_2s = int(round(2 * cluster_info.target_spin))

    mol = gto.M(
        atom="\n".join(atoms),
        charge=cluster_info.total_charge,
        spin=spin_2s,
        basis=basis_set,
        symmetry=False,
        verbose=4,
    )
    mol.build()
    return mol

def build_mol_with_basis(cluster_info, settings: ComputationSettings):
    """Build PySCF Mole object with per-element basis set support.

    Uses settings.get_basis(element) for each element in the cluster.

    Args:
        cluster_info: ClusterInfo object.
        settings: ComputationSettings with basis configuration.

    Returns:
        Built PySCF Mole object.
    """
    from .computation_defaults import build_basis_dict

    atoms = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        atoms.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    spin_2s = int(round(2 * cluster_info.target_spin))
    basis_dict = build_basis_dict(cluster_info, settings)

    mol = gto.M(
        atom="\n".join(atoms),
        charge=cluster_info.total_charge,
        spin=spin_2s,
        basis=basis_dict,
        symmetry=False,
        verbose=1,  # reduced verbosity to avoid excessive output in production
    )
    mol.build()
    return mol

def _run_high_spin_scf(mol, settings: ComputationSettings, chkfile_path: str = None):
    """Run high-spin SCF calculation with two-stage convergence.

    Stage 1 (rough): looser conv_tol, guaranteed level_shift ≥ 0.1 for stability.
    Stage 2 (precise): user-specified conv_tol and level_shift, using Stage 1
    density as initial guess.

    Args:
        mol: PySCF Mole object.
        settings: ComputationSettings with scf parameters and convergence helpers.
        chkfile_path: Path to save checkpoint file.

    Returns:
        Converged SCF object.
    """
    # ── Build mf object ──
    mf = _build_mf_object(mol, settings)

    # ── Apply convergence helpers ──
    mf.init_guess = settings.init_guess
    if settings.scf_damp > 0:
        mf.damp = settings.scf_damp
    if settings.scf_level_shift > 0:
        mf.level_shift = settings.scf_level_shift
    mf.diis_space = settings.diis_space

    # Set chkfile before kernel so PySCF auto-saves
    if chkfile_path:
        mf.chkfile = chkfile_path

    # ── Stage 1: rough convergence ──
    stage1_tol = max(1e-4, settings.conv_tol * 100)
    stage1_shift = max(settings.scf_level_shift, 0.1)
    stage1_cycle = max(settings.max_cycle // 2, 50)

    print(f"\n  SCF Stage 1: conv_tol={stage1_tol:.1e}, level_shift={stage1_shift:.2f}, "
          f"max_cycle={stage1_cycle}")
    mf.conv_tol = stage1_tol
    mf.level_shift = stage1_shift
    mf.max_cycle = stage1_cycle
    mf.verbose = settings.scf_verbose
    mf.kernel()

    # ── Stage 2: precise convergence (only if Stage 1 didn't meet target tol) ──
    if not mf.converged or mf.conv_tol > settings.conv_tol:
        print(f"\n  SCF Stage 2: conv_tol={settings.conv_tol:.1e}, level_shift={settings.scf_level_shift:.2f}, "
              f"max_cycle={settings.max_cycle}")
        mf.conv_tol = settings.conv_tol
        mf.level_shift = settings.scf_level_shift
        mf.max_cycle = settings.max_cycle
        # Keep damp and diis_space from Stage 1
        mf.kernel()

    # ── Check convergence ──
    if not mf.converged:
        print(f"\n{'!' * 60}")
        print("  WARNING: SCF did NOT converge!")
        print(f"  Energy: {mf.e_tot:.10f} Hartree")
        print(f"  Cycles: {mf.cycles}")
        print(f"{'!' * 60}")
        resp = input("  Continue with unconverged result? (y/n): ").strip().lower()
        if resp != "y":
            raise RuntimeError("SCF did not converge. Aborting at user request.")

    return mf


def _build_mf_object(mol, settings: ComputationSettings):
    """Build the SCF mean-field object with relativistic and solvation corrections.

    Returns the mf object without setting convergence parameters, so that the
    caller can configure two-stage convergence separately.
    """
    # Base SCF object
    if settings.scf_method == "uhf":
        mf = scf.UHF(mol)
    elif settings.scf_method == "uks":
        mf = dft.UKS(mol)
        mf.xc = settings.xc_functional
    else:
        raise ValueError(f"Unknown scf_method: {settings.scf_method}")

    # Relativistic correction (replaces base mf)
    if settings.relativistic == "sf-x2c":
        from pyscf import x2c
        if settings.scf_method == "uks":
            mf = x2c.UKS(mol)
            mf.xc = settings.xc_functional
        else:
            mf = x2c.UHF(mol)
    elif settings.relativistic == "dkh":
        mol.set(float("DKH"))
        if settings.scf_method == "uks":
            mf = dft.UKS(mol)
            mf.xc = settings.xc_functional
        else:
            mf = scf.UHF(mol)

    # Solvation model (wraps mf)
    if settings.solvation_model == "ddcosmo":
        from pyscf import solvent
        mf = solvent.ddcosmo.ddcosmo_for_scf(mf)
        mf.with_solvent.epsilon = settings.solvation_epsilon

    return mf


def _localize_orbitals(mol, mo_coeff_block, S, method="pm"):
    """Apply localization to a block of orbitals.

    Args:
        mol: PySCF Mole object.
        mo_coeff_block: MO coefficient matrix block.
        S: Overlap matrix.
        method: "pm" (Pipek-Mezey, default) or "boys" (Boys).
            Falls back to the other method if the primary fails.

    Returns:
        Localized MO coefficient block.
    """
    n_orb = mo_coeff_block.shape[1]
    if n_orb <= 1:
        return mo_coeff_block

    primary, fallback = (lo.PM, lo.Boys) if method == "pm" else (lo.Boys, lo.PM)

    try:
        loc = primary(mol, mo_coeff_block)
        return loc.kernel()
    except Exception:
        try:
            loc = fallback(mol, mo_coeff_block)
            return loc.kernel()
        except Exception:
            return mo_coeff_block


def _select_by_projection(mol, mo_coeff, n_active, cluster_info):
    """Select active orbitals by projecting onto metal-d + bridging-p AO subsets."""
    metal_indices = {m.index for m in cluster_info.metals}
    bridge_indices = {b.index for b in cluster_info.bridging_atoms}
    target_atoms = metal_indices | bridge_indices

    # Get AO slices for target atoms
    aoslices = mol.aoslice_by_atom()
    target_ao_indices = []
    for atom_idx in target_atoms:
        if atom_idx < len(aoslices):
            _, _, ao_start, ao_end = aoslices[atom_idx]
            target_ao_indices.extend(range(ao_start, ao_end))

    if not target_ao_indices:
        return list(range(min(n_active, mo_coeff.shape[1])))

    # Compute projection: sum of |<MO|AO_target>|^2 for each MO
    target_ao = np.array(target_ao_indices)
    S = mol.intor_symmetric("int1e_ovlp")

    # Project each MO onto the target AO subspace
    n_mo = mo_coeff.shape[1]
    projections = np.zeros(n_mo)
    for i in range(n_mo):
        mo_i = mo_coeff[:, i]
        for ao_j in target_ao:
            e_j = np.zeros(mol.nao_nr())
            e_j[ao_j] = 1.0
            projections[i] += abs(np.dot(mo_i, np.dot(S, e_j))) ** 2

    # Select top n_active by projection weight
    selected = np.argsort(-projections)[:n_active]
    return sorted(selected.tolist())

def _select_by_projection_threshold(mol, mo_coeff, cluster_info,
                                     threshold: float = 0.05):
    """Select orbitals with projection weight > threshold onto metal-d + bridging-p.

    Unlike ``_select_by_projection`` which returns exactly *n_active* orbitals,
    this function returns all orbitals whose projection weight onto the target
    AO subspace exceeds *threshold*.

    Args:
        mol: PySCF Mole object.
        mo_coeff: MO coefficient matrix.
        cluster_info: ClusterInfo with metals and bridging_atoms.
        threshold: Minimum projection weight (default 0.05).

    Returns:
        Sorted list of orbital indices.
    """
    metal_indices = {m.index for m in cluster_info.metals}
    bridge_indices = {b.index for b in cluster_info.bridging_atoms}
    target_atoms = metal_indices | bridge_indices

    aoslices = mol.aoslice_by_atom()
    target_ao_indices = []
    for atom_idx in target_atoms:
        if atom_idx < len(aoslices):
            _, _, ao_start, ao_end = aoslices[atom_idx]
            target_ao_indices.extend(range(ao_start, ao_end))

    if not target_ao_indices:
        return list(range(mo_coeff.shape[1]))

    target_ao = np.array(target_ao_indices)
    S = mol.intor_symmetric("int1e_ovlp")

    n_mo = mo_coeff.shape[1]
    projections = np.zeros(n_mo)
    for i in range(n_mo):
        mo_i = mo_coeff[:, i]
        for ao_j in target_ao:
            e_j = np.zeros(mol.nao_nr())
            e_j[ao_j] = 1.0
            projections[i] += abs(np.dot(mo_i, np.dot(S, e_j))) ** 2

    selected = [int(i) for i in range(n_mo) if projections[i] > threshold]
    return sorted(selected)

def _assign_character_labels(mol, mo_coeff, base_labels, cluster_info):
    """Assign character labels (e.g., 'Fe1_dxy', 'S3_px') to localized orbitals."""
    metal_map = {m.index: m.label for m in cluster_info.metals}
    bridge_map = {b.index: f"{b.element}{b.index}" for b in cluster_info.bridging_atoms}
    target_atoms = dict(metal_map)
    target_atoms.update(bridge_map)

    aoslices = mol.aoslice_by_atom()
    ao_labels = mol.ao_labels()

    for i in range(mo_coeff.shape[1]):
        mo_i = mo_coeff[:, i]

        # Find the atom with largest contribution
        best_atom = -1
        best_contrib = 0.0

        for atom_idx in target_atoms:
            if atom_idx < len(aoslices):
                _, _, ao_s, ao_e = aoslices[atom_idx]
                contrib = np.sum(mo_i[ao_s:ao_e] ** 2)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_atom = atom_idx

        if best_atom >= 0 and best_contrib > 0.1:
            # Get the dominant AO type
            _, _, ao_s, ao_e = aoslices[best_atom]
            local_coeffs = mo_i[ao_s:ao_e] ** 2
            dominant_local = np.argmax(local_coeffs)
            dominant_label = ao_labels[ao_s + dominant_local] if ao_s + dominant_local < len(ao_labels) else ""

            atom_label = target_atoms.get(best_atom, f"atom{best_atom}")
            # Extract orbital type from AO label
            parts = dominant_label.split()
            orb_type = parts[-1] if len(parts) > 0 else ""

            base_labels[i] = f"{atom_label}_{orb_type}" if orb_type else f"{atom_label}_orb"

    return base_labels

# ──────────────────────────────────────────────────────────────────
# AVAS (Automated Valence Active Space) section
# Based on Sayfutyarova et al., JCTC 2017.
# Uses PySCF's built-in AVAS module with manual fallback.
# ──────────────────────────────────────────────────────────────────

# Knowledge base loading (shared pattern with active_space_builder)

from ._paths import data_file as _data_file

_avas_metals_db = None
_avas_ligands_db = None


def _load_yaml(filename):
    path = _data_file(filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _get_metals_db():
    global _avas_metals_db
    if _avas_metals_db is None:
        _avas_metals_db = _load_yaml("transition_metals.yaml")
    return _avas_metals_db

def _get_ligands_db():
    global _avas_ligands_db
    if _avas_ligands_db is None:
        _avas_ligands_db = _load_yaml("ligand_database.yaml")
    return _avas_ligands_db

# Period-table helpers for valence s orbital inference

# Maps the d-series (3d, 4d, 5d) to the corresponding valence s shell.
_VALENCE_S_SHELL = {
    "3d": "4s",
    "4d": "5s",
    "5d": "6s",
}

def _valence_s_for_element(element: str) -> str | None:
    """Return the valence s orbital label for a transition metal (e.g. '4s' for Fe)."""
    db = _get_metals_db()
    if element not in db:
        return None
    row = db[element].get("row", "")
    return _VALENCE_S_SHELL.get(row)

# ──────────────────────────────────────────────────────────────────
# AVAS-based CAS construction (renamed from build_avas_active_space)
# ──────────────────────────────────────────────────────────────────

def _construct_avas(
    mol,
    mf,
    cluster_info: ClusterInfo,
    config: AVASConfig,
) -> tuple[CAS, list[dict]]:
    """AVAS-based CAS construction.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object.
        cluster_info: ClusterInfo describing the cluster.
        config: AVASConfig with AVAS parameters.

    Returns:
        Tuple of (CAS, expected_types).
    """
    # Extract mo_coeff from mf
    mo_coeff = mf.mo_coeff
    if isinstance(mo_coeff, (list, tuple)):
        # Unrestricted: use alpha coefficients
        mo_coeff = mo_coeff[0]

    # 1. Determine target valence orbitals.
    if config.avas_valence_orbitals:
        valence_orbitals = config.avas_valence_orbitals
    else:
        valence_orbitals = _build_avas_valence_from_knowledge_base(cluster_info)

    # 2. Run AVAS selection.
    selected_indices, projection_weights = avas_select(
        mol,
        mo_coeff,
        valence_orbitals,
        threshold=config.avas_threshold,
    )

    n_orbitals = len(selected_indices)

    # 3. Estimate electron count from MO occupation analysis.
    n_electrons = _estimate_active_electrons(
        mol, mo_coeff, selected_indices, cluster_info
    )

    # 4. Build orbital groups from the valence_orbitals specification.
    orbital_groups = _build_orbital_groups(valence_orbitals, cluster_info)

    # 5. Build expected_types metadata.
    expected_types = [
        {"element": elem, "ao_types": list(ao_list)}
        for elem, ao_list in valence_orbitals.items()
    ]

    # 6. Construct CAS.
    description = (
        f"({n_electrons}e, {n_orbitals}o) AVAS-selected "
        f"[threshold={config.avas_threshold}]"
    )

    active_space = CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=ActiveSpaceLevel.STANDARD,
        description=description,
        stage="computed",
        cpt_cas_type="avas",
        mo_coeff_full=None,
        occupations_full=None,
        orbital_labels_full=[],
    )

    return active_space, expected_types

def _build_avas_valence_from_knowledge_base(cluster_info: ClusterInfo) -> dict[str, list[str]]:
    """Automatically infer AVAS target valence orbitals from the knowledge base.

    For each metal the active d orbitals are taken from the database, and the
    valence s orbital is added.  For each bridging atom the bridging p
    orbitals are taken from the ligand database.

    Args:
        cluster_info: A ``ClusterInfo`` object describing the cluster.

    Returns:
        dict mapping element symbol to list of AO type labels, e.g.
        ``{"Fe": ["3d", "4s"], "S": ["3p"], "Mo": ["4d"]}``.
    """
    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()
    result: dict[str, list[str]] = {}

    # Metals: use active_orbitals from the knowledge base + valence s.
    for metal in cluster_info.metals:
        elem = metal.element
        if elem in result:
            continue  # already processed this element type
        orbitals = []
        if elem in metals_db:
            active = metals_db[elem].get("active_orbitals", [])
            orbitals.extend(active)
            valence_s = _valence_s_for_element(elem)
            if valence_s and valence_s not in orbitals:
                orbitals.append(valence_s)
        if not orbitals:
            # Fallback: try to infer from the row field
            row = metals_db.get(elem, {}).get("row", "")
            if row:
                orbitals.append(row)  # e.g. "3d"
        if orbitals:
            result[elem] = orbitals

    # Bridging atoms: use bridging_orbitals from the ligand database.
    for bridge in cluster_info.bridging_atoms:
        elem = bridge.element
        if elem in result:
            continue
        if elem in ligands_db:
            bridging = ligands_db[elem].get("bridging_orbitals",
                                             ligands_db[elem].get("active_orbitals", []))
            if bridging:
                result[elem] = list(bridging)

    return result

# ──────────────────────────────────────────────────────────────────
# AVAS selection function
# ──────────────────────────────────────────────────────────────────
def avas_select(
    mol,
    mo_coeff: np.ndarray,
    valence_orbitals: dict[str, list[str]],
    threshold: float = 0.4,
) -> tuple[list[int], np.ndarray]:
    """Select active orbitals via AVAS projection.

    Projects MO coefficients onto a target atomic-orbital subspace defined by
    ``valence_orbitals`` and keeps MOs whose projection weight exceeds
    *threshold*.

    Args:
        mol: PySCF ``gto.Mole`` object.
        mo_coeff: MO coefficient matrix from SCF, shape ``(nao, nmo)``.
        valence_orbitals: Mapping of element symbol to list of AO type labels,
            e.g. ``{"Fe": ["3d", "4s"], "S": ["3p"], "Mo": ["4d"]}``.
        threshold: Minimum projection weight to include an orbital (default 0.4).

    Returns:
        selected_indices: List of MO indices selected for the active space.
        projection_weights: numpy array of projection weights for all MOs.
    """
    # Build AO label list from valence_orbitals dict.
    # Each entry becomes a string like "Fe 3d", "S 3p", etc.
    ao_labels = []
    for element, ao_types in valence_orbitals.items():
        for ao_type in ao_types:
            ao_labels.append(f"{element} {ao_type}")

    # Attempt to use PySCF's built-in AVAS implementation.
    try:
        from pyscf.mcscf import avas as _pyscf_avas

        # PySCF avas.kernel returns (ncore, ncas, nelecas, mo_coeff_new)
        # We only need the orbital information.
        ncore, ncas, nelecas, mo_coeff_avas = _pyscf_avas.kernel(
            mol,
            mo_coeff,
            ao_labels=ao_labels,
            threshold=threshold,
        )
        # In PySCF's AVAS the active orbitals are mo_coeff_avas[:, ncore:ncore+ncas].
        # We map back to indices in the original mo_coeff.
        # For a simpler interface we compute projection weights ourselves so
        # the return value is always consistent.
        selected, weights = _manual_avas_projection(
            mol, mo_coeff, ao_labels, threshold
        )
        return selected, weights

    except (ImportError, AttributeError, Exception) as exc:
        warnings.warn(
            f"PySCF AVAS not available or failed ({exc}); "
            f"using manual projection fallback.",
            stacklevel=2,
        )
        selected, weights = _manual_avas_projection(
            mol, mo_coeff, ao_labels, threshold
        )
        return selected, weights


def _manual_avas_projection(
    mol,
    mo_coeff: np.ndarray,
    ao_labels: list[str],
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    """Manual AVAS projection when PySCF's avas module is unavailable.

    Algorithm:
      1. Build the projection operator P (diagonal mask) from target AOs.
      2. Compute overlap matrix S.
      3. For each MO i, compute weight:
            w_i = c_i^T P S P c_i  /  (c_i^T S c_i)
      4. Select MOs with w_i > threshold.
    """
    nao, nmo = mo_coeff.shape

    # 1. Build projection mask: 1 for target AOs, 0 otherwise.
    target_indices = set()
    for label in ao_labels:
        # mol.search_ao_label returns AO indices matching the label pattern.
        indices = mol.search_ao_label(label)
        target_indices.update(indices.tolist() if hasattr(indices, "tolist") else indices)

    P = np.zeros(nao)
    for idx in target_indices:
        if 0 <= idx < nao:
            P[idx] = 1.0

    # 2. Overlap matrix
    S = mol.intor("int1e_ovlp")

    # 3. Compute projection weights
    weights = np.zeros(nmo)
    for i in range(nmo):
        c_i = mo_coeff[:, i]
        denom = c_i @ S @ c_i
        if abs(denom) < 1e-14:
            weights[i] = 0.0
            continue
        PS = P * S  # broadcast: (nao,) * (nao, nao) -> only target rows kept
        PSP = PS[:, :] * P[np.newaxis, :]  # mask columns too
        weights[i] = (c_i @ PSP @ c_i) / denom

    # 4. Select MOs
    selected = [int(i) for i in range(nmo) if weights[i] > threshold]

    return selected, weights

def _estimate_active_electrons(
    mol,
    mo_coeff: np.ndarray,
    selected_indices: list[int],
    cluster_info: ClusterInfo,
) -> int:
    """Estimate the number of active electrons in the selected orbitals.

    Uses occupation analysis: for a restricted calculation each spatial
    orbital carries 0 or 2 electrons.  For an initial estimate we assume
    all occupied MOs that fall into the active window contribute 2 electrons.
    """
    # Determine the number of occupied orbitals from the SCF density.
    # We use the cluster charge + nuclear charges to find the electron count,
    # then divide by 2 for restricted orbitals.
    nao, nmo = mo_coeff.shape
    n_electrons_total = int(mol.nelectron)
    n_electrons_charge = cluster_info.total_charge
    n_electrons_scf = n_electrons_total - n_electrons_charge
    n_occ = n_electrons_scf // 2  # restricted: doubly occupied

    # Count how many selected orbitals are occupied.
    n_active_electrons = 0
    for idx in selected_indices:
        if idx < n_occ:
            n_active_electrons += 2  # doubly occupied in restricted SCF
        # Virtual orbitals contribute 0 electrons.

    return n_active_electrons


def _build_orbital_groups(
    valence_orbitals: dict[str, list[str]],
    cluster_info: ClusterInfo,
) -> list[OrbitalGroup]:
    """Build OrbitalGroup list from the valence_orbitals specification.

    Each unique element in valence_orbitals maps to one OrbitalGroup per
    site of that element in the cluster.
    """
    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()
    groups: list[OrbitalGroup] = []

    # Count orbitals per AO type (s->1, p->3, d->5).
    _ao_capacity = {"s": 1, "p": 3, "d": 5, "f": 7}

    def _n_orbs_for_type(ao_type: str) -> int:
        # ao_type is like "3d", "4s", "3p".
        # Extract the angular momentum character (last character).
        char = ao_type[-1].lower() if ao_type else "d"
        return _ao_capacity.get(char, 5)

    # Metal groups.
    for metal in cluster_info.metals:
        elem = metal.element
        if elem not in valence_orbitals:
            continue
        ao_types = valence_orbitals[elem]
        n_orb = sum(_n_orbs_for_type(ao) for ao in ao_types)
        # Estimate electrons from the knowledge base.
        if elem in metals_db:
            # Use the most common oxidation state's d count as a rough estimate.
            ox_states = metals_db[elem].get("common_oxidation_states", [2])
            default_ox = ox_states[0] if ox_states else 2
            key = f"{elem}{abs(default_ox)}+" if default_ox > 0 else f"{elem}0"
            hs = metals_db[elem].get("high_spin_states", {})
            n_elec = hs.get(key, {}).get("d_count", 0)
            # Add s electrons if "4s"/"5s"/"6s" is in the target.
            for ao in ao_types:
                if ao.endswith("s"):
                    n_elec += 1  # rough: one s electron
        else:
            n_elec = 0

        groups.append(OrbitalGroup(
            atom_label=metal.label or f"{elem}{metal.index}",
            orbital_type="+".join(ao_types),
            n_orbitals=n_orb,
            n_electrons=n_elec,
        ))

    # Bridging atom groups.
    for bridge in cluster_info.bridging_atoms:
        elem = bridge.element
        if elem not in valence_orbitals:
            continue
        ao_types = valence_orbitals[elem]
        n_orb = sum(_n_orbs_for_type(ao) for ao in ao_types)

        # Estimate electrons from the ligand database.
        if elem in ligands_db:
            # Common estimate: 6 electrons for a filled p shell (S2-, O2-, etc.)
            n_elec = ligands_db[elem].get(
                f"electrons_as_{elem}2minus",
                ligands_db[elem].get("electrons_as_donor", 6),
            )
        else:
            n_elec = 6  # conservative default

        groups.append(OrbitalGroup(
            atom_label=f"{elem}{bridge.index}",
            orbital_type="+".join(ao_types),
            n_orbitals=n_orb,
            n_electrons=n_elec,
        ))

    return groups
