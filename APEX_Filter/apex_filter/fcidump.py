"""Module: FCIDUMP Generation

Generate FCIDUMP integral files from pipeline results for use with
downstream QC codes (block2, CheMPS2, Dice, QCMaquis, etc.).

Supports both full-space and active-space (CAS) FCIDUMP via PySCF's
``pyscf.tools.fcidump`` module.
"""

import os

import numpy as np

from .models import ClusterInfo, CAS, FilteringPlan


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def generate_fcidump(cluster_info: ClusterInfo,
                     active_space: CAS,
                     uhf_npz: str,
                     output_dir: str,
                     dmrg_npz: str = None,
                     basis_set: str = "cc-pVDZ",
                     mode: str = "both",
                     label: str = None) -> dict:
    """Generate FCIDUMP file(s) for a single configuration.

    Args:
        cluster_info: Molecule definition (geometry, charge, spin).
        active_space: Active space specification.
        uhf_npz: Path to the UHF *_uhf.npz result file.
        output_dir: Directory to write FCIDUMP files.
        dmrg_npz: Optional path to DMRG *_dmrg_results.npz (for ncore/cas info).
        basis_set: Basis set name (must match calculation).
        mode: "full", "active", or "both".
        label: Config label for filenames. Derived from UHF NPZ if None.

    Returns:
        Dict with paths to generated files.
    """
    from pyscf import gto, scf, mcscf, ao2mo
    from pyscf.tools import fcidump as fcidump_mod

    os.makedirs(output_dir, exist_ok=True)

    if label is None:
        label = os.path.basename(uhf_npz).replace("_uhf.npz", "")

    # Reconstruct UHF object
    mf = _reconstruct_uhf(cluster_info, basis_set, uhf_npz)

    result = {"label": label}

    # --- Full-space FCIDUMP ---
    if mode in ("full", "both"):
        full_path = os.path.join(output_dir, f"{label}_full.FCIDUMP")
        _write_full_fcidump(mf, full_path)
        result["full_space"] = full_path
        print(f"  Full-space FCIDUMP: {full_path}")

    # --- Active-space FCIDUMP ---
    if mode in ("active", "both"):
        ncore, ncas, nelecas = _get_cas_params(mf, active_space, dmrg_npz)
        cas_path = os.path.join(output_dir,
                                f"{label}_cas{nelecas}_{ncas}.FCIDUMP")
        _write_active_fcidump(mf, ncore, ncas, nelecas, cas_path)
        result["active_space"] = cas_path
        result["ncas"] = ncas
        result["nelecas"] = nelecas
        result["ncore"] = ncore
        print(f"  Active-space FCIDUMP: {cas_path} (ncore={ncore}, CAS({nelecas},{ncas}))")

    return result


def generate_fcidump_for_results(results: list,
                                  cluster_info: ClusterInfo,
                                  active_space: CAS,
                                  workdir: str,
                                  plan: FilteringPlan,
                                  basis_set: str = "cc-pVDZ",
                                  output_dir: str = None,
                                  mode: str = "both") -> list[dict]:
    """Generate FCIDUMP files for all final pipeline results.

    Locates UHF and DMRG NPZ files from the pipeline directory structure
    and generates FCIDUMP for each result.

    Args:
        results: List of CalculationResult from pipeline final selection.
        cluster_info: Molecule definition.
        active_space: Active space specification.
        workdir: Pipeline output directory (e.g., "examples/vcl4/pipeline_output").
        plan: FilteringPlan with level definitions.
        basis_set: Basis set name.
        output_dir: FCIDUMP output directory. Defaults to {parent_of_workdir}/fcidump/.
        mode: "full", "active", or "both".

    Returns:
        List of dicts with generated file info.
    """
    if not results:
        return []

    if output_dir is None:
        # workdir = .../pipeline_output → fcidump dir = .../fcidump/
        output_dir = os.path.join(os.path.dirname(workdir), "fcidump")

    # Find UHF and DMRG level directories
    uhf_level_idx = next(
        (i for i, lv in enumerate(plan.levels) if lv.method == "UHF"), None)
    dmrg_level_indices = [
        i for i, lv in enumerate(plan.levels) if lv.method == "DMRG"]

    # Use the last DMRG level for active-space parameters
    last_dmrg_idx = dmrg_level_indices[-1] if dmrg_level_indices else None

    all_info = []
    for r in results:
        if r.config is None:
            continue
        label = _sanitize_label(r.config.label)

        # Locate UHF NPZ
        uhf_npz = _locate_npz(workdir, uhf_level_idx, "UHF", label, "_uhf.npz")
        if uhf_npz is None:
            print(f"  [SKIP] {label}: UHF NPZ not found")
            continue

        # Locate DMRG NPZ (optional)
        dmrg_npz = None
        if last_dmrg_idx is not None:
            dmrg_npz = _locate_npz(workdir, last_dmrg_idx, "DMRG", label,
                                    "_dmrg_results.npz")

        try:
            info = generate_fcidump(
                cluster_info, active_space, uhf_npz, output_dir,
                dmrg_npz=dmrg_npz, basis_set=basis_set,
                mode=mode, label=label,
            )
            all_info.append(info)
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")

    return all_info


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _reconstruct_uhf(cluster_info, basis_set, npz_path):
    """Reconstruct a PySCF UHF object from saved NPZ data."""
    from pyscf import gto, scf

    data = np.load(npz_path, allow_pickle=True)

    # Build molecule
    geometry_lines = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        geometry_lines.append(f"{elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    geometry = "\n".join(geometry_lines)

    spin_multiplicity = int(round(2 * cluster_info.target_spin))

    mol = gto.M(
        atom=geometry,
        basis=basis_set,
        charge=cluster_info.total_charge,
        spin=spin_multiplicity,
        verbose=0,
        symmetry=False,
    )

    mf = scf.UHF(mol)
    mf.mo_coeff = (data["mo_coeff_a"], data["mo_coeff_b"])
    mf.mo_occ = (data["mo_occ_a"], data["mo_occ_b"])
    mf.mo_energy = (data["mo_energy_a"], data["mo_energy_b"])
    mf.e_tot = float(data["energy"])
    mf.converged = bool(data["converged"])

    return mf


def _write_full_fcidump(mf, output_path):
    """Write full-space integrals in FCIDUMP format.

    For UHF, uses the alpha MO coefficients for spin-free integrals.
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump as fcidump_mod

    # Use alpha MOs for spin-free representation
    mo = mf.mo_coeff[0]
    nmo = mo.shape[1]
    nocc = int(np.sum(mf.mo_occ[0] > 0))
    nelec = nocc + int(np.sum(mf.mo_occ[1] > 0))

    # 1-electron integrals in MO basis
    h1e = mf.get_hcore()
    h1e_mo = mo.T @ h1e @ mo

    # 2-electron integrals in MO basis (4-fold symmetry)
    eri_mo = ao2mo.full(mf.mol, mo, verbose=0)

    # Core energy from nuclear repulsion
    ecore = mf.energy_nuc()

    fcidump_mod.from_integrals(output_path, h1e_mo, eri_mo, nmo, nelec,
                               nuc=ecore)


def _write_active_fcidump(mf, ncore, ncas, nelecas, output_path):
    """Write active-space integrals in FCIDUMP format.

    Uses CASCI to extract 1e/2e integrals in the active space
    defined by ncore frozen orbitals and ncas active orbitals.
    For UHF, uses alpha MOs for the spin-free FCIDUMP representation.
    """
    from pyscf import mcscf, ao2mo
    from pyscf.tools import fcidump as fcidump_mod

    # For UHF: use alpha channel for spin-free integrals
    mo = mf.mo_coeff[0]
    nocc = int(np.sum(mf.mo_occ[0] > 0))
    nmo_total = mo.shape[1]

    # Active space orbital slice
    act_slice = slice(ncore, ncore + ncas)
    mo_act = mo[:, act_slice]

    # 1-electron integrals in active MO basis
    h1e_full = mf.get_hcore()
    h1e_act = mo_act.T @ h1e_full @ mo_act

    # 2-electron integrals in active MO basis
    eri_act = ao2mo.full(mf.mol, mo_act, verbose=0)

    # Core energy: nuclear repulsion + frozen core contribution
    # Use CASCI to get proper ecore
    cas = mcscf.CASCI(mf, ncas, nelecas)
    cas.ncore = ncore
    _, ecore = cas.get_h1eff()

    fcidump_mod.from_integrals(output_path, h1e_act, eri_act, ncas, nelecas,
                               nuc=ecore)


def _get_cas_params(mf, active_space, dmrg_npz=None):
    """Determine ncore, ncas, nelecas for the active space.

    Prefers explicit values from DMRG NPZ. Falls back to active_space
    definition + heuristic ncore.
    """
    if dmrg_npz is not None and os.path.exists(dmrg_npz):
        data = np.load(dmrg_npz, allow_pickle=True)
        if "ncore" in data.files and "cas_n_orb" in data.files:
            ncore = int(data["ncore"])
            ncas = int(data["cas_n_orb"])
            nelecas = int(data["cas_n_elec"])
            return ncore, ncas, nelecas

    # Fallback: use active_space parameters
    ncas = active_space.n_orbitals
    nelecas = active_space.n_electrons

    # Heuristic ncore: freeze all doubly-occupied orbitals below the active space
    nocc_a = int(np.sum(mf.mo_occ[0] > 0))
    nocc_b = int(np.sum(mf.mo_occ[1] > 0))
    n_total_elec = nocc_a + nocc_b
    ncore = max(0, (n_total_elec - nelecas) // 2)

    return ncore, ncas, nelecas


def _locate_npz(workdir, level_idx, method, label, suffix):
    """Find an NPZ file for a config in the level directory.

    Args:
        workdir: Pipeline output root (e.g., .../pipeline_output).
        level_idx: Level index (e.g., 0 for UHF).
        method: Method name (e.g., "UHF", "DMRG").
        label: Sanitized config label.
        suffix: NPZ suffix (e.g., "_uhf.npz", "_dmrg_results.npz").

    Returns:
        Absolute path to NPZ file, or None if not found.
    """
    if level_idx is None:
        return None

    level_dir = os.path.join(workdir, f"level_{level_idx}_{method}")
    npz_path = os.path.join(level_dir, f"{label}{suffix}")

    if os.path.exists(npz_path):
        return npz_path

    # Try globbing in case label has minor variations
    import glob
    candidates = glob.glob(os.path.join(level_dir, f"*{suffix}"))
    if candidates:
        return candidates[0]

    return None


def _sanitize_label(label):
    """Sanitize a config label for filesystem use."""
    return label.replace("|", "_").replace(" ", "_")
