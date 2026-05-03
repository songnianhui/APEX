"""AO -> MO integral transformation and FCIDUMP generation.

Transforms one- and two-electron integrals into an active-space MO basis
under the sf-X2C (spin-free exact two-component) Hamiltonian, then writes
a standard FCIDUMP file for downstream codes (Block2, CheMPS2, Dice, etc.).

Key design points:
  - ``mf.get_hcore()`` already contains the sf-X2C correction when the SCF
    object has been wrapped with ``sfx2c1e()``.
  - Two-electron integrals are non-relativistic (sf-X2C only modifies the
    1-electron part).
  - Core energy (``ecore``) includes nuclear repulsion + frozen-core
    (Hartree–Fock) contribution.
"""

import json
import logging
import os

import numpy as np
from pyscf import ao2mo
from pyscf.mcscf import casci as casci_mod
from pyscf.tools import fcidump as fcidump_mod
from shared import comparison as comparison_module

logger = logging.getLogger(__name__)

def compare_fcidumps(*args, **kwargs):
    """Benchmark-facing thin wrapper over the shared FCIDUMP comparator."""
    return comparison_module.compare_fcidumps(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────
# Public comparison utilities and internal workflow primitives
# ──────────────────────────────────────────────────────────────────

def _transform_active_integrals(
    mol,
    mf,
    mo_active: np.ndarray,
    n_electrons: int,
    target_spin: float = 0.0,
    core_dm: np.ndarray = None,
) -> dict:
    """Transform AO integrals to an active-space MO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf : pyscf SCF object
        Must support ``mf.get_hcore()``, ``mf.mo_coeff``, ``mf.mo_occ``,
        ``mf.get_jk()``.  If wrapped with ``sfx2c1e``, ``get_hcore()``
        returns the sf-X2C 1-electron Hamiltonian.
    mo_active : ndarray (nao, n_active)
        Active-space MO coefficient matrix.
    n_electrons : int
        Number of active electrons.
    target_spin : float
        Spin projection Sz (e.g. 0.0 for Sz=0, 1.5 for Sz=3/2).
        Converted to MS2 = 2*Sz for FCIDUMP header and CASCI (nalpha, nbeta).
    core_dm : ndarray (nao, nao) or None
        Frozen-core density matrix in AO basis. When the active orbitals
        are in a different basis than SCF (e.g. UNO / localized), this must
        be provided to ensure the frozen core is orthogonal to the active
        space. If None, the SCF doubly-occupied density is used (correct
        only when active orbitals are a subset of SCF MOs).

    Returns
    -------
    dict
        ``{"h1e", "eri", "ecore", "n_active", "n_electrons", "ms2", "ncore"}``
    """
    n_active = mo_active.shape[1]
    ms2 = int(round(2 * target_spin))

    logger.info("Transforming integrals: CAS(%de, %do), MS2=%d", n_electrons, n_active, ms2)

    # ── 1-electron integrals ──────────────────────────────────────
    hcore = mf.get_hcore()
    ecore, ncore, V_core = _compute_core_energy(mol, mf, hcore, core_dm=core_dm)
    h1e = mo_active.T @ (hcore + V_core) @ mo_active
    h1e = (h1e + h1e.T) / 2  # enforce exact symmetry

    # ── 2-electron integrals ──────────────────────────────────────
    eri = ao2mo.kernel(mol, mo_active, verbose=0)
    eri = ao2mo.restore(8, eri, n_active)

    logger.info("  ncore=%d, ecore=%.12f", ncore, ecore)
    logger.info("  h1e shape=%s, eri shape=%s", h1e.shape, eri.shape)

    return {
        "h1e": h1e,
        "eri": eri,
        "ecore": ecore,
        "n_active": n_active,
        "n_electrons": n_electrons,
        "ms2": ms2,
        "ncore": ncore,
    }


def _write_fcidump(
    integrals: dict,
    output_path: str,
    orbsym: list[int] = None,
    zero_ecore: bool = True,
) -> str:
    """Write integrals to a standard FCIDUMP file.

    Parameters
    ----------
    integrals : dict
        Output of :func:`_transform_active_integrals`.
    output_path : str
    orbsym : list[int], optional
        Orbital symmetry labels (Molpro convention).  Defaults to all-1 (C1).
    zero_ecore : bool
        If True (default), write ECORE=0 in the FCIDUMP header (DMRG-community
        convention) and append the real core energy as a comment line.  If
        False, write the actual ECORE value (PySCF convention).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    n_active = integrals["n_active"]
    if orbsym is None:
        orbsym = [1] * n_active  # C1 symmetry

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ecore_val = 0.0 if zero_ecore else integrals["ecore"]

    fcidump_mod.from_integrals(
        output_path,
        integrals["h1e"],
        integrals["eri"],
        n_active,
        integrals["n_electrons"],
        nuc=ecore_val,
        ms=integrals["ms2"],
        orbsym=orbsym,
    )

    # When zero_ecore, save the real E_core to a sidecar file so that
    # downstream tools can reconstruct the total energy if needed.
    if zero_ecore and abs(integrals["ecore"]) > 1e-15:
        ecore_path = output_path + ".ecore"
        with open(ecore_path, "w") as f:
            f.write(f"{integrals['ecore']:.15f}\n")

    logger.info("FCIDUMP written to %s: CAS(%de, %do), MS2=%d, ecore=%.12f (zero_ecore=%s)",
                output_path, integrals["n_electrons"], n_active,
                integrals["ms2"], ecore_val, zero_ecore)
    return os.path.abspath(output_path)


def _generate_fcidump_from_selection(
    mol,
    mf,
    mo_coeff_loc: np.ndarray,
    occupations: np.ndarray,
    selected_indices: list[int],
    output_path: str,
    target_spin: float = 0.0,
    zero_ecore: bool = True,
    e_solvent: float = 0.0,
    occ_core_hi: float = 1.98,
    frozen_core_indices: list[int] | None = None,
    n_electrons: int | None = None,
) -> str:
    """One-stop: build active space from selected indices and write FCIDUMP.

    Uses PySCF's CASCI machinery to fold the frozen-core potential into the
    effective one-electron Hamiltonian for localized/UNO active spaces.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf : pyscf SCF object
    mo_coeff_loc : ndarray (nao, nmo)
        Full localized MO coefficient matrix.
    occupations : ndarray (nmo,)
        UNO occupation numbers.
    selected_indices : list[int]
        Sorted list of selected orbital column indices.
    output_path : str
    target_spin : float
        Spin projection Sz.
    zero_ecore : bool
        If True (default), write ECORE=0 and append E_core as a comment line.
    e_solvent : float
        Optional solvent correction used only for the sidecar.
    occ_core_hi : float
        Occupation threshold used to classify frozen core orbitals.
    frozen_core_indices : list[int] | None
        Explicit frozen-core indices; if None, determined from occupations.
    n_electrons : int | None
        Explicit active electron count; if None, derived from occupations.

    Returns
    -------
    str
        Absolute path of the FCIDUMP file.
    """
    selected_indices = sorted(selected_indices)
    mo_active = mo_coeff_loc[:, selected_indices]
    occ_active = occupations[selected_indices]
    if n_electrons is None:
        n_electrons = int(round(float(np.sum(occ_active))))
    n_active = len(selected_indices)
    ms2 = int(round(2 * target_spin))
    nalpha = (n_electrons + ms2) // 2
    nbeta = (n_electrons - ms2) // 2
    if nalpha + nbeta != n_electrons:
        raise ValueError(
            f"Inconsistent electron partition: nalpha={nalpha} + nbeta={nbeta} = "
            f"{nalpha + nbeta} != n_electrons={n_electrons}. "
            f"Check --spin-projection (Sz={target_spin}, MS2={ms2})."
        )

    print(f"  Active space: CAS({n_electrons}e, {n_active}o)")
    print(f"  MS2 = {ms2}")

    selected_set = set(selected_indices)
    if frozen_core_indices is not None:
        frozen_core_idx = sorted(frozen_core_indices)
    else:
        frozen_core_idx = sorted(
            [i for i in range(len(occupations)) if occupations[i] > occ_core_hi and i not in selected_set]
        )
    ncore = len(frozen_core_idx)
    mo_frozen = mo_coeff_loc[:, frozen_core_idx]
    n_frozen_elec = ncore * 2
    print(f"  Frozen core: {ncore} orbitals, {n_frozen_elec} electrons (CASCI closed-shell)")
    print(f"  Total electrons: {n_frozen_elec + n_electrons} (frozen {n_frozen_elec} + active {n_electrons})")

    mo_reordered = np.hstack([mo_frozen, mo_active])
    casci_obj = casci_mod.CASCI(mf, n_active, (nalpha, nbeta), ncore=ncore)
    casci_obj.mo_coeff = mo_reordered

    h1eff, e_core = casci_obj.get_h1eff()
    eri = casci_obj.get_h2eff()
    eri = ao2mo.restore(8, eri, n_active)

    integrals = {
        "h1e": h1eff,
        "eri": eri,
        "ecore": e_core,
        "n_active": n_active,
        "n_electrons": n_electrons,
        "ms2": ms2,
        "ncore": ncore,
    }

    fcidump_path = _write_fcidump(integrals, output_path, zero_ecore=zero_ecore)

    if hasattr(mf, "with_solvent"):
        e_solvent = float(mf.scf_summary.get("e_solvent", e_solvent))
        logger.info("Solvent energy from mf.scf_summary: %.10f Hartree", e_solvent)
    if abs(e_solvent) > 0 and hasattr(mf, "with_solvent"):
        esolv_path = fcidump_path + ".esolv"
        solv_obj = mf.with_solvent
        _write_esolv_file(
            esolv_path,
            e_solvent,
            float(mf.e_tot),
            getattr(solv_obj, "epsilon", 4.0),
        )
        logger.info("Solvent sidecar written to %s", esolv_path)

    return fcidump_path


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _write_esolv_file(esolv_path, e_solvent, e_total_scf, solvation_epsilon):
    """Write solvent energy sidecar alongside FCIDUMP."""
    data = {
        "e_solvent": e_solvent,
        "e_total_scf": e_total_scf,
        "e_gas_phase_scf": e_total_scf - e_solvent,
        "solvation_model": "ddCOSMO",
        "solvation_epsilon": solvation_epsilon,
        "note": "Add e_solvent to DMRG+ecore total energy for solvent-corrected result.",
    }
    with open(esolv_path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_core_energy(mol, mf, hcore, core_dm=None) -> tuple[float, int, np.ndarray]:
    """Compute core energy and frozen-core mean-field potential."""
    if core_dm is None:
        mo_alpha = mf.mo_coeff[0]
        occ_alpha = mf.mo_occ[0]
        occ_beta = mf.mo_occ[1]
        n_alpha_occ = int(np.sum(occ_alpha > 0))
        n_beta_occ = int(np.sum(occ_beta > 0))
        ncore = min(n_alpha_occ, n_beta_occ)
        mo_core = mo_alpha[:, :ncore]
        core_dm = mo_core @ mo_core.T * 2
    else:
        ncore = -1

    e_nuc = mol.energy_nuc()
    e_1e = float(np.einsum("ij,ji->", hcore, core_dm))
    vj, vk = mf.get_jk(mol, core_dm)
    corevhf = vj - 0.5 * vk
    e_2e = 0.5 * float(np.einsum("ij,ji->", corevhf, core_dm))
    ecore = e_nuc + e_1e + e_2e

    return ecore, ncore, corevhf
