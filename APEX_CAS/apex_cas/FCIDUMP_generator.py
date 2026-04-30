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

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def transform_active_integrals(
    mol,
    mf,
    mo_active: np.ndarray,
    n_electrons: int,
    target_spin: float = 0.0,
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
        Target spin *S* (e.g. 1.5 for S = 3/2).  Used to set MS2.

    Returns
    -------
    dict
        ``{"h1e", "eri", "ecore", "n_active", "n_electrons", "ms2", "ncore"}``
    """
    from pyscf import ao2mo

    n_active = mo_active.shape[1]
    ms2 = int(round(2 * target_spin))

    logger.info("Transforming integrals: CAS(%de, %do), MS2=%d", n_electrons, n_active, ms2)

    # ── 1-electron integrals ──────────────────────────────────────
    hcore = mf.get_hcore()
    h1e = mo_active.T @ hcore @ mo_active
    h1e = (h1e + h1e.T) / 2  # enforce exact symmetry

    # ── 2-electron integrals ──────────────────────────────────────
    eri = ao2mo.kernel(mol, mo_active, verbose=0)

    # ── Core energy ───────────────────────────────────────────────
    ecore, ncore = _compute_core_energy(mol, mf, hcore)

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


def write_fcidump(
    integrals: dict,
    output_path: str,
    orbsym: list[int] = None,
    zero_ecore: bool = True,
) -> str:
    """Write integrals to a standard FCIDUMP file.

    Parameters
    ----------
    integrals : dict
        Output of :func:`transform_active_integrals`.
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
    from pyscf.tools import fcidump as fcidump_mod

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


def compare_fcidumps(
    path_ref: str,
    path_new: str,
    h1e_tol: float = 1e-4,
    ecore_tol: float = 1e-6,
) -> dict:
    """Compare two FCIDUMP files.

    Parameters
    ----------
    path_ref, path_new : str
    h1e_tol : float
        Tolerance for h1e Frobenius norm difference.
    ecore_tol : float
        Tolerance for ecore absolute difference.

    Returns
    -------
    dict
        ``{"h1e_frobenius", "h1e_max", "h2e_rms", "h2e_max",
          "ecore_ref", "ecore_new", "ecore_diff", "match"}``
    """
    from pyscf.tools import fcidump as fcidump_mod

    ref = fcidump_mod.read(path_ref, verbose=False)
    new = fcidump_mod.read(path_new, verbose=False)

    h1_ref, h1_new = ref["H1"], new["H1"]
    dh1 = h1_new - h1_ref
    h1_frob = float(np.linalg.norm(dh1))
    h1_max = float(np.max(np.abs(dh1)))

    h2_rms = 0.0
    h2_max = 0.0
    h2_ref, h2_new = ref["H2"], new["H2"]
    if h2_ref.shape == h2_new.shape:
        dh2 = h2_new - h2_ref
        h2_rms = float(np.sqrt(np.mean(dh2 ** 2)))
        h2_max = float(np.max(np.abs(dh2)))

    ecore_ref = float(ref["ECORE"])
    ecore_new = float(new["ECORE"])
    ecore_diff = ecore_new - ecore_ref

    result = {
        "h1e_frobenius": h1_frob,
        "h1e_max": h1_max,
        "h2e_rms": h2_rms,
        "h2e_max": h2_max,
        "ecore_ref": ecore_ref,
        "ecore_new": ecore_new,
        "ecore_diff": ecore_diff,
        "match": h1_frob < h1e_tol and abs(ecore_diff) < ecore_tol,
    }

    logger.info("FCIDUMP comparison:")
    logger.info("  h1e: Frobenius=%.6e, max=%.6e", h1_frob, h1_max)
    logger.info("  h2e: RMS=%.6e, max=%.6e", h2_rms, h2_max)
    logger.info("  ecore: ref=%.12f, new=%.12f, diff=%+.6e", ecore_ref, ecore_new, ecore_diff)
    logger.info("  match: %s", result["match"])

    return result


def generate_fcidump_from_selection(
    mol,
    mf,
    mo_coeff_loc: np.ndarray,
    occupations: np.ndarray,
    selected_indices: list[int],
    output_path: str,
    target_spin: float = 0.0,
    zero_ecore: bool = True,
) -> str:
    """One-stop: build active space from selected indices and write FCIDUMP.

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
    zero_ecore : bool
        If True (default), write ECORE=0 and append E_core as a comment line.

    Returns
    -------
    str
        Absolute path of the FCIDUMP file.
    """
    selected_indices = sorted(selected_indices)
    mo_active = mo_coeff_loc[:, selected_indices]
    occ_active = occupations[selected_indices]
    n_electrons = int(round(float(np.sum(occ_active))))
    n_active = len(selected_indices)

    print(f"  Active space: CAS({n_electrons}e, {n_active}o)")
    print(f"  MS2 = {int(round(2 * target_spin))}")

    integrals = transform_active_integrals(
        mol, mf, mo_active, n_electrons, target_spin=target_spin,
    )
    return write_fcidump(integrals, output_path, zero_ecore=zero_ecore)


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _compute_core_energy(mol, mf, hcore) -> tuple[float, int]:
    """Compute core energy = E_nuc + E_core_1e + E_core_2e.

    Uses the alpha MO channel for the spin-free (restricted) representation.

    Returns
    -------
    (ecore, ncore) : tuple[float, int]
    """
    mo_alpha = mf.mo_coeff[0]
    occ_alpha = mf.mo_occ[0]
    ncore = int(np.sum(occ_alpha > 0))

    # Core density matrix (doubly occupied alpha + beta → factor of 2)
    mo_core = mo_alpha[:, :ncore]
    core_dm = mo_core @ mo_core.T * 2

    e_nuc = mol.energy_nuc()
    e_1e = float(np.einsum("ij,ji->", hcore, core_dm))
    vj, vk = mf.get_jk(mol, core_dm)
    e_2e = 0.5 * float(np.einsum("ij,ji->", vj * 2 - vk, core_dm))
    ecore = e_nuc + e_1e + e_2e

    return ecore, ncore
