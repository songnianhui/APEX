"""FNO-style occupied-space truncation on the active-space Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_ucc import load_reference_mf_from_npz


@dataclass
class FNOSubspaceResult:
    """Spin-resolved occupied-NO truncation derived from active-space UCCSD."""

    mo_coeff: tuple[np.ndarray, np.ndarray]
    mo_occ: tuple[np.ndarray, np.ndarray]
    mo_energy: tuple[np.ndarray, np.ndarray]
    frozen: tuple[list[int], list[int]]
    occupied_noons_alpha: np.ndarray
    occupied_noons_beta: np.ndarray
    kept_occ_alpha: int
    kept_occ_beta: int
    frozen_occ_alpha: int
    frozen_occ_beta: int
    uccsd_energy: float
    uccsd_corr: float
    converged: bool


def _sorted_occ_nos(dm_occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dm_occ = 0.5 * (dm_occ + dm_occ.conj().T)
    occs, vecs = np.linalg.eigh(dm_occ)
    order = np.argsort(occs)[::-1]
    return occs[order].real, vecs[:, order]


def build_fno_subspace_from_uccsd(
    fcidump_data,
    uhf_npz_path: str,
    *,
    freeze_occ: int,
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    diis_space: int = 12,
) -> FNOSubspaceResult:
    """Build a truncated occupied-NO reference for later FNO high-order CC.

    Implementation notes:
    - Run active-space UCCSD on the FCIDUMP Hamiltonian.
    - Form spin-resolved 1-RDMs in MO basis via PySCF ``make_rdm1()``.
    - Diagonalize the occupied-occupied block separately for alpha and beta.
    - Sort occupied NOs by descending occupation and freeze the leading
      ``freeze_occ`` orbitals in each spin channel.
    - Preserve all virtual orbitals.
    """
    from pyscf import cc

    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle
    mycc.diis_space = diis_space
    mycc.direct = False
    mycc.kernel()

    dm1a, dm1b = mycc.make_rdm1()
    nocc_a = int(np.sum(mf.mo_occ[0] > 0))
    nocc_b = int(np.sum(mf.mo_occ[1] > 0))
    max_freeze = max(0, min(nocc_a - 1, nocc_b - 1))
    eff_freeze = max(0, min(int(freeze_occ), max_freeze))

    occs_a, rot_occ_a = _sorted_occ_nos(dm1a[:nocc_a, :nocc_a])
    occs_b, rot_occ_b = _sorted_occ_nos(dm1b[:nocc_b, :nocc_b])

    coeff_occ_a = mf.mo_coeff[0][:, :nocc_a] @ rot_occ_a
    coeff_occ_b = mf.mo_coeff[1][:, :nocc_b] @ rot_occ_b
    coeff_vir_a = mf.mo_coeff[0][:, nocc_a:]
    coeff_vir_b = mf.mo_coeff[1][:, nocc_b:]

    mo_coeff = (
        np.hstack([coeff_occ_a, coeff_vir_a]),
        np.hstack([coeff_occ_b, coeff_vir_b]),
    )
    mo_occ = (np.array(mf.mo_occ[0], copy=True), np.array(mf.mo_occ[1], copy=True))
    mo_energy = (np.array(mf.mo_energy[0], copy=True), np.array(mf.mo_energy[1], copy=True))
    frozen = (list(range(eff_freeze)), list(range(eff_freeze)))

    return FNOSubspaceResult(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        frozen=frozen,
        occupied_noons_alpha=occs_a,
        occupied_noons_beta=occs_b,
        kept_occ_alpha=nocc_a - eff_freeze,
        kept_occ_beta=nocc_b - eff_freeze,
        frozen_occ_alpha=eff_freeze,
        frozen_occ_beta=eff_freeze,
        uccsd_energy=float(mycc.e_tot),
        uccsd_corr=float(mycc.e_corr),
        converged=bool(mycc.converged),
    )
