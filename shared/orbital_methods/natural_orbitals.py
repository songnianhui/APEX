"""Natural-orbital primitives."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as _gen_eigh


def natural_orbitals_from_dm(dm_mo: np.ndarray):
    """Diagonalize a 1-RDM block to obtain occupations and orbitals."""
    dm_sym = 0.5 * (dm_mo + dm_mo.T.conj())
    eigvals, eigvecs = np.linalg.eigh(dm_sym)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx].real, eigvecs[:, idx].real


def compute_unos(mol, mf):
    """Compute unrestricted natural orbitals from a UHF/UKS density matrix."""
    dm = mf.make_rdm1()
    if isinstance(dm, (list, tuple)) and len(dm) == 2:
        dm_total = dm[0] + dm[1]
    elif isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm_total = dm[0] + dm[1]
    else:
        dm_total = np.asarray(dm)

    S = mol.intor_symmetric("int1e_ovlp")
    SD = np.dot(S, dm_total)
    SDS = np.dot(SD, S)
    eigvals, eigvecs = _gen_eigh(SDS, S)
    idx = np.argsort(-eigvals)
    occupations = eigvals[idx]
    mo_coeff_uno = eigvecs[:, idx]
    return mo_coeff_uno, occupations
