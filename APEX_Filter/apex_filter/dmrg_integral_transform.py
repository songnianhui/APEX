"""Basis construction and integral transformation helpers for step8 DMRG."""

from __future__ import annotations

import h5py
import numpy as np
from pyscf import ao2mo


def lowdin_orthonormalize(coeff: np.ndarray) -> np.ndarray:
    """Return a Lowdin-orthonormalized version of a column basis."""
    metric = 0.5 * (coeff.T @ coeff + coeff.T.conj() @ coeff.conj())
    eigvals, eigvecs = np.linalg.eigh(metric.real)
    keep = eigvals > 1e-12
    if not np.all(keep):
        raise ValueError("DMRG basis average is rank-deficient after alpha/beta pairing")
    inv_sqrt = eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.T
    return coeff @ inv_sqrt


def build_spatial_basis(
    *,
    dmrg_basis_npz_path: str | None,
    norb: int,
    basis_mode: str,
) -> np.ndarray:
    """Construct a spatial active-space basis for DMRG."""
    mode = (basis_mode or "step7_paired").strip().lower()
    if mode == "original_identity":
        return np.eye(norb)
    if dmrg_basis_npz_path is None:
        raise ValueError("dmrg_basis_npz_path is required for basis_mode='step7_paired'")

    if dmrg_basis_npz_path.endswith(".h5"):
        with h5py.File(dmrg_basis_npz_path, "r") as f:
            orbitals = f["orbitals"]
            active_a = np.asarray(orbitals["active_coeff_alpha"], dtype=float)
            active_b = np.asarray(orbitals["active_coeff_beta"], dtype=float)
    else:
        data = np.load(dmrg_basis_npz_path, allow_pickle=True)
        try:
            active_a = np.asarray(data["active_coeff_alpha"], dtype=float)
            active_b = np.asarray(data["active_coeff_beta"], dtype=float)
        finally:
            data.close()
    if active_a.shape != (norb, norb) or active_b.shape != (norb, norb):
        raise ValueError(
            "DMRG basis active coefficients have unexpected shape: "
            f"alpha={active_a.shape}, beta={active_b.shape}, expected ({norb}, {norb})"
        )
    spatial_guess = 0.5 * (active_a + active_b)
    try:
        return lowdin_orthonormalize(spatial_guess)
    except ValueError:
        return lowdin_orthonormalize(active_a)


def transform_integrals(fcidump_data, spatial_basis: np.ndarray):
    """Transform FCIDUMP integrals into the chosen spatial basis."""
    h1e = np.asarray(fcidump_data.h1e, dtype=float)
    h2e = np.asarray(fcidump_data.h2e, dtype=float)
    if h2e.ndim != 4:
        h2e = ao2mo.restore(1, h2e, fcidump_data.norb)

    u = np.asarray(spatial_basis, dtype=float)
    h1e_t = u.T @ h1e @ u
    h2e_t = np.einsum("pi,qj,rk,sl,pqrs->ijkl", u, u, u, u, h2e, optimize=True)
    return h1e_t, h2e_t
