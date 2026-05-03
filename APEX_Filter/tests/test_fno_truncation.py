"""Regression tests for FNO-style truncation helpers."""

import os

import numpy as np

from apex_filter.fno_truncation import build_fno_subspace_from_uccsd
from shared.active_space_reference import build_fake_mol, build_reference_uhf_solver
from shared.fcidump_io import FCIDUMPData


def _make_toy_fcidump():
    return FCIDUMPData(
        h1e=np.diag([-1.2, -0.8, -0.3, 0.1]),
        h2e=np.zeros((4, 4, 4, 4)),
        ecore=-5.0,
        norb=4,
        nelec=4,
        ms2=0,
    )


def _write_toy_uhf_npz(tmp_path, fcidump_data):
    mol = build_fake_mol(fcidump_data.norb, fcidump_data.nelec, fcidump_data.ms2, ecore=fcidump_data.ecore)
    mf = build_reference_uhf_solver(fcidump_data, mol, conv_tol=1e-10, max_cycle=50)
    mf.kernel()
    path = os.path.join(tmp_path, "toy_uhf.npz")
    np.savez(
        path,
        energy=mf.e_tot,
        converged=mf.converged,
        mo_coeff_a=mf.mo_coeff[0],
        mo_coeff_b=mf.mo_coeff[1],
        mo_occ_a=mf.mo_occ[0],
        mo_occ_b=mf.mo_occ[1],
        mo_energy_a=mf.mo_energy[0],
        mo_energy_b=mf.mo_energy[1],
    )
    return path


def test_build_fno_subspace_from_uccsd_freezes_spin_resolved_occupied_nos(tmp_path):
    fcid = _make_toy_fcidump()
    uhf_npz = _write_toy_uhf_npz(tmp_path, fcid)

    result = build_fno_subspace_from_uccsd(fcid, uhf_npz, freeze_occ=1, conv_tol=1e-10, max_cycle=50)

    assert result.converged is True
    assert result.frozen == ([0], [0])
    assert result.frozen_occ_alpha == 1
    assert result.frozen_occ_beta == 1
    assert result.kept_occ_alpha == 1
    assert result.kept_occ_beta == 1
    assert result.occupied_noons_alpha.shape == (2,)
    assert result.occupied_noons_beta.shape == (2,)
