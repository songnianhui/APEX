"""Regression tests for active-space reference UCC on FCIDUMP Hamiltonians."""

import os

import numpy as np
import pytest

from apex_filter.CAS_loader import FCIDUMPData
from apex_filter.hdf5_state_io import save_uhf_state_h5
from apex_filter.reference_uhf import build_fake_mol, build_reference_uhf_solver
from apex_filter.reference_ucc import run_reference_ucc, save_reference_ucc_result
from apex_filter.result_parser import parse_npz_result


def _make_toy_fcidump():
    return FCIDUMPData(
        h1e=np.diag([-1.0, -0.5]),
        h2e=np.zeros((2, 2, 2, 2)),
        ecore=-10.0,
        norb=2,
        nelec=2,
        ms2=0,
    )


def _write_toy_uhf_npz(tmp_path, fcidump_data):
    mol = build_fake_mol(
        fcidump_data.norb,
        fcidump_data.nelec,
        fcidump_data.ms2,
        ecore=fcidump_data.ecore,
    )
    mf = build_reference_uhf_solver(fcidump_data, mol, conv_tol=1e-10, max_cycle=50)
    mf.kernel()

    npz_path = os.path.join(tmp_path, "toy_uhf.npz")
    np.savez(
        npz_path,
        energy=mf.e_tot,
        converged=mf.converged,
        spin_sq=mf.spin_square()[0],
        mo_coeff_a=mf.mo_coeff[0],
        mo_coeff_b=mf.mo_coeff[1],
        mo_occ_a=mf.mo_occ[0],
        mo_occ_b=mf.mo_occ[1],
        mo_energy_a=mf.mo_energy[0],
        mo_energy_b=mf.mo_energy[1],
    )
    return npz_path, mf


def _write_toy_uhf_h5(tmp_path, fcidump_data):
    npz_path, mf = _write_toy_uhf_npz(tmp_path, fcidump_data)
    data = np.load(npz_path, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    h5_path = os.path.join(tmp_path, "toy_uhf.h5")
    save_uhf_state_h5(h5_path, payload)
    return h5_path, mf


def test_reference_uhf_includes_ecore_in_total_energy():
    fcid = _make_toy_fcidump()
    mol = build_fake_mol(fcid.norb, fcid.nelec, fcid.ms2, ecore=fcid.ecore)
    mf = build_reference_uhf_solver(fcid, mol, conv_tol=1e-10, max_cycle=50)
    mf.kernel()

    assert mf.converged is True
    assert mf.e_tot == pytest.approx(-12.0)


def test_run_reference_uccsd_on_toy_hamiltonian(tmp_path):
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    result = run_reference_ucc(fcid, uhf_npz, run_triples=False, conv_tol=1e-10, max_cycle=50)

    assert result.method == "UCCSD"
    assert result.converged is True
    assert result.uhf_energy == pytest.approx(-12.0)
    assert result.energy == pytest.approx(-12.0)
    assert result.correlation_energy == pytest.approx(0.0)
    assert result.s_squared == pytest.approx(0.0)
    assert result.two_s is None


def test_run_reference_uccsd_t_on_toy_hamiltonian(tmp_path):
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    result = run_reference_ucc(fcid, uhf_npz, run_triples=True, conv_tol=1e-10, max_cycle=50)

    assert result.method == "UCCSD(T)"
    assert result.converged is True
    assert result.energy == pytest.approx(-12.0)
    assert result.et_correction == pytest.approx(0.0)
    assert result.ccsd_t_total == pytest.approx(-12.0)


def test_saved_reference_ucc_result_roundtrips_through_parser(tmp_path):
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)
    result = run_reference_ucc(fcid, uhf_npz, run_triples=True, conv_tol=1e-10, max_cycle=50)

    out_npz = os.path.join(tmp_path, "toy_ccsd_t_results.npz")
    save_reference_ucc_result(result, out_npz)

    parsed = parse_npz_result(out_npz)
    assert parsed["method"] == "UCCSD(T)"
    assert parsed["energy"] == pytest.approx(-12.0)
    assert parsed["correlation_energy"] == pytest.approx(0.0)


def test_save_reference_ucc_result_persists_optional_observables(tmp_path):
    result = run_reference_ucc(_make_toy_fcidump(), _write_toy_uhf_npz(tmp_path, _make_toy_fcidump())[0], run_triples=False, conv_tol=1e-10, max_cycle=50)
    result.two_s = 3.5
    result.two_sz_fe1 = -4.2
    result.two_sz_fe2 = 4.2
    result.post_scf_observables = {"two_sz_by_metal_label": {"Fe1": -4.2, "Fe2": 4.2}}

    out_npz = os.path.join(tmp_path, "toy_ccsd_results.npz")
    save_reference_ucc_result(result, out_npz)
    data = np.load(out_npz, allow_pickle=True)
    assert float(data["two_s"]) == pytest.approx(3.5)
    assert float(data["two_sz_fe1"]) == pytest.approx(-4.2)
    assert float(data["two_sz_fe2"]) == pytest.approx(4.2)
    assert "post_scf_observables_json" in data.files


def test_run_reference_uccsd_accepts_h5_uhf_state(tmp_path):
    fcid = _make_toy_fcidump()
    uhf_h5, _ = _write_toy_uhf_h5(tmp_path, fcid)

    result = run_reference_ucc(fcid, uhf_h5, run_triples=False, conv_tol=1e-10, max_cycle=50)

    assert result.method == "UCCSD"
    assert result.converged is True
    assert result.energy == pytest.approx(-12.0)
