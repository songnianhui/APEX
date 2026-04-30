"""Regression tests for active-space HAST-UCC on FCIDUMP Hamiltonians."""

import os
import sys
import types

import h5py
import numpy as np
import pytest

from apex_filter.CAS_loader import FCIDUMPData
from apex_filter.reference_hast_ucc import run_reference_hast_ucc, save_reference_hast_result
from apex_filter.reference_uhf import build_fake_mol, build_reference_uhf_solver
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

    npz_path = os.path.join(tmp_path, "toy_uhf_hast.npz")
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


def _install_fake_pyhast(monkeypatch):
    class FakeUCC:
        def __init__(
            self,
            mf,
            t_order=3,
            verbose=4,
            diis=True,
            eval_t=None,
            level_shift=0.0,
            newton_krylov=False,
            frozen=0,
            gen_lamb_eq=False,
            gen_npdm_eq=False,
            npdm_order=None,
        ):
            self.mf = mf
            self.t_order = t_order
            self.verbose = verbose
            self.diis = diis
            self.eval_t = eval_t
            self.level_shift = level_shift
            self.newton_krylov = newton_krylov
            self.frozen = frozen
            self.gen_lamb_eq = gen_lamb_eq
            self.gen_npdm_eq = gen_npdm_eq
            self.npdm_order = npdm_order
            self.converged = True
            self.e_corr = -0.125 if t_order == 3 else -0.25
            self.e_tot = float(mf.e_tot) + self.e_corr
            self.t1_norm = 0.02
            self.diis_space = 6
            self.diis_start_cycle = 0
            self.iterative_damping = 1.0
            self.kernel_kwargs = None
            self.lamps = None
            self.tamps = [[np.zeros((1, 1))], [np.zeros((1, 1, 1, 1))]]
            self.lamb_order = 2

        def kernel(self, max_cycle=50, tol=1e-8, tolnormt=1e-6):
            self.kernel_kwargs = {
                "max_cycle": max_cycle,
                "tol": tol,
                "tolnormt": tolnormt,
            }
            return self.e_tot

        def amplitudes_to_vector(self, amps):
            return np.zeros(2)

        def vector_to_amplitudes(self, vec, order):
            return self.tamps

        def solve_lambda(self, tol=1e-8, max_cycle=50, lamps=None):
            self.lamps = ([[np.zeros((1, 1))]], [[np.zeros((1, 1, 1, 1))]])
            return True, self.lamps

        def make_npdms(self, order=2):
            dm1 = [
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
            ]
            dm2 = [
                np.zeros((2, 2, 2, 2)),
                np.zeros((2, 2, 2, 2)),
                np.zeros((2, 2, 2, 2)),
            ]
            return dm1, dm2

    pyhast_mod = types.ModuleType("pyhast")
    sr_mod = types.ModuleType("pyhast.sr")
    ucc_mod = types.ModuleType("pyhast.sr.ucc")
    ucc_mod.UCC = FakeUCC
    sr_mod.ucc = ucc_mod
    pyhast_mod.sr = sr_mod

    monkeypatch.setitem(sys.modules, "pyhast", pyhast_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr", sr_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr.ucc", ucc_mod)


def test_run_reference_hast_uccsdt_on_toy_hamiltonian(tmp_path, monkeypatch):
    _install_fake_pyhast(monkeypatch)
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    result = run_reference_hast_ucc(fcid, uhf_npz, t_order=3)

    assert result.method == "UCCSDT"
    assert result.converged is True
    assert result.uhf_energy == pytest.approx(-12.0)
    assert result.energy == pytest.approx(-12.125)
    assert result.correlation_energy == pytest.approx(-0.125)
    assert result.nominal_backend == "hast_ucc_t3"


def test_saved_reference_hast_result_roundtrips_through_parser(tmp_path, monkeypatch):
    _install_fake_pyhast(monkeypatch)
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)
    result = run_reference_hast_ucc(fcid, uhf_npz, t_order=3)

    out_npz = os.path.join(tmp_path, "toy_ccsdt_results.npz")
    save_reference_hast_result(result, out_npz)

    parsed = parse_npz_result(out_npz)
    assert parsed["method"] == "UCCSDT"
    assert parsed["energy"] == pytest.approx(-12.125)
    assert parsed["correlation_energy"] == pytest.approx(-0.125)
    out_h5 = out_npz[:-4] + ".h5"
    assert os.path.exists(out_h5)
    with h5py.File(out_h5, "r") as f:
        assert "metadata" in f
        assert "orbitals" in f
        assert "density_matrices" in f
        assert f["metadata"].attrs["artifact_type"] == "apex_filter_step6_hast_ucc_state"


def test_run_reference_hast_ucc_passes_convergence_controls(tmp_path, monkeypatch):
    captured = {}

    class FakeUCC:
        def __init__(self, mf, t_order=3, verbose=4, diis=True, eval_t=None, level_shift=0.0, newton_krylov=False, frozen=0):
            self.mf = mf
            self.t_order = t_order
            self.diis = diis
            self.eval_t = eval_t
            self.level_shift = level_shift
            self.newton_krylov = newton_krylov
            self.frozen = frozen
            self.diis_space = 6
            self.diis_start_cycle = 0
            self.iterative_damping = 1.0
            self.converged = True
            self.e_corr = -0.125
            self.e_tot = float(mf.e_tot) - 0.125
            self.t1_norm = 0.01
            self.tamps = [[np.zeros((1, 1))], [np.zeros((1, 1, 1, 1))]]
            self.lamb_order = 2
            captured["ucc"] = self

        def kernel(self, max_cycle=50, tol=1e-8, tolnormt=1e-6):
            captured["kernel_kwargs"] = {
                "max_cycle": max_cycle,
                "tol": tol,
                "tolnormt": tolnormt,
            }
            return self.e_tot

        def amplitudes_to_vector(self, amps):
            return np.zeros(2)

    pyhast_mod = types.ModuleType("pyhast")
    sr_mod = types.ModuleType("pyhast.sr")
    ucc_mod = types.ModuleType("pyhast.sr.ucc")
    ucc_mod.UCC = FakeUCC
    sr_mod.ucc = ucc_mod
    pyhast_mod.sr = sr_mod

    monkeypatch.setitem(sys.modules, "pyhast", pyhast_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr", sr_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr.ucc", ucc_mod)

    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    run_reference_hast_ucc(
        fcid,
        uhf_npz,
        t_order=3,
        conv_tol=1e-10,
        residual_tol=1e-7,
        max_cycle=777,
        diis_space=12,
        diis_start_cycle=4,
        iterative_damping=1.0,
        level_shift=0.2,
        newton_krylov=True,
    )

    assert captured["kernel_kwargs"] == {
        "max_cycle": 777,
        "tol": pytest.approx(1e-10),
        "tolnormt": pytest.approx(1e-7),
    }
    assert captured["ucc"].diis_space == 12
    assert captured["ucc"].diis_start_cycle == 4
    assert captured["ucc"].iterative_damping == pytest.approx(1.0)
    assert captured["ucc"].level_shift == pytest.approx(0.2)
    assert captured["ucc"].newton_krylov is True


def test_run_reference_hast_ucc_rejects_nonunit_iterative_damping(tmp_path, monkeypatch):
    _install_fake_pyhast(monkeypatch)
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    with pytest.raises(ValueError, match="iterative_damping = 1.0"):
        run_reference_hast_ucc(
            fcid,
            uhf_npz,
            iterative_damping=0.7,
        )


def test_run_reference_hast_ucc_can_emit_observables(tmp_path, monkeypatch):
    _install_fake_pyhast(monkeypatch)
    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)

    result = run_reference_hast_ucc(
        fcid,
        uhf_npz,
        observable_inputs={
            "active_indices": np.array([0, 1]),
            "xyz_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2.xyz",
            "cluster_info_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2_cluster_info.yaml",
            "cas_settings_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2_cas_settings.yaml",
            "cas_data_h5_path": "/Users/snh/Projects/APEX/examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_data.h5",
            "label": "toy",
            "family": "toy",
            "theory": "UCCSDT",
        },
    )

    assert result.two_s is not None
    assert result.two_sz_fe1 is not None
    assert result.two_sz_fe2 is not None
    assert result.observables_complete is True
    assert result.observable_error is None


def test_run_reference_hast_ucc_preserves_energy_when_lambda_not_converged(tmp_path, monkeypatch):
    class FakeUCC:
        def __init__(
            self,
            mf,
            t_order=3,
            verbose=4,
            diis=True,
            eval_t=None,
            level_shift=0.0,
            newton_krylov=False,
            frozen=0,
            gen_lamb_eq=False,
            gen_npdm_eq=False,
            npdm_order=None,
        ):
            self.mf = mf
            self.t_order = t_order
            self.verbose = verbose
            self.diis = diis
            self.eval_t = eval_t
            self.level_shift = level_shift
            self.newton_krylov = newton_krylov
            self.frozen = frozen
            self.gen_lamb_eq = gen_lamb_eq
            self.gen_npdm_eq = gen_npdm_eq
            self.npdm_order = npdm_order
            self.converged = True
            self.e_corr = -0.125
            self.e_tot = float(mf.e_tot) + self.e_corr
            self.t1_norm = 0.02
            self.diis_space = 6
            self.diis_start_cycle = 0
            self.iterative_damping = 1.0
            self.tamps = [[np.zeros((1, 1))], [np.zeros((1, 1, 1, 1))]]
            self.lamb_order = 2

        def kernel(self, max_cycle=50, tol=1e-8, tolnormt=1e-6):
            return self.e_tot

        def amplitudes_to_vector(self, amps):
            return np.zeros(2)

        def vector_to_amplitudes(self, vec, order):
            return self.tamps

        def solve_lambda(self, tol=1e-8, max_cycle=50, lamps=None):
            self.lamps = self.tamps
            self.lamb_converged = False
            return False, self.lamps

    pyhast_mod = types.ModuleType("pyhast")
    sr_mod = types.ModuleType("pyhast.sr")
    ucc_mod = types.ModuleType("pyhast.sr.ucc")
    ucc_mod.UCC = FakeUCC
    sr_mod.ucc = ucc_mod
    pyhast_mod.sr = sr_mod
    monkeypatch.setitem(sys.modules, "pyhast", pyhast_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr", sr_mod)
    monkeypatch.setitem(sys.modules, "pyhast.sr.ucc", ucc_mod)

    fcid = _make_toy_fcidump()
    uhf_npz, _ = _write_toy_uhf_npz(tmp_path, fcid)
    result = run_reference_hast_ucc(
        fcid,
        uhf_npz,
        observable_inputs={
            "active_indices": np.array([0, 1]),
            "xyz_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2.xyz",
            "cluster_info_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2_cluster_info.yaml",
            "cas_settings_path": "/Users/snh/Projects/APEX/examples/fe2s2/inputs/fe2s2_cas_settings.yaml",
            "cas_data_h5_path": "/Users/snh/Projects/APEX/examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_data.h5",
        },
        lambda_max_cycle=123,
    )

    assert result.energy == pytest.approx(-12.125)
    assert result.converged is True
    assert result.observables_complete is False
    assert result.lambda_converged is False
    assert "did not converge" in result.observable_error
    assert result.tamps_vector is not None
    assert result.post_scf_observables is not None
