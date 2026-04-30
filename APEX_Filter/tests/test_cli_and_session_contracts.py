"""CLI and session contract regression tests."""

import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apex_filter.CAS_loader import FCIDUMPData
from apex_filter.main import create_parser
from apex_filter.models import CAS, ActiveSpaceLevel, ClusterInfo, ComputationSettings
from apex_filter.session import SessionManager


def test_cli_accepts_ccsd_t():
    parser = create_parser()

    args = parser.parse_args(["ccsd-t", "--session", "demo"])
    assert args.command == "ccsd-t"


def test_cli_accepts_ccsdt():
    parser = create_parser()

    args = parser.parse_args(["ccsdt", "--session", "demo"])
    assert args.command == "ccsdt"


def test_cli_accepts_ccsdt_convergence_controls():
    parser = create_parser()

    args = parser.parse_args(
        [
            "ccsdt",
            "--session",
            "demo",
            "--conv-tol",
            "1e-10",
            "--residual-tol",
            "1e-7",
            "--max-cycle",
            "777",
            "--diis-space",
            "12",
            "--diis-start-cycle",
            "4",
            "--iterative-damping",
            "0.85",
            "--level-shift",
            "0.2",
            "--newton-krylov",
        ]
    )
    assert args.command == "ccsdt"
    assert args.conv_tol == 1e-10
    assert args.residual_tol == 1e-7
    assert args.max_cycle == 777
    assert args.diis_space == 12
    assert args.diis_start_cycle == 4
    assert args.iterative_damping == 0.85
    assert args.level_shift == 0.2
    assert args.newton_krylov is True


def test_cli_accepts_dmrg_basis():
    parser = create_parser()

    args = parser.parse_args(["dmrg-basis", "--session", "demo"])
    assert args.command == "dmrg-basis"


def test_cli_accepts_dmrg_basis_controls():
    parser = create_parser()

    args = parser.parse_args(
        [
            "dmrg-basis",
            "--session",
            "demo",
            "--cc-conv-tol",
            "1e-10",
            "--cc-max-cycle",
            "777",
            "--cc-diis-space",
            "16",
            "--cc-direct",
            "--pm-pop-method",
            "mulliken",
            "--pm-conv-tol",
            "1e-8",
            "--pm-conv-tol-grad",
            "1e-4",
            "--pm-max-cycle",
            "250",
            "--boys-conv-tol",
            "1e-7",
            "--boys-conv-tol-grad",
            "1e-5",
            "--boys-max-cycle",
            "150",
            "--ordering-matrix-mode",
            "overlap_proxy",
            "--exchange-proxy-max-orbitals",
            "32",
            "--ga-generations",
            "40",
            "--ga-population",
            "60",
            "--ga-mutation-rate",
            "0.2",
            "--ga-seed",
            "11",
        ]
    )
    assert args.command == "dmrg-basis"
    assert args.cc_conv_tol == 1e-10
    assert args.cc_max_cycle == 777
    assert args.cc_diis_space == 16
    assert args.cc_direct is True
    assert args.pm_pop_method == "mulliken"
    assert args.pm_conv_tol == 1e-8
    assert args.pm_conv_tol_grad == 1e-4
    assert args.pm_max_cycle == 250
    assert args.boys_conv_tol == 1e-7
    assert args.boys_conv_tol_grad == 1e-5
    assert args.boys_max_cycle == 150
    assert args.ordering_matrix_mode == "overlap_proxy"
    assert args.exchange_proxy_max_orbitals == 32
    assert args.ga_generations == 40
    assert args.ga_population == 60
    assert args.ga_mutation_rate == 0.2
    assert args.ga_seed == 11


def test_cli_accepts_dmrg():
    parser = create_parser()

    args = parser.parse_args(["dmrg", "--session", "demo"])
    assert args.command == "dmrg"
    assert args.backend == "pyblock2_sz"
    assert args.basis_mode == "step7_paired"


def test_cli_accepts_dmrg_backend_and_basis_controls():
    parser = create_parser()

    args = parser.parse_args(
        [
            "dmrg",
            "--session",
            "demo",
            "--backend",
            "pyscf_dmrgci_sz",
            "--basis-mode",
            "original_identity",
            "--schedule-mode",
            "benchmark",
            "--stack-mem",
            "123456",
            "--twosite-to-onesite",
            "22",
            "--dav-max-iter",
            "5000",
            "--dav-def-max-size",
            "64",
            "--dav-rel-conv-thrd",
            "1e-3",
            "--dav-type",
            "NoPrecond",
        ]
    )
    assert args.backend == "pyscf_dmrgci_sz"
    assert args.basis_mode == "original_identity"
    assert args.schedule_mode == "benchmark"
    assert args.stack_mem == 123456
    assert args.twosite_to_onesite == 22
    assert args.dav_max_iter == 5000
    assert args.dav_def_max_size == 64
    assert args.dav_rel_conv_thrd == 1e-3
    assert args.dav_type == "NoPrecond"


def test_cli_accepts_extrapolate():
    parser = create_parser()

    args = parser.parse_args(["extrapolate", "--session", "demo"])
    assert args.command == "extrapolate"


def test_cli_accepts_report():
    parser = create_parser()

    args = parser.parse_args(["report", "--session", "demo"])
    assert args.command == "report"


def test_cli_accepts_fno_uccsdtq():
    parser = create_parser()

    args = parser.parse_args(["fno-uccsdtq", "--session", "demo"])
    assert args.command == "fno-uccsdtq"


def test_cli_accepts_cc_composite():
    parser = create_parser()

    args = parser.parse_args(["cc-composite", "--session", "demo"])
    assert args.command == "cc-composite"


def test_cli_accepts_uhf_stabilization_controls():
    parser = create_parser()

    args = parser.parse_args(
        [
            "uhf",
            "--session",
            "demo",
            "--stabilize-cycles",
            "80",
            "--level-shift",
            "0.5",
            "--damp",
            "0.3",
        ]
    )
    assert args.command == "uhf"
    assert args.stabilize_cycles == 80
    assert args.level_shift == 0.5
    assert args.damp == 0.3


def test_cli_accepts_uhf_newton_controls():
    parser = create_parser()

    args = parser.parse_args(
        [
            "uhf",
            "--session",
            "demo",
            "--newton-refine",
            "--newton-max-cycle",
            "6",
        ]
    )
    assert args.command == "uhf"
    assert args.newton_refine is True
    assert args.newton_max_cycle == 6


def test_cli_rejects_noncanonical_ccsd_code():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ccsd", "--session", "demo", "--code", "hast_ucc"])


def test_cli_rejects_noncanonical_ccsd_t_code():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ccsd-t", "--session", "demo", "--code", "hast_ucc"])


def test_cli_rejects_noncanonical_ccsdt_code():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ccsdt", "--session", "demo", "--code", "pyscf"])


def test_session_create_includes_step6_ccsdt(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step6_ccsdt"))
    assert os.path.isdir(sm.ccsdt_scripts_dir)


def test_session_create_writes_method_controls_yaml(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.exists(sm.method_controls_path)
    text = open(sm.method_controls_path).read()
    assert "uhf:" in text
    assert "ccsdt:" in text
    assert "dmrg:" in text


def test_session_method_controls_override_defaults(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "ccsdt:\n"
            "  max_cycle: 777\n"
            "  residual_tol: 1.0e-5\n"
        )

    resolved = sm.resolve_method_controls(
        "ccsdt",
        {"max_cycle": 2000, "residual_tol": 1.0e-6},
        {"max_cycle": 2000, "residual_tol": 1.0e-6},
    )
    assert resolved["max_cycle"] == 777
    assert resolved["residual_tol"] == 1.0e-5


def test_session_create_includes_step7_dmrg_basis(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step7_dmrg_basis"))
    assert os.path.isdir(sm.dmrg_basis_results_dir)


def test_session_create_includes_step8_dmrg_and_step9_extrapolate(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step8_dmrg"))
    assert os.path.isdir(os.path.join(sm.session_dir, "step9_extrapolate"))
    assert os.path.isdir(sm.dmrg_results_dir)


def test_session_create_includes_step10_report(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step10_report"))


def test_session_create_includes_step11_and_step12(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step11_fno_uccsdtq"))
    assert os.path.isdir(os.path.join(sm.session_dir, "step12_cc_composite"))
    assert os.path.isdir(sm.fno_results_dir)


def test_save_uhf_result_includes_mo_energy_and_density(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    result = SimpleNamespace(
        energy=-1.23,
        converged=True,
        s_squared=0.75,
        mo_coeff=(np.eye(2), np.eye(2)),
        mo_occ=(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        mo_energy=(np.array([-0.5, 0.2]), np.array([-0.4, 0.3])),
        dm=(np.eye(2) * 0.8, np.eye(2) * 0.2),
        diagnostics={
            "bs_stabilize_history": [
                {"energy": -1.2, "delta_e": -0.1},
                {"energy": -1.23, "delta_e": -0.03},
            ],
            "bs_tight_history": [
                {"energy": -1.231, "delta_e": -0.001},
                {"energy": -1.2312, "delta_e": -0.0002},
            ],
            "final_delta_e": -0.0002,
        },
    )

    fake_state = {
        "settings": ComputationSettings(),
        "cluster_info": ClusterInfo(
            all_elements=["H"],
            all_positions=np.array([[0.0, 0.0, 0.0]]),
            total_charge=-2,
            target_spin=2.0,
        ),
        "fcidump_data": FCIDUMPData(
            h1e=np.eye(2),
            h2e=np.zeros((2, 2, 2, 2)),
            ecore=-1.0,
            norb=2,
            nelec=2,
            ms2=0,
        ),
        "cas": CAS(
            n_electrons=2,
            n_orbitals=2,
            level=ActiveSpaceLevel.MINIMAL,
            active_indices=[0, 1],
            orbital_labels=["Fe1_dxy", "Fe2_dxy"],
        ),
    }

    sm.save_uhf_result("BS7|235", result, state=fake_state)

    npz_path = os.path.join(sm.session_dir, "step3_uhf", "results", "BS7_235_uhf.npz")
    data = np.load(npz_path)

    assert "mo_energy_a" in data.files
    assert "mo_energy_b" in data.files
    assert "dm_a" in data.files
    assert "dm_b" in data.files
    assert "bs_stabilize_energy_history" in data.files
    assert "bs_tight_delta_e_history" in data.files
    assert "final_delta_e" in data.files
    assert np.allclose(data["mo_energy_a"], [-0.5, 0.2])
    assert np.allclose(data["dm_b"], np.eye(2) * 0.2)
    assert np.allclose(data["bs_stabilize_energy_history"], [-1.2, -1.23])
    assert np.allclose(data["bs_tight_delta_e_history"], [-0.001, -0.0002])
    assert np.isclose(float(data["final_delta_e"]), -0.0002)

    h5_path = os.path.join(sm.session_dir, "step3_uhf", "results", "BS7_235_uhf.h5")
    assert os.path.exists(h5_path)
    with h5py.File(h5_path, "r") as f:
        assert f["metadata"].attrs["label"] == "BS7|235"
        assert f["metadata"].attrs["family"] == ""
        assert "settings_json" in f["metadata"].attrs
        assert f["molecule"].attrs["basis_set_default"] == "def2-TZVP"
        assert "serialized_xyz" in f["molecule"].attrs
        assert "serialized_solver_mol" in f["molecule"].attrs
        assert "active_indices" in f["active_space_mapping"]
