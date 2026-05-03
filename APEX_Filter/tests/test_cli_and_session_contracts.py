"""CLI and session contract regression tests."""

import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apex_filter.main import create_parser
from apex_filter.session import SessionManager
from shared.fcidump_io import FCIDUMPData
from shared.models import CAS, ActiveSpaceLevel, ClusterInfo, ComputationSettings


def test_cli_accepts_ccsd_t():
    parser = create_parser()

    args = parser.parse_args(["ccsd-t", "--session", "demo"])
    assert args.command == "ccsd-t"


def test_load_state_ignores_unknown_extra_settings(tmp_path, monkeypatch):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    step1 = os.path.join(sm.session_dir, "step1_load")
    with open(os.path.join(step1, "cluster_info.json"), "w", encoding="utf-8") as f:
        json.dump({"metals": [], "bridging_atoms": [], "terminal_ligands": [], "all_elements": [], "all_positions": None, "formula": "", "total_charge": 0, "target_spin": 0.0, "symmetry_group": "C1", "metal_framework_symmetry": "C1", "reduction_symmetry": "C1", "symmetry_axis_atoms": [], "symmetry_source": "auto", "symmetry_confidence": 0.0, "symmetry_candidates": [], "family_scheme": "", "benchmark_profile": "", "config_reduction_mode": "none", "cluster_info_path": "", "annotation_source": "auto"}, f)
    with open(os.path.join(step1, "cas_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_electrons": 2, "n_orbitals": 2}, f)
    np.savez(os.path.join(step1, "cas_arrays.npz"))
    with open(os.path.join(step1, "fcidump_ref.json"), "w", encoding="utf-8") as f:
        json.dump({"fcidump_path": str(tmp_path / "FCIDUMP.test")}, f)
    with open(os.path.join(step1, "settings.json"), "w", encoding="utf-8") as f:
        json.dump({"basis_set_default": "def2-TZVP", "scf_method": "uks", "scf_spin": 5.0}, f)

    monkeypatch.setattr("apex_filter.session._load_fcidump", lambda path: "fcidump")

    state = sm.load_load_state()
    assert state["settings"].basis_set_default == "def2-TZVP"
    assert state["settings"].scf_spin == 5.0


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
            "--pm-exponent",
            "4",
            "--pm-init-guess",
            "cholesky",
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
    assert args.pm_exponent == 4
    assert args.pm_init_guess == "cholesky"
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
    assert os.path.isdir(sm.step_artifact_dir("step6_ccsdt", "scripts"))


def test_session_create_writes_method_controls_yaml(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.exists(sm.method_controls_path)
    text = open(sm.method_controls_path).read()
    assert "uhf:" in text
    assert "ccsdt:" in text
    assert "dmrg:" in text
    assert "Chan 2026 Fe2S2 oxidized benchmark coverage" in text


def test_step1_settings_json_remains_flat_bootstrap_snapshot(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    cluster_info = ClusterInfo(
        all_elements=["H"],
        all_positions=np.array([[0.0, 0.0, 0.0]]),
        total_charge=-2,
        target_spin=2.0,
    )
    cas = CAS(
        n_electrons=2,
        n_orbitals=2,
        level=ActiveSpaceLevel.MINIMAL,
        active_indices=[0, 1],
        orbital_labels=["Fe1_dxy", "Fe2_dxy"],
    )
    settings = ComputationSettings(
        basis_set_default="def2-TZVP",
        scf_method="uks",
        xc_functional="BP86",
        solvation_model="none",
        conv_tol=1e-10,
    )

    sm.save_load_state(
        cluster_info,
        cas,
        str(tmp_path / "FCIDUMP.test"),
        settings,
        str(tmp_path / "filter_settings.yaml"),
        apex_cas_provenance={"cluster_info_path": "/tmp/cluster_info.yaml"},
    )

    with open(os.path.join(sm.session_dir, "step1_load", "settings.json"), encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["basis_set_default"] == "def2-TZVP"
    assert payload["scf_method"] == "uks"
    assert payload["xc_functional"] == "BP86"
    assert payload["conv_tol"] == 1e-10
    assert payload["apex_cas_provenance"]["cluster_info_path"] == "/tmp/cluster_info.yaml"
    assert "requested_config" not in payload
    assert "effective_method" not in payload
    assert "effective_parameters" not in payload


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


def test_session_method_controls_ignore_none_cli_values(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "dmrg:\n"
            "  bond_dims: [400, 800]\n"
            "  schedule_mode: benchmark\n"
        )

    resolved = sm.resolve_method_controls(
        "dmrg",
        {"bond_dims": [100, 200], "schedule_mode": "workflow"},
        {"bond_dims": None, "schedule_mode": None},
    )
    assert resolved["bond_dims"] == [400, 800]
    assert resolved["schedule_mode"] == "benchmark"


def test_session_method_controls_ignore_none_optional_dmrg_cli_values(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "dmrg:\n"
            "  twosite_to_onesite: 24\n"
            "  dav_max_iter: 5000\n"
            "  dav_def_max_size: 64\n"
            "  dav_rel_conv_thrd: 1.0e-3\n"
            "  dav_type: NoPrecond\n"
        )

    resolved = sm.resolve_method_controls(
        "dmrg",
        {
            "twosite_to_onesite": None,
            "dav_max_iter": None,
            "dav_def_max_size": None,
            "dav_rel_conv_thrd": None,
            "dav_type": None,
        },
        {
            "twosite_to_onesite": None,
            "dav_max_iter": None,
            "dav_def_max_size": None,
            "dav_rel_conv_thrd": None,
            "dav_type": None,
        },
    )
    assert resolved["twosite_to_onesite"] == 24
    assert resolved["dav_max_iter"] == 5000
    assert resolved["dav_def_max_size"] == 64
    assert resolved["dav_rel_conv_thrd"] == 1.0e-3
    assert resolved["dav_type"] == "NoPrecond"


def test_session_method_controls_cli_override_beats_session_yaml(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "dmrg_basis:\n"
            "  localization_method: boys\n"
            "  ga_seed: 17\n"
        )

    resolved = sm.resolve_method_controls(
        "dmrg_basis",
        {"localization_method": "pm", "ga_seed": 17},
        {"localization_method": "pm_lowdin", "ga_seed": 23},
    )
    assert resolved["localization_method"] == "pm_lowdin"
    assert resolved["ga_seed"] == 23


def test_session_method_controls_cli_override_beats_session_yaml_for_optional_dmrg_controls(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "dmrg:\n"
            "  twosite_to_onesite: 24\n"
            "  dav_max_iter: 5000\n"
        )

    resolved = sm.resolve_method_controls(
        "dmrg",
        {
            "twosite_to_onesite": None,
            "dav_max_iter": None,
        },
        {
            "twosite_to_onesite": 18,
            "dav_max_iter": 4000,
        },
    )
    assert resolved["twosite_to_onesite"] == 18
    assert resolved["dav_max_iter"] == 4000


def test_session_method_controls_default_cli_does_not_override_session_yaml(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    with open(sm.method_controls_path, "w") as f:
        f.write(
            "uhf:\n"
            "  max_cycle: 777\n"
            "  conv_tol: 1.0e-10\n"
        )

    defaults = {"max_cycle": 2000, "conv_tol": 1.0e-8}
    resolved = sm.resolve_method_controls("uhf", defaults, dict(defaults))
    assert resolved["max_cycle"] == 777
    assert resolved["conv_tol"] == 1.0e-10


def test_session_build_step_settings_payload_uses_control_source_and_theory(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    payload = sm._build_step_settings_payload(
        {"basis_set_default": "tzp-dkh", "scf_method": "uks"},
        theory="UHF",
        conv_tol=1.0e-10,
    )

    assert payload["basis_set_default"] == "tzp-dkh"
    assert payload["scf_method"] == "uks"
    assert payload["control_source"] == sm.method_controls_path
    assert payload["theory"] == "UHF"
    assert payload["conv_tol"] == 1.0e-10


def test_session_build_step_settings_payload_accepts_empty_source_settings(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    payload = sm._build_step_settings_payload(
        None,
        theory="DMRG",
        bond_dim=1000,
        n_sweeps=8,
    )

    assert payload == {
        "control_source": sm.method_controls_path,
        "theory": "DMRG",
        "bond_dim": 1000,
        "n_sweeps": 8,
    }


def test_session_create_includes_step7_dmrg_basis(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step7_dmrg_basis"))
    assert os.path.isdir(sm.step_artifact_dir("step7_dmrg_basis", "results"))


def test_session_create_includes_step8_dmrg_and_step9_extrapolate(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    assert os.path.isdir(os.path.join(sm.session_dir, "step8_dmrg"))
    assert os.path.isdir(os.path.join(sm.session_dir, "step9_extrapolate"))
    assert os.path.isdir(sm.step_artifact_dir("step8_dmrg", "results"))


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

    sm.save_uhf_result(
        "BS7|235",
        result,
        state=fake_state,
        settings_payload={
            "control_source": "/tmp/method_controls.yaml",
            "theory": "UHF",
            "conv_tol": 1e-8,
        },
    )

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
        settings = json.loads(f["metadata"].attrs["settings_json"])
        assert settings["control_source"] == "/tmp/method_controls.yaml"
        assert settings["theory"] == "UHF"
        assert settings["effective_method"]["theory"] == "UHF"
        for key in ("scf_method", "xc_functional", "relativistic", "solvation_model"):
            assert key not in settings["effective_parameters"]
        assert settings["effective_parameters"]["conv_tol"] == 1e-8
        assert settings["requested_config"]["conv_tol"] == 1e-8
        assert f["molecule"].attrs["basis_set_default"] == "def2-TZVP"
        assert "serialized_xyz" in f["molecule"].attrs
        assert "serialized_solver_mol" in f["molecule"].attrs
        assert "active_indices" in f["active_space_mapping"]
