"""Regression tests for DMRG solve and extrapolation steps."""

import os

import numpy as np

from apex_filter.session import SessionManager
from apex_filter.steps_dmrg_basis import _write_dmrg_basis_qc_artifacts
from apex_filter.steps_dmrg import step_dmrg, step_extrapolate_dmrg


def _seed_session_for_dmrg(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    for step in [
        "step1_load",
        "step2_enumerate",
        "step3_uhf",
        "step4_ccsd",
        "step5_ccsd_t",
        "step6_ccsdt",
        "step7_dmrg_basis",
    ]:
        sm.mark_step_completed(step)
    return sm


def test_step_dmrg_runs_mocked_solver_and_saves_summary(monkeypatch, tmp_path):
    sm = _seed_session_for_dmrg(tmp_path)

    monkeypatch.setattr(
        SessionManager,
        "load_load_state",
        lambda self: {"fcidump_data": object(), "fcidump_path": "FCIDUMP.mock"},
    )
    cfg = type("Cfg", (), {"label": "BS7_235", "spin_isomer": type("Iso", (), {"family": "BS7_1"})()})()
    monkeypatch.setattr(
        SessionManager,
        "load_enumeration",
        lambda self: {"configs": [cfg]},
    )
    monkeypatch.setattr(
        SessionManager,
        "load_dmrg_basis_summary",
        lambda self: [{"label": "BS7_235", "converged": True, "family": "BS7_1"}],
    )

    def fake_run(*args, **kwargs):
        assert kwargs["fcidump_path"] == "FCIDUMP.mock"
        return type(
            "R",
            (),
            {
                "energy": -1.23 - 0.001 * kwargs["bond_dim"],
                "correlation_energy": -0.1,
                "converged": True,
                "s_squared": 0.75,
                "uhf_energy": -1.0,
                "backend": kwargs["backend"],
                "basis_mode": kwargs["basis_mode"],
                "bond_dim": kwargs["bond_dim"],
                "n_sweeps": kwargs["n_sweeps"],
                "bond_dims": [kwargs["bond_dim"]] * kwargs["n_sweeps"],
                "noises": [1e-5] * kwargs["n_sweeps"],
                "thresholds": [1e-8] * kwargs["n_sweeps"],
            },
        )()

    monkeypatch.setattr("apex_filter.steps_dmrg.run_reference_dmrg", fake_run)
    monkeypatch.setattr(
        "apex_filter.steps_dmrg.save_reference_dmrg_result",
        lambda result, npz_path: np.savez(npz_path, dmrg_total=result.energy, bond_dim=result.bond_dim),
    )

    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    basis_dir = sm.dmrg_basis_results_dir
    os.makedirs(uhf_dir, exist_ok=True)
    os.makedirs(basis_dir, exist_ok=True)
    np.savez(os.path.join(uhf_dir, "BS7_235_uhf.npz"), dummy=1)
    np.savez(os.path.join(basis_dir, "BS7_235_dmrg_basis.npz"), dummy=1)

    step_dmrg(sm.session_dir, bond_dims=[500, 1000], n_sweeps=4)

    summary = sm.load_dmrg_summary()
    assert len(summary) == 2
    assert {row["bond_dim"] for row in summary} == {500, 1000}
    assert {row["backend"] for row in summary} == {"pyblock2_sz"}
    assert {row["basis_mode"] for row in summary} == {"step7_paired"}
    assert all(row["converged"] for row in summary)
    guide_path = os.path.join(sm.session_dir, "step8_dmrg", "selection_guide.md")
    csv_path = os.path.join(sm.session_dir, "step8_dmrg", "selection_candidates.csv")
    worklist_path = os.path.join(sm.session_dir, "step8_dmrg", "selection_worklist.csv")
    assert not os.path.exists(guide_path)
    assert not os.path.exists(csv_path)
    assert not os.path.exists(worklist_path)
    assert not os.path.exists(os.path.join(sm.session_dir, "step8_dmrg", "pick_labels_all.json"))
    assert not os.path.exists(os.path.join(sm.session_dir, "step8_dmrg", "pick_labels_template.json"))


def test_step_dmrg_uses_shell_safe_artifact_names(monkeypatch, tmp_path):
    sm = _seed_session_for_dmrg(tmp_path)

    monkeypatch.setattr(
        SessionManager,
        "load_load_state",
        lambda self: {"fcidump_data": object(), "fcidump_path": "FCIDUMP.mock"},
    )
    cfg = type(
        "Cfg",
        (),
        {"label": "Fe1↓Fe2↑|2xFe(III)|d:none", "spin_isomer": type("Iso", (), {"family": "BS7_1"})()},
    )()
    monkeypatch.setattr(SessionManager, "load_enumeration", lambda self: {"configs": [cfg]})
    monkeypatch.setattr(
        SessionManager,
        "load_dmrg_basis_summary",
        lambda self: [{"label": cfg.label, "converged": True, "family": "BS7_1"}],
    )

    calls = {}

    def fake_run(*args, **kwargs):
        calls["scratch"] = kwargs["scratch"]
        calls["log_path"] = kwargs["log_path"]
        return type(
            "R",
            (),
            {
                "energy": -1.23,
                "correlation_energy": -0.1,
                "converged": True,
                "s_squared": 0.75,
                "uhf_energy": -1.0,
                "backend": kwargs["backend"],
                "basis_mode": kwargs["basis_mode"],
                "bond_dim": kwargs["bond_dim"],
                "n_sweeps": kwargs["n_sweeps"],
                "schedule_mode": kwargs["schedule_mode"],
                "bond_dims": [kwargs["bond_dim"]] * kwargs["n_sweeps"],
                "noises": [1e-5] * kwargs["n_sweeps"],
                "thresholds": [1e-8] * kwargs["n_sweeps"],
            },
        )()

    monkeypatch.setattr("apex_filter.steps_dmrg.run_reference_dmrg", fake_run)
    monkeypatch.setattr(
        "apex_filter.steps_dmrg.save_reference_dmrg_result",
        lambda result, npz_path: np.savez(npz_path, dmrg_total=result.energy, bond_dim=result.bond_dim),
    )

    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    basis_dir = sm.dmrg_basis_results_dir
    os.makedirs(uhf_dir, exist_ok=True)
    os.makedirs(basis_dir, exist_ok=True)
    legacy_safe_label = cfg.label.replace("|", "_").replace(" ", "_")
    np.savez(os.path.join(uhf_dir, f"{legacy_safe_label}_uhf.npz"), dummy=1)
    np.savez(os.path.join(basis_dir, f"{legacy_safe_label}_dmrg_basis.npz"), dummy=1)

    step_dmrg(sm.session_dir, bond_dims=[1000], n_sweeps=4)

    assert "Fe1" in calls["scratch"]
    assert "(" not in calls["scratch"]
    assert ")" not in calls["scratch"]
    assert ":" not in calls["scratch"]
    assert "|" not in calls["scratch"]
    assert all(ord(ch) < 128 for ch in calls["scratch"])
    assert all(ord(ch) < 128 for ch in calls["log_path"])
    assert calls["scratch"].endswith("_scratch")
    token = os.path.basename(calls["log_path"]).replace("_M1000_dmrg.log", "")
    assert len(token.rsplit("_", 1)[-1]) == 8


def test_shell_safe_artifact_token_is_unique_for_distinct_labels():
    from apex_filter.steps_dmrg import _shell_safe_artifact_token

    label_a = "Fe1↓Fe2↑|2xFe(III)|d:none"
    label_b = "Fe1↑Fe2↓|2xFe(III)|d:none"
    token_a = _shell_safe_artifact_token(label_a)
    token_b = _shell_safe_artifact_token(label_b)

    assert token_a != token_b
    assert token_a.startswith("Fe1Fe2_2xFe_III_d_none_")
    assert token_b.startswith("Fe1Fe2_2xFe_III_d_none_")


def test_step_extrapolate_dmrg_groups_by_label_and_saves_result(monkeypatch, tmp_path):
    sm = _seed_session_for_dmrg(tmp_path)
    sm.mark_step_completed("step8_dmrg")
    sm.save_dmrg_summary(
        [
            {"label": "BS7_235", "bond_dim": 500, "energy": -100.0, "converged": True, "family": "BS7_1"},
            {"label": "BS7_235", "bond_dim": 1000, "energy": -100.2, "converged": True, "family": "BS7_1"},
            {"label": "BS8_237", "bond_dim": 500, "energy": -99.0, "converged": False, "family": "BS8_1"},
        ]
    )

    step_extrapolate_dmrg(sm.session_dir)

    summary = sm.load_dmrg_extrapolation_summary()
    assert len(summary) == 1
    assert summary[0]["label"] == "BS7_235"
    assert summary[0]["bond_dims"] == [500, 1000]
    guide_path = os.path.join(sm.session_dir, "step9_extrapolate", "selection_guide.md")
    csv_path = os.path.join(sm.session_dir, "step9_extrapolate", "selection_candidates.csv")
    worklist_path = os.path.join(sm.session_dir, "step9_extrapolate", "selection_worklist.csv")
    assert not os.path.exists(guide_path)
    assert not os.path.exists(csv_path)
    assert not os.path.exists(worklist_path)


def test_write_dmrg_basis_qc_artifacts(tmp_path):
    step_dir = tmp_path / "step7_dmrg_basis"
    step_dir.mkdir()
    _write_dmrg_basis_qc_artifacts(
        str(step_dir),
        [
            {
                "label": "BS7_235",
                "display_label": "Fe1:dz^2",
                "converged": True,
                "orth_err_alpha": 1e-14,
                "orth_err_beta": 2e-14,
                "pair_diag_overlap_min": 0.8,
                "pair_diag_overlap_mean": 0.9,
                "diag_dominant_fraction": 0.95,
                "ordering_is_permutation": True,
                "ga_cost": 1.0,
                "fiedler_cost": 1.2,
            }
        ],
    )
    assert (step_dir / "dmrg_basis_qc.json").exists()
    assert (step_dir / "dmrg_basis_qc.csv").exists()


def test_step7_worklist_defaults_keep_to_one(monkeypatch, tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    for step in [
        "step1_load",
        "step2_enumerate",
        "step3_uhf",
        "step4_ccsd",
        "step5_ccsd_t",
        "step6_ccsdt",
    ]:
        sm.mark_step_completed(step)

    cfg = type("Cfg", (), {"label": "BS7_235", "spin_isomer": type("Iso", (), {"family": "BS7_1"})()})()
    monkeypatch.setattr(
        SessionManager,
        "load",
        lambda self: {"config_path": "dummy.yaml", "completed_steps": ["step1_load", "step2_enumerate", "step3_uhf", "step4_ccsd", "step5_ccsd_t", "step6_ccsdt"]},
    )
    monkeypatch.setattr(SessionManager, "load_load_state", lambda self: {"cas": object(), "fcidump_data": object()})
    monkeypatch.setattr(SessionManager, "load_enumeration", lambda self: {"configs": [cfg]})
    monkeypatch.setattr(SessionManager, "load_ccsdt_summary", lambda self: [{"label": "BS7_235", "display_label": "Fe1:dz^2"}])
    monkeypatch.setattr("apex_filter.steps_dmrg_basis.load_filter_inputs", lambda config: type("Inputs", (), {"mol": object()})())

    basis = type(
        "Basis",
        (),
        {
            "localization_method": "pm",
            "source_method": "UCCSD-NO/split-localized/paired/GA-ordered",
            "nocc_alpha": 1,
            "nocc_beta": 1,
            "orth_err_alpha": 1e-14,
            "orth_err_beta": 2e-14,
            "pair_diag_overlap_min": 0.8,
            "pair_diag_overlap_mean": 0.9,
            "diag_dominant_fraction": 1.0,
            "ordering_is_permutation": True,
            "ga_cost": 1.0,
            "fiedler_cost": 1.1,
        },
    )()
    monkeypatch.setattr("apex_filter.steps_dmrg_basis.build_dmrg_orbital_basis", lambda *args, **kwargs: basis)
    monkeypatch.setattr("apex_filter.steps_dmrg_basis.save_dmrg_orbital_basis", lambda result, path: np.savez(path, dummy=1))

    from apex_filter.steps_dmrg_basis import step_dmrg_basis

    step_dmrg_basis(sm.session_dir)

    worklist = os.path.join(sm.session_dir, "step7_dmrg_basis", "selection_worklist.csv")
    with open(worklist) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    assert lines[1].startswith("1,")
