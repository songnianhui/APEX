"""Regression tests for FNO high-order CC and composite steps."""

import os

import numpy as np

from apex_filter.session import SessionManager
from apex_filter.steps_fno import step_cc_composite, step_fno_uccsdtq


def _seed_session_for_fno(tmp_path):
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
    return sm


def test_step_fno_uccsdtq_runs_mocked_subspace_and_hast(monkeypatch, tmp_path):
    sm = _seed_session_for_fno(tmp_path)
    monkeypatch.setattr(SessionManager, "load_load_state", lambda self: {"fcidump_data": object()})
    cfg = type("Cfg", (), {"label": "BS7_235", "spin_isomer": type("Iso", (), {"family": "BS7"})()})()
    monkeypatch.setattr(SessionManager, "load_enumeration", lambda self: {"configs": [cfg]})
    monkeypatch.setattr(
        SessionManager,
        "load_ccsdt_summary",
        lambda self: [{"label": "BS7_235", "energy": -100.2, "converged": True, "family": "BS7"}],
    )

    monkeypatch.setattr(
        "apex_filter.steps_fno.build_fno_subspace_from_uccsd",
        lambda *args, **kwargs: type(
            "Subspace",
            (),
            {
                "frozen": ([0], [0]),
                "mo_coeff": (np.eye(2), np.eye(2)),
                "mo_occ": (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
                "mo_energy": (np.array([-1.0, 0.5]), np.array([-1.0, 0.5])),
                "occupied_noons_alpha": np.array([1.0]),
                "occupied_noons_beta": np.array([1.0]),
                "kept_occ_alpha": 1,
                "kept_occ_beta": 1,
                "frozen_occ_alpha": 1,
                "frozen_occ_beta": 1,
                "uccsd_energy": -100.0,
                "uccsd_corr": -1.0,
                "converged": True,
            },
        )(),
    )

    def fake_hast(*args, **kwargs):
        t_order = kwargs["t_order"]
        energy = -100.3 if t_order == 3 else -100.35
        return type(
            "HAST",
            (),
            {"energy": energy, "correlation_energy": -1.2, "converged": True},
        )()

    monkeypatch.setattr("apex_filter.steps_fno.run_reference_hast_ucc", fake_hast)

    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    os.makedirs(uhf_dir, exist_ok=True)
    np.savez(os.path.join(uhf_dir, "BS7_235_uhf.npz"), dummy=1)

    step_fno_uccsdtq(sm.session_dir, freeze_occ=[1])

    summary = sm.load_fno_summary()
    assert len(summary) == 1
    assert summary[0]["fno_scheme"] == "occupied_no_freeze"
    assert summary[0]["freeze_occ"] == 1
    assert summary[0]["ccsdt_fno_energy"] == -100.3
    assert summary[0]["ccsdtq_fno_energy"] == -100.35
    worklist = os.path.join(sm.session_dir, "step11_fno_uccsdtq", "selection_worklist.csv")
    assert os.path.exists(worklist)
    with open(worklist) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    assert lines[1].startswith("1,")


def test_step_cc_composite_builds_summary(tmp_path):
    sm = _seed_session_for_fno(tmp_path)
    sm.mark_step_completed("step11_fno_uccsdtq")
    sm.save_ccsdt_summary([{"label": "BS7_235", "energy": -100.2, "converged": True, "family": "BS7"}])
    sm.save_fno_summary(
        [
            {
                "label": "BS7_235",
                "family": "BS7",
                "freeze_occ": 1,
                "ccsdt_fno_energy": -100.3,
                "ccsdtq_fno_energy": -100.35,
                "converged": True,
            }
        ]
    )

    step_cc_composite(sm.session_dir)

    summary = sm.load_cc_composite_summary()
    assert len(summary) == 1
    assert summary[0]["label"] == "BS7_235"
    assert summary[0]["fno_scheme"] == "occupied_no_freeze"
    assert summary[0]["freeze_occ"] == 1
    assert summary[0]["energy"] == -100.25
    worklist = os.path.join(sm.session_dir, "step12_cc_composite", "selection_worklist.csv")
    assert os.path.exists(worklist)
    with open(worklist) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    assert lines[1].startswith("1,")
