"""Regression tests for the final reporting step."""

import os

import numpy as np

from apex_filter import CAS, BridgingAtom, ClusterInfo, MetalCenter
from apex_filter.models import ActiveSpaceLevel
from apex_filter.report import generate_report
from apex_filter.session import SessionManager
from apex_filter.steps_report import step_report


def _seed_session_for_report(tmp_path):
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
        "step8_dmrg",
        "step9_extrapolate",
    ]:
        sm.mark_step_completed(step)
    return sm


def test_step_report_builds_ranked_final_summary(tmp_path):
    sm = _seed_session_for_report(tmp_path)
    sm.save_uhf_summary(
        [
            {"label": "BS7_235", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS7"},
            {"label": "BS8_237", "energy": -99.90, "converged": True, "s_squared": 1.6, "family": "BS8"},
            {"label": "BS8_236", "energy": -99.85, "converged": True, "s_squared": 1.7, "family": "BS8"},
        ]
    )
    sm.save_ccsd_summary(
        [
            {"label": "BS7_235", "energy": -100.08, "correlation_energy": -0.08, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.01, "correlation_energy": -0.11, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_ccsd_t_summary(
        [
            {"label": "BS7_235", "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2", "energy": -100.10, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.05, "converged": True, "family": "BS8"},
            {"label": "BS8_236", "energy": -99.95, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_ccsdt_summary(
        [
            {"label": "BS7_235", "energy": -100.20, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.10, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_dmrg_extrapolation_summary(
        [
            {
                "label": "BS7_235",
                "energy": -100.24,
                "uncertainty": 5e-4,
                "bond_dims": [5000, 10000],
                "family": "BS7",
            },
            {
                "label": "BS8_236",
                "energy": -100.12,
                "uncertainty": 1e-3,
                "bond_dims": [5000, 10000],
                "family": "BS8",
            },
        ]
    )

    step_report(sm.session_dir)

    summary = sm.load_final_summary()
    assert [row["label"] for row in summary] == ["BS7_235", "BS8_236", "BS8_237"]
    assert summary[0]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"
    assert summary[0]["representative_label"] == "BS7_235"
    assert summary[0]["source_initial_labels"] == ["BS7_235"]
    assert summary[0]["uhf_energy"] == -100.00
    assert summary[0]["ccsd_energy"] == -100.08
    assert summary[0]["ranking_method"] == "CCSDT+DMRG_consensus"
    assert abs(summary[0]["consensus_energy"] - (-100.22)) < 1e-12
    assert abs(summary[0]["consensus_uncertainty"] - 0.02) < 1e-12
    assert summary[1]["ranking_method"] == "DMRG_extrapolated"
    assert summary[2]["ranking_method"] == "CCSDT"

    report_path = os.path.join(sm.session_dir, "step10_report", "final_report.md")
    assert os.path.exists(report_path)
    with open(report_path) as f:
        report = f.read()
    assert "CCSDT+DMRG consensus" in report
    assert "BS7_235" in report
    assert "Fe1↓Fe2↑\\|Fe1(II)+Fe2(III)\\|Fe1:dz^2" in report
    assert "## Provenance" in report
    assert "Representative label" in report
    assert "Method ladder" in report
    assert "UHF:" in report
    assert "CCSD:" in report

    step10_dir = os.path.join(sm.session_dir, "step10_report")
    assert not os.path.exists(os.path.join(step10_dir, "selection_guide.md"))
    assert not os.path.exists(os.path.join(step10_dir, "selection_candidates.csv"))
    assert not os.path.exists(os.path.join(step10_dir, "selection_worklist.csv"))
    assert not os.path.exists(os.path.join(step10_dir, "pick_labels_all.json"))
    assert not os.path.exists(os.path.join(step10_dir, "pick_labels_template.json"))


def test_step_report_prefers_cc_composite_when_available(tmp_path):
    sm = _seed_session_for_report(tmp_path)
    sm.save_uhf_summary([{"label": "BS7_235", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS7"}])
    sm.save_ccsd_summary([{"label": "BS7_235", "energy": -100.08, "correlation_energy": -0.08, "converged": True, "family": "BS7"}])
    sm.save_ccsd_t_summary([{"label": "BS7_235", "energy": -100.10, "converged": True, "family": "BS7"}])
    sm.save_ccsdt_summary([{"label": "BS7_235", "energy": -100.20, "converged": True, "family": "BS7"}])
    sm.save_dmrg_extrapolation_summary([{"label": "BS7_235", "energy": -100.24, "uncertainty": 5e-4, "bond_dims": [5000, 10000], "family": "BS7"}])
    sm.save_cc_composite_summary([{"label": "BS7_235", "family": "BS7", "freeze_occ": 2, "energy": -100.30, "uncertainty": 0.01, "converged": True}])

    step_report(sm.session_dir)

    summary = sm.load_final_summary()
    assert len(summary) == 1
    assert summary[0]["cc_composite_energy"] == -100.30
    assert summary[0]["ranking_method"] == "CC_composite+DMRG_consensus"
    assert abs(summary[0]["ranking_energy"] - (-100.27)) < 1e-12


def test_step_report_groups_by_final_state_and_preserves_provenance(tmp_path):
    sm = _seed_session_for_report(tmp_path)
    sm.save_uhf_summary(
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.01, "converged": True, "s_squared": 1.6, "family": "BS8"},
        ]
    )
    sm.save_ccsd_summary(
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.06, "correlation_energy": -0.06, "converged": True, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.07, "correlation_energy": -0.07, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_ccsd_t_summary(
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.10, "converged": True, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.11, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_ccsdt_summary([])
    sm.save_dmrg_extrapolation_summary(
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.30, "uncertainty": float("inf"), "bond_dims": [500, 1000, 1500], "family": "BS8", "source_mode": "unconverged_fallback"},
        ]
    )

    step_report(sm.session_dir)

    summary = sm.load_final_summary()
    assert len(summary) == 1
    assert summary[0]["display_label"] == "Fe2:dz^2"
    assert summary[0]["source_initial_labels"] == ["guess_d1", "guess_d2"]
    assert summary[0]["source_initial_count"] == 2
    assert summary[0]["dmrg_source_mode"] == "unconverged_fallback"
    assert summary[0]["uhf_energy"] == -100.01
    assert summary[0]["ccsd_energy"] == -100.07
    assert summary[0]["ranking_method"] == "CCSD(T)"
    assert abs(summary[0]["ranking_energy"] - (-100.11)) < 1e-12


def test_generate_report_uses_explicit_bridging_labels():
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), coordination=4, label="Fe1"),
            MetalCenter(element="Fe", index=5, position=np.ones(3), coordination=4, label="Fe2"),
        ],
        bridging_atoms=[
            BridgingAtom(
                element="S",
                index=1,
                position=np.array([0.5, 0.5, 0.0]),
                bridged_metals=[0, 1],
                role="bridging",
                label="S1",
            )
        ],
        total_charge=-2,
        target_spin=0.0,
        symmetry_group="C1",
        formula="Fe2S1",
    )
    active_space = CAS(
        n_electrons=10,
        n_orbitals=10,
        n_qubits=20,
        description="test",
        level=ActiveSpaceLevel.STANDARD,
    )

    report = generate_report(
        cluster,
        active_space,
        spin_families=[],
        spin_isomers=[],
        n_electronic_configs=0,
        filtering_plan=None,
        results=[],
        extrapolated=[],
        output_format="markdown",
    )
    assert "Fe1, Fe2" in report
    assert "| 1 | S | bridging | Fe1, Fe2 |" in report
