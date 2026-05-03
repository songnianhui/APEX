"""Regression tests for Step 10 reporting and internal final-summary persistence."""

import csv
import os

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
    sm.save_step_summary("step3_uhf", "uhf_summary.json", 
        [
            {"label": "BS7_235", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS7"},
            {"label": "BS8_237", "energy": -99.90, "converged": True, "s_squared": 1.6, "family": "BS8"},
            {"label": "BS8_236", "energy": -99.85, "converged": True, "s_squared": 1.7, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step4_ccsd", "ccsd_summary.json", 
        [
            {"label": "BS7_235", "energy": -100.08, "correlation_energy": -0.08, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.01, "correlation_energy": -0.11, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step5_ccsd_t", "ccsd_t_summary.json", 
        [
            {"label": "BS7_235", "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2", "energy": -100.10, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.05, "converged": True, "family": "BS8"},
            {"label": "BS8_236", "energy": -99.95, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step6_ccsdt", "ccsdt_summary.json", 
        [
            {"label": "BS7_235", "energy": -100.20, "converged": True, "family": "BS7"},
            {"label": "BS8_237", "energy": -100.10, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step9_extrapolate", "dmrg_extrapolation_summary.json", 
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

    summary = sm._load_final_summary()
    assert [row["label"] for row in summary] == ["BS7_235", "BS8_236", "BS8_237"]
    assert summary[0]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"
    assert summary[0]["representative_label"] == "BS7_235"
    assert summary[0]["source_initial_labels"] == ["BS7_235"]
    assert summary[0]["uhf_energy"] == -100.00
    assert summary[0]["ccsd_energy"] == -100.08
    assert summary[0]["ranking_method"] == "CCSDT + DMRG consensus"
    assert abs(summary[0]["consensus_energy"] - (-100.22)) < 1e-12
    assert abs(summary[0]["consensus_uncertainty"] - 0.02) < 1e-12
    assert summary[1]["ranking_method"] == "DMRG extrapolated"
    assert summary[2]["ranking_method"] == "CCSDT"

    energies_csv = os.path.join(sm.session_dir, "step10_report", "final_report_energies.csv")
    observables_csv = os.path.join(sm.session_dir, "step10_report", "final_report_observables.csv")
    assert os.path.exists(energies_csv)
    assert os.path.exists(observables_csv)
    assert not os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report.md"))
    assert not os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report.csv"))
    assert not os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report_ranking.csv"))
    assert not os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report_dmrg.csv"))

    with open(energies_csv, newline="", encoding="utf-8-sig") as f:
        energy_rows = list(csv.DictReader(f))
    assert energy_rows[0]["metric"] == "rank"
    assert energy_rows[0]["description"] == "Final ranking position"
    state_columns = [name for name in energy_rows[0].keys() if name not in {"metric", "description"}]
    assert state_columns == [
        "rank1:Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
        "rank2:BS8_236",
        "rank3:BS8_237",
    ]
    energies_by_metric = {row["metric"]: row for row in energy_rows}
    assert energies_by_metric["representative_label"][state_columns[0]] == "BS7_235"
    assert energies_by_metric["ranking_method"][state_columns[0]] == "CCSDT + DMRG consensus"
    assert energies_by_metric["source_initial_labels"][state_columns[0]] == "BS7_235"
    assert energies_by_metric["e_core"][state_columns[0]] == "N/A"
    assert energies_by_metric["ccsd_energy"][state_columns[0]] == "-100.08"
    assert energies_by_metric["ranking_method"][state_columns[1]] == "DMRG extrapolated"
    assert energies_by_metric["ranking_method"][state_columns[2]] == "CCSDT"

    with open(observables_csv, newline="", encoding="utf-8-sig") as f:
        observables_rows = list(csv.DictReader(f))
    observables_by_metric = {row["metric"]: row for row in observables_rows}
    assert observables_by_metric["uhf_s_squared"][state_columns[0]] == "1.5"

    step10_dir = os.path.join(sm.session_dir, "step10_report")
    assert not os.path.exists(os.path.join(step10_dir, "selection_guide.md"))
    assert not os.path.exists(os.path.join(step10_dir, "selection_candidates.csv"))
    assert not os.path.exists(os.path.join(step10_dir, "selection_worklist.csv"))
    assert not os.path.exists(os.path.join(step10_dir, "pick_labels_all.json"))
    assert not os.path.exists(os.path.join(step10_dir, "pick_labels_template.json"))


def test_step_report_prefers_cc_composite_when_available(tmp_path):
    sm = _seed_session_for_report(tmp_path)
    sm.save_step_summary("step3_uhf", "uhf_summary.json", [{"label": "BS7_235", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS7"}])
    sm.save_step_summary("step4_ccsd", "ccsd_summary.json", [{"label": "BS7_235", "energy": -100.08, "correlation_energy": -0.08, "converged": True, "family": "BS7"}])
    sm.save_step_summary("step5_ccsd_t", "ccsd_t_summary.json", [{"label": "BS7_235", "energy": -100.10, "converged": True, "family": "BS7"}])
    sm.save_step_summary("step6_ccsdt", "ccsdt_summary.json", [{"label": "BS7_235", "energy": -100.20, "converged": True, "family": "BS7"}])
    sm.save_step_summary("step9_extrapolate", "dmrg_extrapolation_summary.json", [{"label": "BS7_235", "energy": -100.24, "uncertainty": 5e-4, "bond_dims": [5000, 10000], "family": "BS7"}])
    sm.save_cc_composite_summary([{"label": "BS7_235", "family": "BS7", "freeze_occ": 2, "energy": -100.30, "uncertainty": 0.01, "converged": True}])

    step_report(sm.session_dir)

    summary = sm._load_final_summary()
    assert len(summary) == 1
    assert summary[0]["cc_composite_energy"] == -100.30
    assert summary[0]["ranking_method"] == "CC composite + DMRG consensus"
    assert abs(summary[0]["ranking_energy"] - (-100.27)) < 1e-12

    energies_csv = os.path.join(sm.session_dir, "step10_report", "final_report_energies.csv")
    with open(energies_csv, newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    by_metric = {row["metric"]: row for row in csv_rows}
    state_column = next(name for name in csv_rows[0].keys() if name not in {"metric", "description"})
    assert by_metric["cc_composite_energy"][state_column] == "-100.3"
    assert by_metric["ranking_method"][state_column] == "CC composite + DMRG consensus"
    assert os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report_energies.csv"))


def test_step_report_groups_by_final_state_and_preserves_provenance(tmp_path):
    sm = _seed_session_for_report(tmp_path)
    sm.save_step_summary("step3_uhf", "uhf_summary.json", 
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.00, "converged": True, "s_squared": 1.5, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.01, "converged": True, "s_squared": 1.6, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step4_ccsd", "ccsd_summary.json", 
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.06, "correlation_energy": -0.06, "converged": True, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.07, "correlation_energy": -0.07, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step5_ccsd_t", "ccsd_t_summary.json", 
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.10, "converged": True, "family": "BS8"},
            {"label": "guess_d2", "display_label": "Fe2:dz^2", "energy": -100.11, "converged": True, "family": "BS8"},
        ]
    )
    sm.save_step_summary("step6_ccsdt", "ccsdt_summary.json", [])
    sm.save_step_summary("step9_extrapolate", "dmrg_extrapolation_summary.json", 
        [
            {"label": "guess_d1", "display_label": "Fe2:dz^2", "energy": -100.30, "uncertainty": float("inf"), "bond_dims": [500, 1000, 1500], "family": "BS8", "source_mode": "unconverged_fallback"},
        ]
    )

    step_report(sm.session_dir)

    summary = sm._load_final_summary()
    assert len(summary) == 1
    assert summary[0]["display_label"] == "Fe2:dz^2"
    assert summary[0]["source_initial_labels"] == ["guess_d1", "guess_d2"]
    assert summary[0]["source_initial_count"] == 2
    assert summary[0]["dmrg_source_mode"] == "unconverged_fallback"
    assert summary[0]["uhf_energy"] == -100.01
    assert summary[0]["ccsd_energy"] == -100.07
    assert summary[0]["ranking_method"] == "CCSD(T)"
    assert abs(summary[0]["ranking_energy"] - (-100.11)) < 1e-12

    observables_csv = os.path.join(sm.session_dir, "step10_report", "final_report_observables.csv")
    with open(observables_csv, newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    by_metric = {row["metric"]: row for row in csv_rows}
    state_column = next(name for name in csv_rows[0].keys() if name not in {"metric", "description"})
    assert by_metric["representative_label"][state_column] == "guess_d2"
    assert os.path.exists(os.path.join(sm.session_dir, "step10_report", "final_report_observables.csv"))
