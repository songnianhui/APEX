"""Acceptance-style checks for the committed Fe2S2 ox example bundle."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "examples" / "fe2s2"
SESSION_DIR = CASE_DIR / "filter_session"
CHAN_REF_DIR = CASE_DIR / "chan_ref"


def _load_records(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    return [data]


def test_fe2s2_step_summaries_expose_expected_benchmark_fields():
    step3 = _load_records(SESSION_DIR / "step3_uhf" / "uhf_summary.json")
    step6 = _load_records(SESSION_DIR / "step6_ccsdt" / "ccsdt_summary.json")
    step8 = _load_records(SESSION_DIR / "step8_dmrg" / "dmrg_summary.json")
    step9 = _load_records(SESSION_DIR / "step9_extrapolate" / "dmrg_extrapolation_summary.json")
    step10 = _load_records(SESSION_DIR / "step10_report" / "final_summary.json")

    assert len(step3) == 2
    assert all({"two_s", "two_sz_fe1", "two_sz_fe2", "final_state_signature"} <= set(row.keys()) for row in step3)

    assert len(step6) == 1
    assert all(row["observables_complete"] is True for row in step6)
    assert all(row["lambda_converged"] is True for row in step6)
    assert all({"two_s", "two_sz_fe1", "two_sz_fe2"} <= set(row.keys()) for row in step6)

    assert len(step8) == 10
    assert {row["bond_dim"] for row in step8} == {100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400}
    assert all("result_path" in row and "log_path" in row for row in step8)
    assert {row["label"] for row in step8} == {"Fe1↑Fe2↓|2xFe(III)|d:none"}
    step8_result_names = {Path(row["result_path"]).name for row in step8}
    assert all("_M" in name for name in step8_result_names)
    assert any(name.startswith("Fe1upFe2down_2xFe_III_d_none_") for name in step8_result_names)
    assert not any("6d50a843" in name or "857bec61" in name for name in step8_result_names)

    assert len(step9) == 1
    assert all(row["source_mode"] == "converged_only" for row in step9)
    assert all(row["n_points_used"] == 10 for row in step9)

    assert len(step10) == 2
    assert all({"consensus_energy", "dmrg_extrapolated_energy", "ccsdt_energy", "ranking_method"} <= set(row.keys()) for row in step10)
    ranking_by_label = {row["label"]: row["ranking_method"] for row in step10}
    assert ranking_by_label["Fe1↑Fe2↓|2xFe(III)|d:none"] == "CCSDT + DMRG consensus"
    assert ranking_by_label["Fe1↓Fe2↑|2xFe(III)|d:none"] == "CCSD(T)"


def test_fe2s2_final_report_exports_current_csv_artifacts():
    summary = json.loads((SESSION_DIR / "step10_report" / "final_summary.json").read_text())
    energies_csv = (SESSION_DIR / "step10_report" / "final_report_energies.csv").read_text(encoding="utf-8-sig")
    observables_csv = (SESSION_DIR / "step10_report" / "final_report_observables.csv").read_text(encoding="utf-8-sig")

    assert len(summary["records"]) == 2
    assert "CCSDT + DMRG consensus" in energies_csv
    assert "ranking_energy" in energies_csv
    assert "e_core" in energies_csv
    assert "uhf_s_squared" in observables_csv
    assert "ccsdt_two_sz_fe1" in observables_csv


def test_fe2s2_chan_ref_bundle_is_present_and_readable():
    required = [
        CHAN_REF_DIR / "README_benchmarks.md",
        CHAN_REF_DIR / "fe2s2_cas_settings_2014.yaml",
        CHAN_REF_DIR / "fe2s2_chan2026_oxidized_benchmark.json",
        CHAN_REF_DIR / "fe2s2_chan_benchmark_asset_index.md",
    ]
    for path in required:
        assert path.exists(), f"Missing benchmark artifact: {path}"

    benchmark = json.loads((CHAN_REF_DIR / "fe2s2_chan2026_oxidized_benchmark.json").read_text())
    assert "table5_ucc_series" in benchmark
    assert "table7_udmrg_series" in benchmark
    assert any(row.get("theory") == "UHF" for row in benchmark["table5_ucc_series"])
    assert any(row.get("theory") == "UCCSDT" for row in benchmark["table5_ucc_series"])
    assert len(benchmark["table7_udmrg_series"]) == 10
    asset_index = (CHAN_REF_DIR / "fe2s2_chan_benchmark_asset_index.md").read_text()
    assert "Table 5" in asset_index
    assert "Table 7" in asset_index
    assert "Chan" in asset_index
