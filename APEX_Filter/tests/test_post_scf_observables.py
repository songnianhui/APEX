"""Regression tests for post-SCF observable analysis entry points."""

from pathlib import Path

import pytest

from apex_filter.post_scf_observables import analyze_step3_uhf_observables
from shared.spin_metrics import compute_two_s_from_s2


def test_compute_two_s_from_s2_matches_chan_formula():
    assert compute_two_s_from_s2(4.89315) == pytest.approx(3.5357032977, abs=1e-6)


def test_fe2s2_step3_two_sz_matches_chan_uhf_reasonably():
    repo_root = Path(__file__).resolve().parents[2]
    case_dir = repo_root / "examples" / "fe2s2"
    result = analyze_step3_uhf_observables(
        step3_h5_path=str(
            case_dir
            / "filter_session"
            / "step3_uhf"
            / "results"
            / "Fe1↑Fe2↓_2xFe(III)_d:none_uhf.h5"
        ),
        cas_data_h5_path=str(
            case_dir
            / "outputs"
            / "orbitals"
            / "C4H12Fe2S6_uks_BP86_tzp-dkh_cas_data.h5"
        ),
        xyz_path=str(case_dir / "inputs" / "fe2s2.xyz"),
        cluster_info_path=str(case_dir / "inputs" / "fe2s2_cluster_info.yaml"),
        cas_settings_path=str(case_dir / "inputs" / "fe2s2_cas_settings.yaml"),
        chan_benchmark_json=str(
            case_dir / "chan_ref" / "fe2s2_chan2026_oxidized_benchmark.json"
        ),
    )

    assert result["chan_benchmark_comparison"]["best_alignment"] == "global_sign_flip"
    best = result["chan_benchmark_comparison"]["best_delta"]
    assert abs(best["Fe1"]) < 0.08
    assert abs(best["Fe2"]) < 0.08
    methods = result["two_sz_methods"]
    assert "meta_lowdin_atomic" in methods
    assert "meta_lowdin_fe_d" in methods
    primary_err = max(abs(v) for v in methods["ao_projected_fe_d"]["chan_benchmark_comparison"]["best_delta"].values())
    meta_atomic_err = max(abs(v) for v in methods["meta_lowdin_atomic"]["chan_benchmark_comparison"]["best_delta"].values())
    meta_d_err = max(abs(v) for v in methods["meta_lowdin_fe_d"]["chan_benchmark_comparison"]["best_delta"].values())
    assert primary_err < meta_atomic_err
    assert primary_err < meta_d_err
