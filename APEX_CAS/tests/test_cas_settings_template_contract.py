"""Regression coverage for the canonical APEX_CAS settings template."""

from __future__ import annotations

from pathlib import Path


_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "config"
    / "cas_settings_template.yaml"
)


def _load_template_text() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def test_cas_settings_template_exposes_current_release_controls():
    text = _load_template_text()
    required_keys = [
        "preset:",
        "charge:",
        "spin:",
        "scf_spin:",
        "basis_set_default:",
        "basis_per_element:",
        "basis_set_file:",
        "relativistic:",
        "solvation_model:",
        "conv_tol:",
        "max_cycle:",
        "scf_stage1_rough:",
        "scf_stage3_newton:",
        "newton_max_cycle:",
        "newton_conv_tol:",
        "scf_allow_unconverged:",
        "localization_method:",
        "pm_conv_tol_grad:",
        "boys_conv_tol:",
        "boys_conv_tol_grad:",
        "boys_max_cycle:",
        "cpt_cas_type:",
        "projection_threshold:",
        "avas_config:",
        "generate_cubes:",
        "cube_grid:",
        "pw_plot_threshold:",
        "render_png:",
        "png_isovalue:",
        "cluster_info_path:",
        "family_scheme:",
        "benchmark_profile:",
        "config_reduction_mode:",
    ]

    missing = [key for key in required_keys if key not in text]
    assert not missing, f"CAS settings template is missing release controls: {missing}"


def test_cas_settings_template_documents_requested_vs_effective_settings_split():
    text = _load_template_text()
    for marker in (
        "requested_config",
        "effective_method",
        "effective_parameters",
        "results",
    ):
        assert marker in text
