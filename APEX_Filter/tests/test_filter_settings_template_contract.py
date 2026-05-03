"""Regression coverage for the canonical Step 1 bootstrap template."""

from __future__ import annotations

from pathlib import Path


_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "config"
    / "filter_settings_template.yaml"
)
def _load_template_text() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def test_filter_settings_template_exposes_canonical_step1_bootstrap_fields():
    template_text = _load_template_text()

    expected_keys = {
        "apex_cas_case_dir",
        "charge",
        "spin",
        "symmetry_group",
        "reduction_symmetry",
        "family_scheme",
        "benchmark_profile",
        "config_reduction_mode",
        "structure_path",
        "cluster_info_path",
        "fcidump_path",
        "fcidump_ecore_path",
        "preset",
        "scf_method",
        "xc_functional",
        "basis_set_default",
        "basis_per_element",
        "relativistic",
        "solvation_model",
        "solvation_epsilon",
        "conv_tol",
        "max_cycle",
    }

    missing = [key for key in sorted(expected_keys) if key not in template_text]
    assert not missing, (
        "filter_settings template is missing canonical Step 1 bootstrap keys: "
        f"{missing}"
    )


def test_filter_settings_template_keeps_method_controls_as_the_step2_plus_authority():
    template_text = _load_template_text()
    assert "method_controls.yaml" in template_text
    assert "Step 2 及之后" in template_text
    assert "enumerate / uhf / ccsd / ccsd-t / ccsdt / dmrg-basis / dmrg / fno-uccsdtq" in template_text
