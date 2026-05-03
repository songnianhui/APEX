"""Regression coverage for the canonical Step 1-10 method-controls template."""

from __future__ import annotations

from pathlib import Path

import yaml

from apex_filter.steps_dmrg import _DMRG_DEFAULTS
from apex_filter.steps_dmrg_basis import _DMRG_BASIS_DEFAULTS
from apex_filter.steps_enumeration import _ENUMERATE_DEFAULTS
from apex_filter.steps_reference_uhf import _UHF_DEFAULTS
from apex_filter.steps_ucc import _CCSDT_DEFAULTS, _CCSD_DEFAULTS, _CCSD_T_DEFAULTS


_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "config"
    / "method_controls_template.yaml"
)


def _load_template() -> dict:
    with _TEMPLATE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_method_controls_template_covers_step1_to_step10_runtime_defaults():
    template = _load_template()

    expected_sections = {
        "enumerate": _ENUMERATE_DEFAULTS,
        "uhf": _UHF_DEFAULTS,
        "ccsd": _CCSD_DEFAULTS,
        "ccsd_t": _CCSD_T_DEFAULTS,
        "ccsdt": _CCSDT_DEFAULTS,
        "dmrg_basis": _DMRG_BASIS_DEFAULTS,
        "dmrg": _DMRG_DEFAULTS,
    }

    for section_name, defaults in expected_sections.items():
        assert section_name in template, f"Missing template section: {section_name}"
        template_keys = set(template[section_name].keys())
        default_keys = set(defaults.keys())
        assert default_keys <= template_keys, (
            f"Template section '{section_name}' is missing runtime keys: "
            f"{sorted(default_keys - template_keys)}"
        )


def test_method_controls_template_may_preserve_post_step10_sections():
    template = _load_template()
    assert "fno_uccsdtq" in template
