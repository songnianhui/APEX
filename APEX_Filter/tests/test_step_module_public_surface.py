"""Regression coverage for the intentionally tiny staged-step module surface."""

from __future__ import annotations

import importlib


def _public_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def test_step_modules_only_expose_step_entrypoints():
    expected = {
        "apex_filter.steps_setup": {"step_load"},
        "apex_filter.steps_enumeration": {"step_enumerate"},
        "apex_filter.steps_reference_uhf": {"step_uhf"},
        "apex_filter.steps_ucc": {"step_ccsd", "step_ccsd_t", "step_ccsdt"},
        "apex_filter.steps_dmrg_basis": {"step_dmrg_basis"},
        "apex_filter.steps_dmrg": {"step_dmrg", "step_extrapolate_dmrg"},
        "apex_filter.steps_report": {"step_report"},
    }

    for module_name, public_names in expected.items():
        assert _public_callables(module_name) == public_names
