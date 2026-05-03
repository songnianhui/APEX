"""Regression coverage for the intentionally small reference-driver surface."""

from __future__ import annotations

import importlib


def _public_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def test_reference_modules_only_expose_method_drivers():
    expected = {
        "apex_filter.reference_uhf": {"converge_reference_uhf"},
        "apex_filter.reference_ucc": {"run_reference_ucc"},
        "apex_filter.reference_hast_ucc": {"run_reference_hast_ucc"},
        "apex_filter.reference_dmrg": {"run_reference_dmrg"},
    }

    for module_name, public_names in expected.items():
        assert _public_callables(module_name) == public_names
