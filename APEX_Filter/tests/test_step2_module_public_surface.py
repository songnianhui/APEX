"""Regression coverage for the intentionally tiny Step 2 utility surface."""

from __future__ import annotations

import importlib


def _public_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def test_step2_module_only_exposes_orchestration_entrypoints():
    assert _public_callables("apex_filter.elec_spin_config_generator") == {
        "generate_all_configs",
        "canonicalize_config_spin_labels",
    }
