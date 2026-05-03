"""Regression coverage for the intentionally small analysis-module surface."""

from __future__ import annotations

import importlib


def _public_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def test_post_scf_module_only_exposes_analysis_entrypoints():
    assert _public_callables("apex_filter.post_scf_observables") == {
        "analyze_active_space_spin_observables",
        "analyze_step3_uhf_observables",
    }
