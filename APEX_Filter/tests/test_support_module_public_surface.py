"""Regression coverage for intentionally tiny support-module surfaces."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _public_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def _public_top_level_defs_from_ast(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    }


def test_support_modules_keep_minimal_runtime_surfaces():
    expected = {
        "apex_filter.session": {"SessionManager"},
        "apex_filter.hdf5_state_io": set(),
        "apex_filter.pick": set(),
        "apex_filter.CAS_loader": set(),
        "apex_filter._case_artifacts": set(),
        "apex_filter._dmrg_summary": set(),
        "apex_filter.selection_guidance": set(),
        "apex_filter._step_selection_artifacts": set(),
        "apex_filter.dmrg_orbital_basis": set(),
        "apex_filter.dmrg_integral_transform": set(),
    }

    for module_name, public_names in expected.items():
        assert _public_callables(module_name) == public_names

    assert _public_top_level_defs_from_ast(
        "/Users/snh/Projects/APEX/APEX_Filter/apex_filter/reference_dmrg_worker.py"
    ) == set()
