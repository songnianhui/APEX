"""Regression coverage for the intentionally small APEX_CAS module surface."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import apex_cas.FCIDUMP_generator as fcidump_generator


REPO_ROOT = Path("/Users/snh/Projects/APEX")


def _public_top_level_defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def _public_runtime_callables(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }


def test_apex_cas_modules_only_define_intended_public_top_level_names():
    expected = {
        REPO_ROOT / "APEX_CAS" / "apex_cas" / "main.py": {"create_parser", "main"},
        REPO_ROOT / "APEX_CAS" / "apex_cas" / "CAS_builder.py": {
            "build_cas_from_mean_field",
            "run_scf_initialization",
        },
        REPO_ROOT / "APEX_CAS" / "apex_cas" / "state_io.py": {"load_cas_state"},
        REPO_ROOT / "APEX_CAS" / "apex_cas" / "orbital_visualizer.py": {"plot_orbitals"},
        REPO_ROOT / "APEX_CAS" / "apex_cas" / "FCIDUMP_generator.py": {"compare_fcidumps"},
    }

    for path, public_names in expected.items():
        assert _public_top_level_defs(path) == public_names


def test_fcidump_generator_public_surface_keeps_compare_alias():
    assert callable(fcidump_generator.compare_fcidumps)


def test_apex_cas_runtime_surfaces_stay_minimal():
    expected = {
        "apex_cas.main": {"create_parser", "main"},
        "apex_cas.CAS_builder": {"build_cas_from_mean_field", "run_scf_initialization"},
        "apex_cas.state_io": {"load_cas_state"},
        "apex_cas.orbital_visualizer": {"plot_orbitals"},
        "apex_cas.FCIDUMP_generator": {"compare_fcidumps"},
        "apex_cas.selection_io": set(),
        "apex_cas.CAS_tester": set(),
        "apex_cas.prepare": set(),
        "apex_cas.CAS_quality": set(),
        "apex_cas.ao_shell_analysis": set(),
    }

    for module_name, public_names in expected.items():
        assert _public_runtime_callables(module_name) == public_names
