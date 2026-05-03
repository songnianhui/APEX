"""Regression coverage for retained shared support-module public surfaces."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


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


def test_shared_support_modules_only_define_intended_public_top_level_names():
    expected = {
        REPO_ROOT / "shared" / "active_space_reference.py": {
            "build_fake_mol",
            "build_reference_uhf_solver",
        },
        REPO_ROOT / "shared" / "ao_basis.py": {"get_d_ao_indices"},
        REPO_ROOT / "shared" / "apex_cas_provenance.py": {
            "load_apex_cas_provenance",
            "build_effective_settings_from_apex_cas",
        },
        REPO_ROOT / "shared" / "artifact_paths.py": {
            "load_json_if_exists",
            "load_fcidump_summary",
            "candidate_structure_paths",
            "resolve_structure_path",
            "resolve_cluster_info_path",
            "auto_detect_fcidump",
            "resolve_fcidump_path",
        },
        REPO_ROOT / "shared" / "chem_knowledge.py": {
            "get_metals_db",
            "get_ligands_db",
            "get_cluster_templates_db",
            "get_local_spin",
            "get_common_oxidation_states",
            "get_d_electron_count",
            "get_n_active_orbitals",
            "get_valence_s_orbital",
            "match_cluster_template",
        },
        REPO_ROOT / "shared" / "dmrg_controls.py": {
            "build_dmrg_sweep_schedule",
            "compress_dmrg_schedule_for_dmrgci",
            "infer_pyblock2_benchmark_controls",
        },
        REPO_ROOT / "shared" / "element_data.py": {
            "get_atomic_number",
            "get_period",
            "build_electron_config",
            "get_electron_config",
            "get_valence_shells",
            "is_transition_metal",
            "is_bridging_element",
            "get_covalent_radius",
            "get_3d_metals",
            "get_4d_metals",
            "get_5d_metals",
            "get_metal_row",
            "get_default_d_label",
        },
        REPO_ROOT / "shared" / "fcidump_io.py": {"FCIDUMPData", "load_fcidump"},
        REPO_ROOT / "shared" / "formatting.py": {
            "format_energy",
            "shell_safe_artifact_token",
        },
        REPO_ROOT / "shared" / "knowledge_base.py": {"data_file"},
        REPO_ROOT / "shared" / "roman.py": {"to_roman"},
        REPO_ROOT / "shared" / "setting_utils.py": {
            "load_basis_file",
            "apply_overrides",
            "load_cas_settings_file",
            "settings_from_preset",
            "build_basis_dict",
        },
    }

    for path, public_names in expected.items():
        assert _public_top_level_defs(path) == public_names


def test_shared_support_runtime_surfaces_stay_minimal():
    expected = {
        "shared.active_space_reference": {
            "build_fake_mol",
            "build_reference_uhf_solver",
        },
        "shared.ao_basis": {"get_d_ao_indices"},
        "shared.apex_cas_provenance": {
            "load_apex_cas_provenance",
            "build_effective_settings_from_apex_cas",
        },
        "shared.artifact_paths": {
            "load_json_if_exists",
            "load_fcidump_summary",
            "candidate_structure_paths",
            "resolve_structure_path",
            "resolve_cluster_info_path",
            "auto_detect_fcidump",
            "resolve_fcidump_path",
        },
        "shared.chem_knowledge": {
            "get_metals_db",
            "get_ligands_db",
            "get_cluster_templates_db",
            "get_local_spin",
            "get_common_oxidation_states",
            "get_d_electron_count",
            "get_n_active_orbitals",
            "get_valence_s_orbital",
            "match_cluster_template",
        },
        "shared.dmrg_controls": {
            "build_dmrg_sweep_schedule",
            "compress_dmrg_schedule_for_dmrgci",
            "infer_pyblock2_benchmark_controls",
        },
        "shared.element_data": {
            "get_atomic_number",
            "get_period",
            "build_electron_config",
            "get_electron_config",
            "get_valence_shells",
            "is_transition_metal",
            "is_bridging_element",
            "get_covalent_radius",
            "get_3d_metals",
            "get_4d_metals",
            "get_5d_metals",
            "get_metal_row",
            "get_default_d_label",
        },
        "shared.fcidump_io": {"FCIDUMPData", "load_fcidump"},
        "shared.formatting": {
            "format_energy",
            "shell_safe_artifact_token",
        },
        "shared.knowledge_base": {"data_file"},
        "shared.roman": {"to_roman"},
        "shared.setting_utils": {
            "load_basis_file",
            "apply_overrides",
            "load_cas_settings_file",
            "settings_from_preset",
            "build_basis_dict",
        },
    }

    for module_name, public_names in expected.items():
        assert _public_runtime_callables(module_name) == public_names
