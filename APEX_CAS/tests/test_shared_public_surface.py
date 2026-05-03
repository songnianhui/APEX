"""Regression coverage for the intentional shared Step 1-10 authority surface."""

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


def test_shared_modules_only_define_intended_public_top_level_names():
    expected = {
        REPO_ROOT / "shared" / "comparison.py": {
            "compare_artifacts",
            "compare_matrix_entries",
            "compare_matrix_spectra",
            "compare_density_matrices",
            "compare_basis_states",
            "compare_two_particle_density_tensors",
            "find_chan_benchmark_row",
            "compare_two_sz_with_benchmark",
            "compare_energy_triplet",
            "compare_fcidumps",
        },
        REPO_ROOT / "shared" / "models.py": {
            "MetalCenter",
            "BridgingAtom",
            "TerminalLigand",
            "ClusterInfo",
            "ActiveSpaceLevel",
            "OrbitalGroup",
            "CAS",
            "AVASConfig",
            "ActiveSpaceQuality",
            "SpinIsomer",
            "SpinIsomerFamily",
            "OxidationAssignment",
            "ElectronicConfig",
            "CalculationResult",
            "ExtrapolatedEnergy",
            "ComputationSettings",
        },
        REPO_ROOT / "shared" / "cluster_info_labels.py": {
            "is_authoritative_cluster_info",
            "require_authoritative_cluster_info",
            "resolve_explicit_label",
            "resolve_metal_site_label",
        },
        REPO_ROOT / "shared" / "cluster_info_io.py": {
            "load_cluster_info_yaml",
            "resolve_cluster_metadata",
        },
        REPO_ROOT / "shared" / "settings_payloads.py": {
            "build_base_settings_payload",
            "extend_settings_payload",
            "build_effective_localization_payload",
            "build_effective_selection_payload",
            "build_requested_cas_payload",
            "build_effective_parameter_payload",
            "normalize_settings_payload",
            "find_effective_parameter_leaks",
            "missing_normalized_settings_sections",
        },
        REPO_ROOT / "shared" / "selection_io.py": {"load_active_selection"},
        REPO_ROOT / "shared" / "dmrg_solvers.py": {
            "resolve_dmrgci_twodot_to_onedot",
            "run_sz_dmrg",
            "run_su2_dmrg",
        },
        REPO_ROOT / "shared" / "molecule_builder.py": {"build_mol_with_basis"},
        REPO_ROOT / "shared" / "spin_metrics.py": {"compute_two_s_from_s2"},
        REPO_ROOT / "shared" / "reference_states.py": {
            "load_reference_state_payload",
            "load_reference_mf_from_npz",
        },
        REPO_ROOT / "shared" / "structure_parser.py": {
            "parse_structure",
            "analyze_symmetry_report",
            "format_symmetry_report",
            "symmetry_report_json",
        },
        REPO_ROOT / "shared" / "final_state_signatures.py": {
            "parse_orbital_metal_mapping",
            "summarize_final_state_from_dm",
        },
        REPO_ROOT / "shared" / "chkfiles.py": {"find_chkfile"},
    }

    for path, public_names in expected.items():
        assert _public_top_level_defs(path) == public_names


def test_shared_runtime_surfaces_stay_minimal():
    expected = {
        "shared.comparison": {
            "compare_artifacts",
            "compare_matrix_entries",
            "compare_matrix_spectra",
            "compare_density_matrices",
            "compare_basis_states",
            "compare_two_particle_density_tensors",
            "find_chan_benchmark_row",
            "compare_two_sz_with_benchmark",
            "compare_energy_triplet",
            "compare_fcidumps",
        },
        "shared.models": {
            "MetalCenter",
            "BridgingAtom",
            "TerminalLigand",
            "ClusterInfo",
            "ActiveSpaceLevel",
            "OrbitalGroup",
            "CAS",
            "AVASConfig",
            "ActiveSpaceQuality",
            "SpinIsomer",
            "SpinIsomerFamily",
            "OxidationAssignment",
            "ElectronicConfig",
            "CalculationResult",
            "ExtrapolatedEnergy",
            "ComputationSettings",
        },
        "shared.cluster_info_labels": {
            "is_authoritative_cluster_info",
            "require_authoritative_cluster_info",
            "resolve_explicit_label",
            "resolve_metal_site_label",
        },
        "shared.cluster_info_io": {
            "load_cluster_info_yaml",
            "resolve_cluster_metadata",
        },
        "shared.settings_payloads": {
            "build_base_settings_payload",
            "extend_settings_payload",
            "build_effective_localization_payload",
            "build_effective_selection_payload",
            "build_requested_cas_payload",
            "build_effective_parameter_payload",
            "normalize_settings_payload",
            "find_effective_parameter_leaks",
            "missing_normalized_settings_sections",
        },
        "shared.selection_io": {"load_active_selection"},
        "shared.dmrg_solvers": {
            "resolve_dmrgci_twodot_to_onedot",
            "run_sz_dmrg",
            "run_su2_dmrg",
        },
        "shared.molecule_builder": {"build_mol_with_basis"},
        "shared.spin_metrics": {"compute_two_s_from_s2"},
        "shared.reference_states": {
            "load_reference_state_payload",
            "load_reference_mf_from_npz",
        },
        "shared.structure_parser": {
            "parse_structure",
            "analyze_symmetry_report",
            "format_symmetry_report",
            "symmetry_report_json",
        },
        "shared.final_state_signatures": {
            "parse_orbital_metal_mapping",
            "summarize_final_state_from_dm",
        },
        "shared.chkfiles": {"find_chkfile"},
    }

    for module_name, public_names in expected.items():
        assert _public_runtime_callables(module_name) == public_names
