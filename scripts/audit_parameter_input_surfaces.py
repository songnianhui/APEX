#!/usr/bin/env python3
"""Audit canonical V1 parameter input surfaces.

This script intentionally scopes itself to the V1 mainline only:

- APEX_CAS: prepare -> scf -> buildcas -> fcidump -> testcas
- APEX_Filter: load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt
  -> dmrg-basis -> dmrg -> extrapolate -> report

It does not attempt to prove every sink dynamically at runtime. Instead it
checks the maintained public input surfaces against the current parsers,
templates, and section defaults so that we can detect:

- template keys with no classification
- CLI inputs that disappeared or drifted
- method-controls template/runtime key mismatches
- accidental expansion of the V1 scope
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_import_paths() -> None:
    """Allow the audit script to run directly from the repo root."""
    for path in (
        REPO_ROOT,
        REPO_ROOT / "APEX_CAS",
        REPO_ROOT / "APEX_Filter",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_import_paths()

from apex_cas.main import create_parser as create_apex_cas_parser  # noqa: E402
from apex_filter.main import create_parser as create_apex_filter_parser  # noqa: E402
from apex_filter.steps_dmrg import _DMRG_DEFAULTS  # noqa: E402
from apex_filter.steps_dmrg_basis import _DMRG_BASIS_DEFAULTS  # noqa: E402
from apex_filter.steps_enumeration import _ENUMERATE_DEFAULTS  # noqa: E402
from apex_filter.steps_reference_uhf import _UHF_DEFAULTS  # noqa: E402
from apex_filter.steps_ucc import _CCSDT_DEFAULTS, _CCSD_DEFAULTS, _CCSD_T_DEFAULTS  # noqa: E402


CAS_TEMPLATE = REPO_ROOT / "shared" / "config" / "cas_settings_template.yaml"
FILTER_TEMPLATE = REPO_ROOT / "shared" / "config" / "filter_settings_template.yaml"
METHOD_TEMPLATE = REPO_ROOT / "shared" / "config" / "method_controls_template.yaml"


CAS_TEMPLATE_KEYS = {
    "charge": {"category": "consumed_runtime", "sink": "APEX_CAS cluster charge"},
    "spin": {"category": "consumed_runtime", "sink": "APEX_CAS target total spin"},
    "scf_spin": {"category": "consumed_runtime", "sink": "APEX_CAS high-spin SCF reference"},
    "preset": {"category": "consumed_runtime", "sink": "ComputationSettings preset baseline"},
    "scf_method": {"category": "consumed_runtime", "sink": "SCF builder method identity"},
    "xc_functional": {"category": "consumed_runtime", "sink": "SCF builder XC functional"},
    "basis_set_default": {"category": "consumed_runtime", "sink": "SCF / AO basis default"},
    "basis_per_element": {"category": "consumed_runtime", "sink": "SCF / AO basis overrides"},
    "basis_set_file": {"category": "consumed_runtime", "sink": "SCF / AO basis overrides from file"},
    "relativistic": {"category": "consumed_runtime", "sink": "SCF relativistic treatment"},
    "solvation_model": {"category": "consumed_runtime", "sink": "SCF solvent model"},
    "solvation_epsilon": {"category": "consumed_runtime", "sink": "ddCOSMO dielectric"},
    "conv_tol": {"category": "consumed_runtime", "sink": "SCF convergence"},
    "max_cycle": {"category": "consumed_runtime", "sink": "SCF max cycles"},
    "scf_verbose": {"category": "consumed_runtime", "sink": "SCF verbosity"},
    "init_guess": {"category": "consumed_runtime", "sink": "SCF initial guess"},
    "scf_damp": {"category": "consumed_runtime", "sink": "SCF damping"},
    "scf_level_shift": {"category": "consumed_runtime", "sink": "SCF level shift"},
    "diis_space": {"category": "consumed_runtime", "sink": "SCF DIIS size"},
    "density_fit": {"category": "consumed_runtime", "sink": "SCF DF enable"},
    "density_fit_auxbasis": {"category": "consumed_runtime", "sink": "SCF DF auxiliary basis"},
    "density_fit_only_dfj": {"category": "consumed_runtime", "sink": "SCF DFJ-only"},
    "grids_level": {"category": "consumed_runtime", "sink": "DFT grid level"},
    "grids_small_rho_cutoff": {"category": "consumed_runtime", "sink": "DFT grid pruning cutoff"},
    "grids_prune": {"category": "consumed_runtime", "sink": "DFT grid pruning mode"},
    "frac_occ": {"category": "consumed_runtime", "sink": "SCF fractional occupations"},
    "smearing_method": {"category": "consumed_runtime", "sink": "SCF smearing mode"},
    "smearing_sigma": {"category": "consumed_runtime", "sink": "SCF smearing width"},
    "scf_stage1_rough": {"category": "consumed_runtime", "sink": "staged SCF rough pass"},
    "scf_stage3_newton": {"category": "consumed_runtime", "sink": "Newton refinement enable"},
    "newton_max_cycle": {"category": "consumed_runtime", "sink": "Newton max cycles"},
    "newton_conv_tol": {"category": "consumed_runtime", "sink": "Newton convergence"},
    "scf_allow_unconverged": {"category": "consumed_runtime", "sink": "allow unconverged SCF continuation"},
    "localization_method": {"category": "consumed_runtime", "sink": "CAS build route localization choice"},
    "pm_pop_method": {"category": "consumed_runtime", "sink": "PM localization population backend"},
    "pm_conv_tol": {"category": "consumed_runtime", "sink": "PM localization convergence"},
    "pm_conv_tol_grad": {"category": "consumed_runtime", "sink": "PM localization gradient convergence"},
    "pm_max_cycle": {"category": "consumed_runtime", "sink": "PM localization max cycles"},
    "pm_exponent": {"category": "consumed_runtime", "sink": "PM localization exponent"},
    "pm_init_guess": {"category": "consumed_runtime", "sink": "PM localization initial guess"},
    "boys_conv_tol": {"category": "consumed_runtime", "sink": "Boys localization convergence"},
    "boys_conv_tol_grad": {"category": "consumed_runtime", "sink": "Boys localization gradient convergence"},
    "boys_max_cycle": {"category": "consumed_runtime", "sink": "Boys localization max cycles"},
    "cpt_cas_type": {"category": "consumed_runtime", "sink": "CAS route selection"},
    "projection_threshold": {"category": "consumed_runtime", "sink": "projection-based selection threshold"},
    "avas_config": {"category": "consumed_runtime", "sink": "AVAS route config"},
    "generate_cubes": {"category": "consumed_runtime", "sink": "buildcas visualization generation"},
    "cube_grid": {"category": "consumed_runtime", "sink": "cube-grid resolution"},
    "pw_plot_threshold": {"category": "consumed_runtime", "sink": "reportable cube threshold"},
    "render_png": {"category": "consumed_runtime", "sink": "gallery/html bundle generation"},
    "png_isovalue": {"category": "consumed_runtime", "sink": "gallery isovalue"},
    "cluster_info_path": {"category": "bootstrap_only", "sink": "authoritative cluster_info.yaml discovery"},
    "symmetry_group": {"category": "bootstrap_only", "sink": "cluster metadata override"},
    "symmetry_group_override": {"category": "bootstrap_only", "sink": "cluster metadata override"},
    "reduction_symmetry": {"category": "bootstrap_only", "sink": "downstream symmetry metadata"},
    "reduction_symmetry_override": {"category": "bootstrap_only", "sink": "downstream symmetry metadata"},
    "symmetry_detection_mode": {"category": "bootstrap_only", "sink": "structure parser mode"},
    "symmetry_mode": {"category": "bootstrap_only", "sink": "structure parser mode"},
    "family_scheme": {"category": "bootstrap_only", "sink": "cluster benchmark metadata"},
    "benchmark_profile": {"category": "bootstrap_only", "sink": "cluster benchmark metadata"},
    "config_reduction_mode": {"category": "bootstrap_only", "sink": "cluster benchmark metadata"},
}


FILTER_TEMPLATE_KEYS = {
    "apex_cas_case_dir": {"category": "bootstrap_only", "sink": "Step 1 case discovery"},
    "charge": {"category": "bootstrap_only", "sink": "ClusterInfo reconstruction override"},
    "spin": {"category": "bootstrap_only", "sink": "ClusterInfo reconstruction override"},
    "symmetry_group": {"category": "bootstrap_only", "sink": "ClusterInfo metadata override"},
    "reduction_symmetry": {"category": "bootstrap_only", "sink": "ClusterInfo metadata override"},
    "family_scheme": {"category": "bootstrap_only", "sink": "ClusterInfo metadata override"},
    "benchmark_profile": {"category": "bootstrap_only", "sink": "ClusterInfo metadata override"},
    "config_reduction_mode": {"category": "bootstrap_only", "sink": "ClusterInfo metadata override"},
    "structure_path": {"category": "bootstrap_only", "sink": "authoritative structure rediscovery"},
    "cluster_info_path": {"category": "bootstrap_only", "sink": "authoritative cluster-info rediscovery"},
    "fcidump_path": {"category": "bootstrap_only", "sink": "FCIDUMP discovery"},
    "fcidump_ecore_path": {"category": "record_only", "sink": "APEX_CAS provenance bookkeeping"},
    "preset": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "scf_method": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "xc_functional": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "basis_set_default": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "basis_per_element": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "relativistic": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "solvation_model": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "solvation_epsilon": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "conv_tol": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
    "max_cycle": {"category": "record_only", "sink": "downstream APEX_CAS provenance override"},
}


CAS_CLI_SURFACES = {
    "prepare": {"structure", "case_dir", "cas_settings", "charge", "total_spin", "scf_spin", "symmetry_group", "reduction_symmetry", "symmetry_mode", "finalize", "draft_csv", "force"},
    "scf": {"structure", "case_dir", "cas_settings", "charge", "total_spin", "scf_spin", "symmetry_group", "reduction_symmetry", "symmetry_mode"},
    "buildcas": {"structure", "case_dir", "cas_settings", "charge", "total_spin", "scf_spin", "symmetry_group", "reduction_symmetry", "symmetry_mode", "no_cubes", "cube_grid"},
    "compute": {"structure", "case_dir", "cas_settings", "charge", "total_spin", "scf_spin", "symmetry_group", "reduction_symmetry", "symmetry_mode", "no_cubes", "cube_grid"},
    "fcidump": {"case_dir", "spin_projection", "output", "active_file", "zero_ecore"},
    "testcas": {"fcidump_path", "bond_dim", "symm", "output_dir", "label", "dmrg_mode", "n_sweeps", "max_iter", "stack_mem_gb"},
}


FILTER_CLI_SURFACES = {
    "load": {"config", "session"},
    "enumerate": {"session", "target_sz", "forced_oxidation", "max_configs"},
    "uhf": {"session", "pick", "conv_tol", "max_cycle", "stabilize_cycles", "level_shift", "damp", "newton_refine", "newton_max_cycle"},
    "ccsd": {"session", "pick", "code", "basis_set"},
    "ccsd-t": {"session", "pick", "code", "basis_set", "n_final"},
    "ccsdt": {"session", "pick", "code", "basis_set", "n_final", "conv_tol", "residual_tol", "max_cycle", "lambda_max_cycle", "diis_space", "diis_start_cycle", "iterative_damping", "level_shift", "newton_krylov"},
    "dmrg-basis": {"session", "pick", "localization_method", "cc_conv_tol", "cc_max_cycle", "cc_diis_space", "cc_direct", "pm_pop_method", "pm_conv_tol", "pm_conv_tol_grad", "pm_max_cycle", "pm_exponent", "pm_init_guess", "boys_conv_tol", "boys_conv_tol_grad", "boys_max_cycle", "ordering_matrix_mode", "exchange_proxy_max_orbitals", "ga_generations", "ga_population", "ga_mutation_rate", "ga_seed"},
    "dmrg": {"session", "pick", "backend", "basis_mode", "schedule_mode", "bond_dims", "n_sweeps", "convergence_tol", "n_threads", "stack_mem", "twosite_to_onesite", "dav_max_iter", "dav_def_max_size", "dav_rel_conv_thrd", "dav_type"},
    "extrapolate": {"session"},
    "report": {"session"},
    "fno-uccsdtq": {"session", "pick", "freeze_occ"},
    "cc-composite": {"session"},
}


METHOD_CONTROL_EXPECTED = {
    "enumerate": _ENUMERATE_DEFAULTS,
    "uhf": _UHF_DEFAULTS,
    "ccsd": _CCSD_DEFAULTS,
    "ccsd_t": _CCSD_T_DEFAULTS,
    "ccsdt": _CCSDT_DEFAULTS,
    "dmrg_basis": _DMRG_BASIS_DEFAULTS,
    "dmrg": _DMRG_DEFAULTS,
}


def _collect_subparser_dests(parser: argparse.ArgumentParser) -> dict[str, set[str]]:
    surfaces: dict[str, set[str]] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                dests = {
                    sub_action.dest
                    for sub_action in subparser._actions
                    if sub_action.dest not in {"help", argparse.SUPPRESS}
                }
                surfaces[name] = dests
    return surfaces


def _missing_template_keys(template_text: str, expected: dict[str, dict[str, str]]) -> list[str]:
    return [key for key in sorted(expected) if key not in template_text]


def main() -> int:
    issues: list[str] = []
    checked = {
        "cas_template_keys": len(CAS_TEMPLATE_KEYS),
        "filter_template_keys": len(FILTER_TEMPLATE_KEYS),
        "method_control_sections": len(METHOD_CONTROL_EXPECTED),
        "cas_cli_commands": len(CAS_CLI_SURFACES),
        "filter_cli_commands": len(FILTER_CLI_SURFACES),
    }

    cas_text = CAS_TEMPLATE.read_text(encoding="utf-8")
    filter_text = FILTER_TEMPLATE.read_text(encoding="utf-8")
    method_template = yaml.safe_load(METHOD_TEMPLATE.read_text(encoding="utf-8"))

    missing = _missing_template_keys(cas_text, CAS_TEMPLATE_KEYS)
    if missing:
        issues.append(f"cas_settings template missing classified keys: {missing}")

    missing = _missing_template_keys(filter_text, FILTER_TEMPLATE_KEYS)
    if missing:
        issues.append(f"filter_settings template missing classified keys: {missing}")

    for section, defaults in METHOD_CONTROL_EXPECTED.items():
        if section not in method_template:
            issues.append(f"method_controls template missing section: {section}")
            continue
        template_keys = set(method_template[section].keys())
        default_keys = set(defaults.keys())
        missing_keys = sorted(default_keys - template_keys)
        if missing_keys:
            issues.append(
                f"method_controls template section '{section}' missing runtime keys: {missing_keys}"
            )

    if "fno_uccsdtq" not in method_template:
        issues.append("method_controls template missing out-of-scope fno_uccsdtq section")

    cas_cli = _collect_subparser_dests(create_apex_cas_parser())
    for command, expected in CAS_CLI_SURFACES.items():
        actual = cas_cli.get(command)
        if actual is None:
            issues.append(f"APEX_CAS CLI missing command: {command}")
            continue
        if expected != actual:
            issues.append(
                f"APEX_CAS CLI command '{command}' drifted: expected {sorted(expected)}, got {sorted(actual)}"
            )

    filter_cli = _collect_subparser_dests(create_apex_filter_parser())
    for command, expected in FILTER_CLI_SURFACES.items():
        actual = filter_cli.get(command)
        if actual is None:
            issues.append(f"APEX_Filter CLI missing command: {command}")
            continue
        if expected != actual:
            issues.append(
                f"APEX_Filter CLI command '{command}' drifted: expected {sorted(expected)}, got {sorted(actual)}"
            )

    result = {
        "status": "OK" if not issues else "ISSUES",
        "scope": {
            "apex_cas": "prepare -> scf -> buildcas -> fcidump -> testcas",
            "apex_filter": "load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report",
            "excluded": ["step11+", "fno_uccsdtq runtime execution"],
        },
        "checked": checked,
        "categories": {
            "consumed_runtime": "Actively changes runtime behavior on the V1 mainline.",
            "bootstrap_only": "Used to discover or rebuild authoritative inputs before Step 1 / staged runtime.",
            "record_only": "Kept for provenance or retained sidecar reporting; does not change the active-space Hamiltonian on the V1 route.",
            "out_of_scope": "Retained in repo but explicitly excluded from this V1 audit pass.",
        },
        "issues": issues,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
