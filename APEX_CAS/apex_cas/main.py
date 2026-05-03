"""APEX_CAS CLI entry point."""

from __future__ import annotations

import argparse
import os
from dataclasses import fields as _fields, replace as _replace

from pyscf import lib
from shared.cluster_info_labels import require_authoritative_cluster_info as _require_authoritative_cluster_info
from shared.chkfiles import find_chkfile as _find_chkfile
from shared.models import ComputationSettings as _ComputationSettings
from shared.selection_io import load_active_selection as _load_active_selection
from shared.settings_payloads import build_base_settings_payload as _build_base_settings_payload
from shared.setting_utils import (
    apply_overrides as _apply_overrides,
    load_cas_settings_file as _load_cas_settings_file,
    settings_from_preset as _settings_from_preset,
)
from shared.structure_parser import parse_structure as _parse_structure

from . import CAS_builder as cas_builder_module
from . import FCIDUMP_generator as fcidump_generator_module
from . import orbital_visualizer as orbital_visualizer_module
from . import prepare as prepare_module
from . import selection_io as selection_io_module
from . import state_io as state_io_module
from .CAS_quality import _print_quality_report, _validate_noon
from .CAS_tester import _run_dmrg_test


def create_parser():
    """Create the staged APEX_CAS CLI parser."""
    parser = argparse.ArgumentParser(
        prog="apex-cas",
        description="APEX_CAS — staged active-space construction workflow",
    )
    subparsers = parser.add_subparsers(dest="command")

    def add_common_structure_args(cmd):
        cmd.add_argument("structure", help="Path to structure file (XYZ/PDB)")
        cmd.add_argument(
            "--case-dir",
            help="Case directory (default: structure file parent, or parent of inputs/)",
        )
        cmd.add_argument(
            "--cas-settings",
            help="CAS settings YAML (see shared/config/cas_settings_template.yaml)",
        )
        cmd.add_argument("--charge", type=int, default=None, help="Override total charge")
        cmd.add_argument(
            "--spin",
            "--total-spin",
            dest="total_spin",
            type=float,
            default=None,
            help="Override target total spin S stored in cluster metadata",
        )
        cmd.add_argument(
            "--scf-spin",
            type=float,
            default=None,
            help="Override high-spin SCF reference S",
        )
        cmd.add_argument(
            "--symmetry-group",
            default=None,
            help="Override full-molecule symmetry_group metadata",
        )
        cmd.add_argument(
            "--reduction-symmetry",
            default=None,
            help="Override reduction_symmetry metadata",
        )
        cmd.add_argument(
            "--symmetry-mode",
            default=None,
            help="Symmetry detection mode override (default: auto)",
        )

    prepare_cmd = subparsers.add_parser(
        "prepare",
        help="Generate draft/finalized cluster_info artifacts from a structure",
    )
    add_common_structure_args(prepare_cmd)
    prepare_cmd.add_argument(
        "--finalize",
        action="store_true",
        help="Validate edited draft CSV and write authoritative cluster_info.yaml",
    )
    prepare_cmd.add_argument(
        "--draft-csv",
        default=None,
        help="Optional edited draft CSV to finalize",
    )
    prepare_cmd.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing finalized cluster_info.yaml",
    )

    scf_cmd = subparsers.add_parser(
        "scf",
        help="Run only the high-spin SCF stage and save checkpoint/summary",
    )
    add_common_structure_args(scf_cmd)

    buildcas_cmd = subparsers.add_parser(
        "buildcas",
        help="Build CAS from a saved SCF checkpoint",
    )
    add_common_structure_args(buildcas_cmd)
    buildcas_cmd.add_argument(
        "--no-cubes",
        action="store_true",
        help="Skip cube-file generation",
    )
    buildcas_cmd.add_argument(
        "--cube-grid",
        default=None,
        help="Override cube grid resolution, e.g. 80x80x80",
    )

    compute_cmd = subparsers.add_parser(
        "compute",
        help="Convenience wrapper over scf -> buildcas",
    )
    add_common_structure_args(compute_cmd)
    compute_cmd.add_argument(
        "--no-cubes",
        action="store_true",
        help="Skip cube-file generation",
    )
    compute_cmd.add_argument(
        "--cube-grid",
        default=None,
        help="Override cube grid resolution, e.g. 80x80x80",
    )

    fci_cmd = subparsers.add_parser(
        "fcidump",
        help="Generate active-space FCIDUMP from saved CAS state",
    )
    fci_cmd.add_argument("--case-dir", required=True, help="Case directory containing outputs/")
    fci_cmd.add_argument(
        "--spin-projection",
        type=float,
        default=None,
        help="Spin projection S used to set FCIDUMP MS2 (defaults to saved CAS target spin)",
    )
    fci_cmd.add_argument(
        "--output",
        default=None,
        help="Optional FCIDUMP output path (default: outputs/fcidump/FCIDUMP.<stem>)",
    )
    fci_cmd.add_argument(
        "--active-file",
        default=None,
        help="Selection file to use instead of the default saved selection",
    )
    fci_cmd.add_argument(
        "--zero-ecore",
        action="store_true",
        default=True,
        help="Write ECORE=0 and store the real value in a .ecore sidecar",
    )
    fci_cmd.add_argument(
        "--no-zero-ecore",
        dest="zero_ecore",
        action="store_false",
        help="Write the physical ECORE value directly into FCIDUMP",
    )

    test_cmd = subparsers.add_parser(
        "testcas",
        help="Run a small DMRG smoke test on an FCIDUMP",
    )
    test_cmd.add_argument("fcidump_path", help="Path to the FCIDUMP file")
    test_cmd.add_argument("-M", "--bond-dim", type=int, default=500, help="DMRG bond dimension")
    test_cmd.add_argument(
        "--symm",
        choices=["sz", "su2"],
        default="sz",
        help="DMRG symmetry channel",
    )
    test_cmd.add_argument("--output-dir", default=None, help="Optional output directory")
    test_cmd.add_argument("--label", default=None, help="Optional output label override")
    test_cmd.add_argument(
        "--dmrg-mode",
        default="benchmark",
        choices=["benchmark", "workflow"],
        help="DMRG sweep-schedule mode",
    )
    test_cmd.add_argument("--n-sweeps", type=int, default=None, help="Optional total sweeps override")
    test_cmd.add_argument("--max-iter", type=int, default=None, help="Optional SZ maxIter override")
    test_cmd.add_argument(
        "--stack-mem-gb",
        type=float,
        default=None,
        help="Optional pyblock2 stack memory in GB",
    )

    return parser


def _resolve_case_dir(args) -> str:
    """Resolve the case directory from CLI inputs."""
    if getattr(args, "case_dir", None):
        case_dir = os.path.abspath(args.case_dir)
    else:
        struct_dir = os.path.dirname(os.path.abspath(args.structure))
        case_dir = os.path.dirname(struct_dir) if os.path.basename(struct_dir) == "inputs" else struct_dir

    for subdir in ("scf", "orbitals", "fcidump"):
        os.makedirs(os.path.join(case_dir, "outputs", subdir), exist_ok=True)

    return case_dir


def _ensure_settings_defaults(settings):
    """Backfill workflow settings that older presets may still omit."""
    extra_defaults = {
        "basis_set_file": None,
        "density_fit": False,
        "density_fit_auxbasis": None,
        "density_fit_only_dfj": False,
        "grids_level": 3,
        "grids_small_rho_cutoff": 1e-7,
        "grids_prune": "nwchem",
        "frac_occ": False,
        "smearing_method": "none",
        "smearing_sigma": 0.01,
        "scf_stage1_rough": False,
        "scf_stage3_newton": False,
        "newton_max_cycle": 10,
        "newton_conv_tol": 1e-10,
        "scf_allow_unconverged": False,
        "pm_pop_method": "mulliken",
        "pm_conv_tol": 1e-8,
        "pm_conv_tol_grad": None,
        "pm_max_cycle": 100,
        "pm_exponent": 2,
        "pm_init_guess": "atomic",
        "boys_conv_tol": 1e-7,
        "boys_conv_tol_grad": None,
        "boys_max_cycle": 150,
    }
    for key, value in extra_defaults.items():
        if not hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def _build_settings(args):
    """Build staged-workflow settings and metadata from CLI + YAML."""
    raw = _load_cas_settings_file(args.cas_settings) if getattr(args, "cas_settings", None) else {}

    charge = raw.pop("charge", 0)
    total_spin = raw.pop("spin", 0.0)
    preset_name = raw.pop("preset", "default")
    settings = _settings_from_preset(preset_name)

    dataclass_keys = {f.name for f in _fields(_ComputationSettings)}
    dataclass_overrides = {}
    for key in list(raw):
        if key in dataclass_keys or key in {"basis_set_per_element", "basis_set_file"}:
            dataclass_overrides[key] = raw.pop(key)
    # If the user explicitly overrides the default basis but does not provide
    # per-element overrides, do not silently inherit the preset's mixed basis.
    if (
        "basis_set_default" in dataclass_overrides
        and "basis_set_per_element" not in dataclass_overrides
        and "basis_set_file" not in dataclass_overrides
    ):
        settings = _replace(settings, basis_set_per_element={})
    settings = _apply_overrides(settings, **dataclass_overrides)
    settings = _ensure_settings_defaults(settings)

    known_setting_extras = [
        "density_fit",
        "density_fit_auxbasis",
        "density_fit_only_dfj",
        "grids_level",
        "grids_small_rho_cutoff",
        "grids_prune",
        "frac_occ",
        "smearing_method",
        "smearing_sigma",
        "scf_stage1_rough",
        "scf_stage3_newton",
        "newton_max_cycle",
        "newton_conv_tol",
        "scf_allow_unconverged",
        "pm_pop_method",
        "pm_conv_tol",
        "pm_conv_tol_grad",
        "pm_max_cycle",
        "pm_exponent",
        "pm_init_guess",
        "boys_conv_tol",
        "boys_conv_tol_grad",
        "boys_max_cycle",
    ]
    for key in known_setting_extras:
        if key in raw:
            setattr(settings, key, raw.pop(key))

    localization_method = raw.pop("localization_method", "pm")
    cpt_cas_type = raw.pop("cpt_cas_type", "uno")
    projection_threshold = raw.pop("projection_threshold", 0.3)

    viz_config = {
        "generate_cubes": raw.pop("generate_cubes", True),
        "cube_grid": raw.pop("cube_grid", "80x80x80"),
        "pw_plot_threshold": raw.pop("pw_plot_threshold", None),
        "render_png": raw.pop("render_png", False),
        "png_isovalue": raw.pop("png_isovalue", 0.05),
    }
    if getattr(args, "no_cubes", False):
        viz_config["generate_cubes"] = False
    if getattr(args, "cube_grid", None):
        viz_config["cube_grid"] = args.cube_grid

    cas_build_config = {
        "projection_threshold": projection_threshold,
        "localization_method": localization_method,
        "cpt_cas_type": cpt_cas_type,
    }

    symmetry_group_override = raw.pop("symmetry_group_override", raw.pop("symmetry_group", None))
    reduction_symmetry_override = raw.pop(
        "reduction_symmetry_override",
        raw.pop("reduction_symmetry", None),
    )
    avas_config = raw.pop("avas_config", None)
    cluster_info_path = raw.pop("cluster_info_path", None)
    symmetry_mode = raw.pop("symmetry_detection_mode", raw.pop("symmetry_mode", "auto"))
    family_scheme = raw.pop("family_scheme", "")
    benchmark_profile = raw.pop("benchmark_profile", "")
    config_reduction_mode = raw.pop("config_reduction_mode", "none")

    if args.charge is not None:
        charge = args.charge
    if getattr(args, "total_spin", None) is not None:
        total_spin = args.total_spin
    if getattr(args, "scf_spin", None) is not None:
        settings.scf_spin = args.scf_spin
    if getattr(args, "symmetry_group", None) is not None:
        symmetry_group_override = args.symmetry_group
    if getattr(args, "reduction_symmetry", None) is not None:
        reduction_symmetry_override = args.reduction_symmetry
    if getattr(args, "symmetry_mode", None) is not None:
        symmetry_mode = args.symmetry_mode

    return (
        settings,
        charge,
        total_spin,
        localization_method,
        viz_config,
        cas_build_config,
        cpt_cas_type,
        symmetry_group_override,
        reduction_symmetry_override,
        avas_config,
        cluster_info_path,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
    )


def _resolve_case_relative_path(case_dir: str, path_value: str | None) -> str | None:
    """Resolve an optional path relative to the case directory."""
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(case_dir, path_value))


def _get_output_stem_safe(cluster_info, settings) -> str:
    """Best-effort stem resolution for runtime and lightweight tests."""
    try:
        return cas_builder_module._get_output_stem(cluster_info, settings)
    except Exception:
        return getattr(cluster_info, "formula", "cluster")


def _parse_compute_cluster(
    args,
    case_dir: str,
    charge: int,
    total_spin: float,
    symmetry_group_override: str | None,
    reduction_symmetry_override: str | None,
    symmetry_mode: str,
    family_scheme: str,
    benchmark_profile: str,
    config_reduction_mode: str,
    cluster_info_path: str | None,
):
    """Parse structure + cluster-info authority into ClusterInfo."""
    resolved_cluster_info = _resolve_case_relative_path(case_dir, cluster_info_path)
    if not resolved_cluster_info:
        raise FileNotFoundError(
            "APEX_CAS staged workflows require a finalized cluster_info.yaml. "
            "Run 'apex-cas prepare --finalize' first, or provide cluster_info_path "
            "in cas_settings.yaml."
        )
    cluster_info = _parse_structure(
        args.structure,
        charge=charge,
        target_spin=total_spin,
        cluster_info_path=resolved_cluster_info,
        symmetry_group_override=symmetry_group_override,
        reduction_symmetry_override=reduction_symmetry_override,
        symmetry_detection_mode=symmetry_mode,
        family_scheme=family_scheme,
        benchmark_profile=benchmark_profile,
        config_reduction_mode=config_reduction_mode,
    )
    _require_authoritative_cluster_info(
        cluster_info,
        context="APEX_CAS staged workflow",
    )
    return cluster_info


def _run_scf_only(cluster_info, settings, case_dir: str):
    """Run only the SCF initialization stage."""
    return cas_builder_module.run_scf_initialization(
        cluster_info,
        settings,
        save_dir=os.path.join(case_dir, "outputs", "scf"),
    )


def _restore_mean_field_from_chkfile(chkfile_path: str, settings):
    """Restore mol/mf from a saved chkfile using the current settings stack."""
    mol = lib.chkfile.load_mol(chkfile_path)
    scf_data = lib.chkfile.load(chkfile_path, "scf")
    mf = cas_builder_module._build_mf_object(mol, settings)
    mf.__dict__.update(scf_data)
    mf.chkfile = chkfile_path
    stem = os.path.splitext(os.path.basename(chkfile_path))[0]
    output_dir = os.path.dirname(os.path.dirname(chkfile_path))
    scf_summary = state_io_module._load_scf_summary(output_dir, stem)
    if "converged" in scf_summary:
        mf.converged = bool(scf_summary["converged"])
    return mol, mf


def _build_cas_from_saved_scf(
    cluster_info,
    settings,
    case_dir: str,
    *,
    localization_method: str,
    projection_threshold: float,
    cpt_cas_type: str,
    avas_config,
):
    """Reconstruct mol/mf from saved SCF output and build CAS."""
    scf_dir = os.path.join(case_dir, "outputs", "scf")
    chkfile_path = _find_chkfile(scf_dir)
    mol, mf = _restore_mean_field_from_chkfile(chkfile_path, settings)
    cas = cas_builder_module.build_cas_from_mean_field(
        mol,
        mf,
        cluster_info,
        computation_settings=settings,
        cpt_cas_type=cpt_cas_type,
        localization_method=localization_method,
        projection_threshold=projection_threshold,
        avas_config=avas_config,
    )
    return cas, mol, mf, chkfile_path


def _synchronize_cas_labels(cas, mol=None, cluster_info=None):
    """Normalize CAS labels and, when possible, refresh chemical labels."""
    if mol is not None and getattr(cas, "mo_coeff_full", None) is not None:
        cas.orbital_labels_full = orbital_visualizer_module._ensure_chemical_labels(
            mol,
            cas.mo_coeff_full,
            getattr(cas, "orbital_labels_full", None) or [],
            cluster_info,
        )
        if getattr(cas, "active_indices", None) is not None:
            cas.orbital_labels = [cas.orbital_labels_full[i] for i in cas.active_indices]

    if getattr(cas, "orbital_labels_full", None):
        cas.orbital_labels_full = [str(label) for label in cas.orbital_labels_full]
    if getattr(cas, "orbital_labels", None):
        cas.orbital_labels = [str(label) for label in cas.orbital_labels]
    return cas


def _save_and_validate_compute_outputs(
    cas,
    mol,
    mf,
    *,
    output_dir: str,
    stem: str,
    settings,
    charge: int,
    total_spin: float,
    cas_build_config: dict,
):
    """Persist CAS state + active-space selection sidecars."""
    cas_h5_path = state_io_module._save_cas_state(
        cas,
        mol,
        mf,
        output_dir=output_dir,
        stem=stem,
        settings=settings,
        charge=charge,
        target_spin=total_spin,
        settings_payload=cas_build_config,
    )
    selection_path = selection_io_module._generate_selection_file(
        cas,
        os.path.join(output_dir, "orbitals", f"{stem}_selection.txt"),
    )
    state_io_module._update_cas_summary(
        output_dir,
        stem,
        generated_files={
            "selection_file": selection_path,
        },
    )
    return {
        "cas_h5_path": cas_h5_path,
        "selection_path": selection_path,
    }


def _generate_compute_visualizations(
    cas,
    mol,
    *,
    case_dir: str,
    stem: str,
    cluster_info,
    viz_config: dict,
):
    """Generate the buildcas/compute review artifacts."""
    orbitals_dir = os.path.join(case_dir, "outputs", "orbitals")
    os.makedirs(orbitals_dir, exist_ok=True)
    viz = orbital_visualizer_module.plot_orbitals(
        cas,
        mol,
        orbitals_dir,
        cluster_info=cluster_info,
        generate_cubes=viz_config.get("generate_cubes", True),
        cube_grid=viz_config.get("cube_grid", "80x80x80"),
        stem=stem,
        pw_plot_threshold=viz_config.get("pw_plot_threshold"),
        render_png=viz_config.get("render_png", False),
        png_isovalue=viz_config.get("png_isovalue", 0.05),
    )

    if viz.get("chemical_labels"):
        cas.orbital_labels_full = list(viz["chemical_labels"])
        if getattr(cas, "active_indices", None) is not None:
            cas.orbital_labels = [cas.orbital_labels_full[i] for i in cas.active_indices]

    result = {
        "report_path": viz.get("report_path", ""),
        "noon_path": viz.get("noon_path", ""),
        "cube_dir": viz.get("cube_dir", ""),
        "html_gallery_path": viz.get("html_gallery_path", ""),
    }
    state_io_module._update_cas_summary(
        os.path.join(case_dir, "outputs"),
        stem,
        generated_files={
            "orbital_report": result["report_path"],
            "noon_plot": result["noon_path"],
            "cube_dir": result["cube_dir"],
            "orbital_gallery_html": result["html_gallery_path"],
            "orbital_gallery_server": (
                result["html_gallery_path"].replace(".html", "_server.py")
                if result["html_gallery_path"]
                else ""
            ),
        },
    )
    return result


def _print_scf_next_steps(case_dir: str, args):
    """Print suggested next commands after SCF."""
    print("\nNext step:")
    print(
        f"  apex-cas buildcas {args.structure} "
        f"--case-dir {case_dir}"
        + (f" --cas-settings {args.cas_settings}" if getattr(args, "cas_settings", None) else "")
    )


def _print_compute_next_steps(case_dir: str):
    """Print suggested next commands after buildcas/compute."""
    print("\nNext step:")
    print(f"  apex-cas fcidump --case-dir {case_dir}")


def _resolve_fcidump_spin_projection(args, cas) -> float:
    """Resolve the spin projection used for FCIDUMP MS2."""
    if getattr(args, "spin_projection", None) is not None:
        return float(args.spin_projection)
    return float(getattr(cas, "target_spin", 0.0) or 0.0)


def _resolve_fcidump_selection_path(args, case_dir: str, output_dir: str, stem: str) -> str:
    """Resolve the active-space selection file for FCIDUMP generation."""
    if getattr(args, "active_file", None):
        return _resolve_case_relative_path(case_dir, args.active_file)

    selection_path = os.path.join(output_dir, "orbitals", f"{stem}_selection.txt")
    if os.path.isfile(selection_path):
        return selection_path

    raise FileNotFoundError(
        "No active-space selection file found. Expected "
        f"{selection_path}. Re-run 'apex-cas buildcas' or provide --active-file."
    )


def _print_selected_orbitals(indices, labels, occupations):
    """Print the selected active orbitals before FCIDUMP generation."""
    print(f"  Selected {len(indices)} orbitals")
    for local_idx, global_idx in enumerate(indices):
        occ = float(occupations[global_idx]) if occupations is not None else float("nan")
        label = labels[local_idx] if local_idx < len(labels) else f"orb_{global_idx}"
        print(f"    {local_idx:3d}: MO idx {global_idx:4d}  occ={occ:.4f}  {label}")


def _run_prepare(args):
    """Generate or finalize authoritative cluster_info inputs."""
    case_dir = _resolve_case_dir(args)
    (
        _settings,
        charge,
        total_spin,
        _loc_method,
        _viz_config,
        _cas_build_config,
        _cpt_cas_type,
        symmetry_group_override,
        reduction_symmetry_override,
        _avas_config,
        _cluster_info_path,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
    ) = _build_settings(args)

    if getattr(args, "finalize", False):
        result = prepare_module._finalize_cluster_info_draft(
            args.structure,
            case_dir=case_dir,
            charge=charge,
            target_spin=total_spin,
            symmetry_group_override=symmetry_group_override,
            reduction_symmetry_override=reduction_symmetry_override,
            symmetry_detection_mode=symmetry_mode,
            family_scheme=family_scheme,
            benchmark_profile=benchmark_profile,
            config_reduction_mode=config_reduction_mode,
            draft_csv_path=getattr(args, "draft_csv", None),
            force=getattr(args, "force", False),
        )
        print(f"Finalized cluster_info.yaml: {result['final_yaml']}")
    else:
        result = prepare_module._prepare_cluster_inputs(
            args.structure,
            case_dir=case_dir,
            charge=charge,
            target_spin=total_spin,
            symmetry_group_override=symmetry_group_override,
            reduction_symmetry_override=reduction_symmetry_override,
            symmetry_detection_mode=symmetry_mode,
            family_scheme=family_scheme,
            benchmark_profile=benchmark_profile,
            config_reduction_mode=config_reduction_mode,
            force=getattr(args, "force", False),
        )
        print(f"Draft CSV: {result['draft_csv']}")
        print(f"Labeled structure: {result['labeled_png']}")
        if result["cluster_info_exists"]:
            print(f"Existing cluster_info.yaml: {result['final_yaml']}")


def _run_scf(args):
    """Run only the SCF stage."""
    case_dir = _resolve_case_dir(args)
    (
        settings,
        charge,
        total_spin,
        _loc_method,
        _viz_config,
        _cas_build_config,
        _cpt_cas_type,
        symmetry_group_override,
        reduction_symmetry_override,
        _avas_config,
        cluster_info_path,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
    ) = _build_settings(args)
    cluster_info = _parse_compute_cluster(
        args,
        case_dir,
        charge,
        total_spin,
        symmetry_group_override,
        reduction_symmetry_override,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
        cluster_info_path,
    )
    mol, mf, chkfile_path = _run_scf_only(cluster_info, settings, case_dir)
    stem = _get_output_stem_safe(cluster_info, settings)
    state_io_module._save_scf_summary(
        mf,
        os.path.join(case_dir, "outputs"),
        stem=stem,
        settings=settings,
        charge=charge,
        target_spin=total_spin,
    )
    print(f"SCF complete: {chkfile_path}")
    _print_scf_next_steps(case_dir, args)


def _run_buildcas(args):
    """Build CAS from a saved SCF checkpoint."""
    case_dir = _resolve_case_dir(args)
    (
        settings,
        charge,
        total_spin,
        localization_method,
        viz_config,
        cas_build_config,
        cpt_cas_type,
        symmetry_group_override,
        reduction_symmetry_override,
        avas_config,
        cluster_info_path,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
    ) = _build_settings(args)
    cluster_info = _parse_compute_cluster(
        args,
        case_dir,
        charge,
        total_spin,
        symmetry_group_override,
        reduction_symmetry_override,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
        cluster_info_path,
    )
    cas, mol, mf, _chkfile = _build_cas_from_saved_scf(
        cluster_info,
        settings,
        case_dir,
        localization_method=localization_method,
        projection_threshold=cas_build_config["projection_threshold"],
        cpt_cas_type=cpt_cas_type,
        avas_config=avas_config,
    )
    cas = _synchronize_cas_labels(cas, mol=mol, cluster_info=cluster_info)
    stem = _get_output_stem_safe(cluster_info, settings)
    output_dir = os.path.join(case_dir, "outputs")
    _save_and_validate_compute_outputs(
        cas,
        mol,
        mf,
        output_dir=output_dir,
        stem=stem,
        settings=settings,
        charge=charge,
        total_spin=total_spin,
        cas_build_config=cas_build_config,
    )
    _generate_compute_visualizations(
        cas,
        mol,
        case_dir=case_dir,
        stem=stem,
        cluster_info=cluster_info,
        viz_config=viz_config,
    )
    if cas.occupations is not None:
        print(_print_quality_report(_validate_noon(cas)))
    print(f"CAS: ({cas.n_electrons}e, {cas.n_orbitals}o)")
    _print_compute_next_steps(case_dir)


def _run_compute(args):
    """Convenience wrapper over SCF then buildcas."""
    case_dir = _resolve_case_dir(args)
    (
        settings,
        charge,
        total_spin,
        localization_method,
        viz_config,
        cas_build_config,
        cpt_cas_type,
        symmetry_group_override,
        reduction_symmetry_override,
        avas_config,
        cluster_info_path,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
    ) = _build_settings(args)
    cluster_info = _parse_compute_cluster(
        args,
        case_dir,
        charge,
        total_spin,
        symmetry_group_override,
        reduction_symmetry_override,
        symmetry_mode,
        family_scheme,
        benchmark_profile,
        config_reduction_mode,
        cluster_info_path,
    )

    mol, mf, _chkfile = _run_scf_only(cluster_info, settings, case_dir)
    stem = _get_output_stem_safe(cluster_info, settings)
    state_io_module._save_scf_summary(
        mf,
        os.path.join(case_dir, "outputs"),
        stem=stem,
        settings=settings,
        charge=charge,
        target_spin=total_spin,
    )
    cas = cas_builder_module.build_cas_from_mean_field(
        mol,
        mf,
        cluster_info,
        computation_settings=settings,
        cpt_cas_type=cpt_cas_type,
        localization_method=localization_method,
        projection_threshold=cas_build_config["projection_threshold"],
        avas_config=avas_config,
    )
    cas = _synchronize_cas_labels(cas, mol=mol, cluster_info=cluster_info)
    output_dir = os.path.join(case_dir, "outputs")
    _save_and_validate_compute_outputs(
        cas,
        mol,
        mf,
        output_dir=output_dir,
        stem=stem,
        settings=settings,
        charge=charge,
        total_spin=total_spin,
        cas_build_config=cas_build_config,
    )
    _generate_compute_visualizations(
        cas,
        mol,
        case_dir=case_dir,
        stem=stem,
        cluster_info=cluster_info,
        viz_config=viz_config,
    )
    if cas.occupations is not None:
        print(_print_quality_report(_validate_noon(cas)))
    print(f"CAS: ({cas.n_electrons}e, {cas.n_orbitals}o)")
    _print_compute_next_steps(case_dir)


def _run_fcidump(args):
    """Generate FCIDUMP from saved CAS state + selection."""
    case_dir = os.path.abspath(args.case_dir)
    output_dir = os.path.join(case_dir, "outputs")
    cas, mol, mf = state_io_module.load_cas_state(case_dir)

    chkfile = _find_chkfile(os.path.join(output_dir, "scf"))
    stem = os.path.splitext(os.path.basename(chkfile))[0]
    selection_path = _resolve_fcidump_selection_path(args, case_dir, output_dir, stem)
    selected_indices, n_electrons = _load_active_selection(selection_path)
    selected_labels = [
        cas.orbital_labels_full[idx] if getattr(cas, "orbital_labels_full", None) and idx < len(cas.orbital_labels_full)
        else f"orb_{idx}"
        for idx in selected_indices
    ]
    _print_selected_orbitals(selected_indices, selected_labels, getattr(cas, "occupations_full", None))

    spin_projection = _resolve_fcidump_spin_projection(args, cas)
    fcidump_dir = os.path.join(output_dir, "fcidump")
    os.makedirs(fcidump_dir, exist_ok=True)
    output_path = args.output or os.path.join(fcidump_dir, f"FCIDUMP.{stem}")
    if not os.path.isabs(output_path):
        output_path = os.path.join(fcidump_dir, output_path)

    if cas.cpt_cas_type == "alpha_sl":
        n_alpha = mol.nelec[0]
        selected_set = set(selected_indices)
        frozen_core_idx = sorted([i for i in range(n_alpha) if i not in selected_set])
    else:
        frozen_core_idx = []

    fcidump_path = fcidump_generator_module._generate_fcidump_from_selection(
        mol,
        mf,
        cas.mo_coeff_full,
        cas.occupations_full,
        selected_indices=selected_indices,
        output_path=output_path,
        target_spin=spin_projection,
        zero_ecore=args.zero_ecore,
        frozen_core_indices=frozen_core_idx or None,
        n_electrons=n_electrons,
    )
    state_io_module._save_fcidump_summary(
        output_dir,
        stem,
        fcidump_path=fcidump_path,
        selection_file=selection_path,
        n_electrons=n_electrons,
        n_orbitals=len(selected_indices),
        ms2=int(round(2 * spin_projection)),
        target_spin=spin_projection,
        zero_ecore=args.zero_ecore,
        frozen_core_indices=frozen_core_idx,
        settings_payload=_build_base_settings_payload(
            None,
            spin_projection=spin_projection,
            zero_ecore=bool(args.zero_ecore),
            active_file=os.path.abspath(selection_path),
            output=os.path.abspath(output_path),
        ),
    )
    print(f"FCIDUMP written to: {fcidump_path}")


def _run_testcas(args):
    """Run the FCIDUMP DMRG smoke test."""
    results = _run_dmrg_test(
        args.fcidump_path,
        bond_dim=args.bond_dim,
        output_dir=args.output_dir,
        label=args.label,
        symm_type=args.symm,
        stack_mem_gb=args.stack_mem_gb,
        dmrg_mode=args.dmrg_mode,
        n_sweeps=args.n_sweeps,
        max_iter=args.max_iter,
    )
    print(f"E_total = {results['e_total']:.12f} Eh")


def main():
    """Main CLI dispatcher."""
    parser = create_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    commands = {
        "prepare": _run_prepare,
        "scf": _run_scf,
        "buildcas": _run_buildcas,
        "compute": _run_compute,
        "fcidump": _run_fcidump,
        "testcas": _run_testcas,
    }

    cmd = commands.get(args.command)
    if cmd is None:
        parser.print_help()
        return
    cmd(args)


if __name__ == "__main__":
    main()
