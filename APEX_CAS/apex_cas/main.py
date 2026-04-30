"""APEX_CAS — CLI Entry Point

Commands:
  compute  — Parse structure → SCF → CAS construction → visualization
  fcidump  — Load saved state → read user YAML → generate FCIDUMP
"""

import argparse
import os
import sys

import numpy as np


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="apex-cas",
        description="APEX_CAS — Automated Chemical Active Space Construction",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── compute command ──
    comp = subparsers.add_parser("compute",
        help="Parse structure → compute CAS → save state → visualize orbitals")
    comp.add_argument("structure", help="Path to structure file (XYZ/PDB)")
    comp.add_argument("--case-dir",
        help="Case directory (default: structure file's parent directory)")
    comp.add_argument("--cas-settings",
        help="YAML file with all CAS settings (see shared/config/cas_settings_template.yaml)")
    comp.add_argument("--charge", type=int, default=None,
        help="Total charge (overrides YAML)")
    comp.add_argument("--spin", type=float, default=None,
        help="High-spin S value for SCF (overrides YAML, e.g. 17.5 for S=35/2)")
    comp.add_argument("--no-cubes", action="store_true",
        help="Skip cube file generation")
    comp.add_argument("--cube-grid", default="80x80x80",
        help="Cube grid resolution (default: 80x80x80)")

    # ── fcidump command ──
    fci = subparsers.add_parser("fcidump",
        help="Load saved state → read user YAML → generate FCIDUMP")
    fci.add_argument("--case-dir", required=True,
        help="Case directory containing outputs/")
    fci.add_argument("--target-spin", type=float, default=0.0,
        help="Target spin S for MS2 (e.g., 1.5 for S=3/2)")
    fci.add_argument("--output", default="FCIDUMP",
        help="Output FCIDUMP file path (relative to case-dir/outputs/fcidump/)")
    fci.add_argument("--reference-fcidump",
        help="Reference FCIDUMP for comparison")
    fci.add_argument("--selection",
        help="User-edited orbital_report.yaml (default: case-dir/outputs/orbitals/orbital_report.yaml)")
    fci.add_argument("--zero-ecore", action="store_true", default=True,
        help="Write ECORE=0 in FCIDUMP (DMRG convention, default)")
    fci.add_argument("--no-zero-ecore", dest="zero_ecore", action="store_false",
        help="Write actual ECORE value in FCIDUMP (PySCF convention)")

    return parser


# ── Shared Helpers ──────────────────────────────────────────

def _resolve_case_dir(args) -> str:
    """Determine case directory.

    Priority: --case-dir > grandparent if parent is "inputs" > parent directory.
    Creates outputs/ subdirectories if they don't exist.
    """
    if getattr(args, "case_dir", None):
        case_dir = os.path.abspath(args.case_dir)
    else:
        struct_dir = os.path.dirname(os.path.abspath(args.structure))
        # If the structure file lives in an "inputs" directory, use its parent
        if os.path.basename(struct_dir) == "inputs":
            case_dir = os.path.dirname(struct_dir)
        else:
            case_dir = struct_dir

    # Create output subdirectories
    for subdir in ["scf", "orbitals", "fcidump"]:
        os.makedirs(os.path.join(case_dir, "outputs", subdir), exist_ok=True)

    return case_dir


def _build_settings(args):
    """Build ComputationSettings from CLI args and optional YAML file.

    All computation parameters are managed through the YAML file specified
    by --cas-settings.  If no YAML is provided, preset defaults are used.

    Returns:
        Tuple of (settings, charge, spin, localization_method).
    """
    from .computation_defaults import PRESETS, load_cas_settings_file, apply_overrides

    charge = 0
    spin = 0.0
    localization_method = "pm"
    settings = PRESETS["default"]

    if getattr(args, "cas_settings", None):
        yaml_data = load_cas_settings_file(args.cas_settings)

        # Extract CLI-layer fields (not part of ComputationSettings)
        charge = yaml_data.pop("charge", 0)
        spin = yaml_data.pop("spin", 0.0)
        localization_method = yaml_data.pop("localization_method", "pm")
        preset_name = yaml_data.pop("preset", "default")

        settings = PRESETS.get(preset_name, PRESETS["default"])
        settings = apply_overrides(settings, **yaml_data)

    # CLI flags override YAML values
    if args.charge is not None:
        charge = args.charge
    if args.spin is not None:
        spin = args.spin

    return settings, charge, spin, localization_method


def _print_next_steps(case_dir, viz, args):
    """Print next-step instructions after compute."""
    report_path = viz.get("report_path", "")
    print(f"\n{'=' * 60}")
    print(f"  NEXT STEPS:")
    print(f"  1. View cube files in VESTA/Jmol (if generated)")
    print(f"  2. Check NOON plot: {viz.get('noon_path', '')}")
    print(f"  3. Edit the orbital report, mark selected orbitals:")
    print(f"     {report_path}")
    print(f"     Set 'selected: true' for orbitals to include.")
    print(f"  4. Generate FCIDUMP:")
    print(f"     apex-cas fcidump --case-dir {case_dir} \\")
    print(f"       --target-spin <S>")
    print(f"{'=' * 60}")


# ── compute Command ─────────────────────────────────────────

def run_compute(args):
    """Step 1+2+3: Parse → Compute CAS → Save state → Plot orbitals."""
    from .structure_analyzer import parse_structure
    from .CAS_builder_computing import build_computed_CAS
    from .orbital_visualizer import plot_orbitals, save_cas_state
    from .CAS_quality import validate_noon, print_quality_report

    # ── Resolve paths ──
    case_dir = _resolve_case_dir(args)
    output_dir = os.path.join(case_dir, "outputs")

    # ── Build settings (from YAML + CLI overrides) ──
    settings, charge, spin, loc_method = _build_settings(args)

    # ── Step 1: Parse Structure ──
    cluster_info = parse_structure(args.structure, charge, spin)
    print(f"Structure: {cluster_info.formula} (charge={charge}, S={spin})")

    # ── Step 2: Build Computed CAS ──
    print(f"Running {settings.xc_functional}/{settings.scf_method.upper()} ...")
    cas, mol, mf, chkfile_path = build_computed_CAS(
        cluster_info, settings,
        localization_method=loc_method,
        save_dir=os.path.join(output_dir, "scf"),
    )
    print(f"CAS: ({cas.n_electrons}e, {cas.n_orbitals}o)")

    # ── Save state ──
    save_cas_state(cas, mol, mf, output_dir)
    print(f"  State saved to {output_dir}/")

    # ── Validate (automatic) ──
    if cas.occupations is not None:
        quality = validate_noon(cas)
        print(print_quality_report(quality))

    # ── Step 3: Plot Orbitals ──
    if cas.mo_coeff_full is not None:
        orbitals_dir = os.path.join(output_dir, "orbitals")
        print(f"Generating orbital visualizations ...")
        viz = plot_orbitals(
            cas, mol, orbitals_dir,
            cluster_info=cluster_info,
            generate_cubes=not args.no_cubes,
            cube_grid=args.cube_grid,
        )
        print(f"  Report: {viz['report_path']}")
        print(f"  NOON plot: {viz['noon_path']}")
        if viz.get("cube_dir"):
            print(f"  Cubes: {viz['cube_dir']}/")
        _print_next_steps(case_dir, viz, args)
    else:
        print("  WARNING: No full orbital data available for visualization.")


# ── fcidump Command ─────────────────────────────────────────

def run_fcidump(args):
    """Step 4: Load saved state → read user YAML → generate FCIDUMP."""
    from .orbital_visualizer import load_cas_state, load_user_selection
    from .FCIDUMP_generator import (
        generate_fcidump_from_selection,
        compare_fcidumps,
    )

    case_dir = os.path.abspath(args.case_dir)
    output_dir = os.path.join(case_dir, "outputs")

    # ── Load saved state ──
    print(f"Loading state from {case_dir} ...")
    cas, mol, mf = load_cas_state(case_dir)
    print(f"  CAS: ({cas.n_electrons}e, {cas.n_orbitals}o), type={cas.cpt_cas_type}")

    # ── Resolve paths relative to case_dir ──
    # selection: default is case_dir/outputs/orbitals/orbital_report.yaml
    yaml_path = args.selection or os.path.join(
        output_dir, "orbitals", "orbital_report.yaml")
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(case_dir, yaml_path)
    if not os.path.isfile(yaml_path):
        print(f"ERROR: Selection file not found: {yaml_path}")
        sys.exit(1)

    # reference-fcidump: resolve relative to case_dir if not absolute
    ref_fcidump = args.reference_fcidump
    if ref_fcidump and not os.path.isabs(ref_fcidump):
        if not os.path.exists(ref_fcidump):
            ref_fcidump = os.path.join(case_dir, ref_fcidump)

    selected_indices, selected_labels, metadata = load_user_selection(yaml_path)
    if not selected_indices:
        print("ERROR: No orbitals selected. Edit the YAML and set selected: true.")
        sys.exit(1)

    occ_uno = cas.occupations_full
    n_electrons = int(round(float(np.sum(occ_uno[selected_indices]))))
    n_orbitals = len(selected_indices)

    print(f"\n  Selected: {n_orbitals} orbitals, {n_electrons} electrons")
    for i, (idx, lab) in enumerate(zip(selected_indices, selected_labels)):
        print(f"    {i:3d}: MO idx {idx:4d}  occ={occ_uno[idx]:.4f}  {lab}")

    # ── Generate FCIDUMP ──
    fcidump_dir = os.path.join(output_dir, "fcidump")
    os.makedirs(fcidump_dir, exist_ok=True)
    output_path = os.path.join(fcidump_dir, args.output)

    print(f"\n  Generating FCIDUMP ...")
    mo_coeff_loc = cas.mo_coeff_full
    fcidump_path = generate_fcidump_from_selection(
        mol, mf, mo_coeff_loc, occ_uno,
        selected_indices=selected_indices,
        output_path=output_path,
        target_spin=args.target_spin,
        zero_ecore=args.zero_ecore,
    )
    print(f"  FCIDUMP written to: {fcidump_path}")

    # ── Compare if reference provided ──
    if ref_fcidump:
        if os.path.exists(ref_fcidump):
            print(f"\n  Comparing with reference: {ref_fcidump}")
            result = compare_fcidumps(ref_fcidump, fcidump_path)
            print(f"    h1e Frobenius: {result['h1e_frobenius']:.6e}")
            print(f"    h1e max diff:  {result['h1e_max']:.6e}")
            print(f"    h2e RMS diff:  {result['h2e_rms']:.6e}")
            print(f"    ecore diff:    {result['ecore_diff']:+.6e}")
            if result["match"]:
                print("    Status: GOOD agreement")
            else:
                print("    Status: WARNING — significant differences detected")
        else:
            print(f"  WARNING: Reference FCIDUMP not found: {ref_fcidump}")


# ── Main ────────────────────────────────────────────────────

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "compute": run_compute,
        "fcidump": run_fcidump,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
