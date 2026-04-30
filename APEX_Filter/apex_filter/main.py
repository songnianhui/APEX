"""APEX_Filter — CLI Entry Point

Orchestrates the filtering pipeline:
  1. Parse structure → ClusterInfo (via apex_cas)
  2. Build active space → CAS (via apex_cas)
  3. Enumerate spin isomers → list[SpinIsomer]
  4. Enumerate electronic configs → list[ElectronicConfig]
  5. Design filtering funnel → FilteringPlan
  6. Generate input files
  7. Generate FCIDUMP
"""

import argparse
import os
import sys
from itertools import product

import numpy as np

from .models import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ExtrapolatedEnergy,
    ComputationSettings,
    MetalCenter,
)


def create_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="apex-filter",
        description="APEX_Filter — Electronic Structure Filtering Pipeline for transition metal clusters",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # spin-only command (fast, no structure parsing)
    spin_cmd = subparsers.add_parser("spin", help="Spin isomer enumeration only")
    spin_cmd.add_argument("--metals", nargs="+", required=True,
                           help="Metal elements, e.g., Fe Fe Fe Fe Fe Fe Fe Mo")
    spin_cmd.add_argument("--spin", type=float, required=True,
                           help="Target total Sz")
    spin_cmd.add_argument("--oxidation", nargs="+", type=int, default=None,
                           help="Oxidation states for each metal")
    spin_cmd.add_argument("--symmetry", default="C1", help="Symmetry group")

    # filter command
    filter_cmd = subparsers.add_parser("filter", help="Design filtering funnel")
    filter_cmd.add_argument("--n-configs", type=int, required=True,
                             help="Total number of configurations")
    filter_cmd.add_argument("--n-electrons", type=int, required=True,
                             help="Active space electrons")
    filter_cmd.add_argument("--n-orbitals", type=int, required=True,
                             help="Active space orbitals")
    filter_cmd.add_argument("--n-isomers", type=int, default=None,
                             help="Number of spin isomers")
    filter_cmd.add_argument("--style", choices=["femoco", "conservative", "minimal"],
                             default="femoco", help="Filtering funnel style")

    # fcidump command
    fcicmd = subparsers.add_parser("fcidump",
                                    help="Generate FCIDUMP from pipeline results")
    fcicmd.add_argument("structure", help="Path to structure file (XYZ/PDB)")
    fcicmd.add_argument("--charge", type=int, default=0, help="Total charge")
    fcicmd.add_argument("--spin", type=float, default=0.0, help="Target spin S")
    fcicmd.add_argument("--basis", default="cc-pVDZ", help="Basis set")
    fcicmd.add_argument("--uhf-npz", default=None,
                         help="Path to UHF *_uhf.npz (skips SCF if provided)")
    fcicmd.add_argument("--dmrg-npz", default=None,
                         help="Path to DMRG *_dmrg_results.npz (for ncore/cas info)")
    fcicmd.add_argument("--pipeline-dir", default=None,
                         help="Pipeline output dir to auto-locate NPZ files")
    fcicmd.add_argument("--active-electrons", type=int, default=None,
                         help="Active space electrons (if no DMRG NPZ)")
    fcicmd.add_argument("--active-orbitals", type=int, default=None,
                         help="Active space orbitals (if no DMRG NPZ)")
    fcicmd.add_argument("--mode", choices=["both", "full", "active"],
                         default="both", help="FCIDUMP mode")
    fcicmd.add_argument("--output", "-o", default=None,
                         help="Output directory (default: {structure_dir}/fcidump/)")

    return parser


def run_spin(args):
    """Run spin isomer enumeration only (no structure file needed)."""
    from .models import MetalCenter, ClusterInfo
    from .spin_config import enumerate_spin_isomers
    from apex_cas.CAS_builder_noncomputing import get_local_spin, get_common_oxidation_states

    # Build minimal ClusterInfo
    metals = []
    for i, elem in enumerate(args.metals):
        metals.append(MetalCenter(
            element=elem,
            index=i,
            position=np.zeros(3),
            label=f"{elem}{i + 1}",
        ))

    cluster_info = ClusterInfo(
        metals=metals,
        target_spin=args.spin,
        symmetry_group=args.symmetry,
    )

    # Determine oxidation states
    ox_states = {}
    if args.oxidation:
        for i, ox in enumerate(args.oxidation):
            ox_states[i] = ox
    else:
        for i, elem in enumerate(args.metals):
            states = get_common_oxidation_states(elem)
            ox_states[i] = states[0] if states else 2

    # Enumerate
    isomers = enumerate_spin_isomers(cluster_info, oxidation_states=ox_states)

    print(f"Metals: {' '.join(args.metals)}")
    print(f"Target Sz = {args.spin}")
    print(f"Oxidation states: {ox_states}")
    print(f"\nLocal spins:")
    for i, metal in enumerate(metals):
        S = get_local_spin(metal.element, ox_states[i])
        print(f"  {metal.label}: oxidation={ox_states[i]:+d}, S={S}")
    print(f"\nTotal spin isomers: {len(isomers)}")

    # Group by n_minority
    from collections import Counter
    n_minority_counts = Counter(iso.n_minority for iso in isomers)
    print("\nBy number of minority-spin metals:")
    for n in sorted(n_minority_counts):
        print(f"  BS{n}: {n_minority_counts[n]} isomer(s)")

    # Print first few
    print("\nFirst 10 isomers:")
    for iso in isomers[:10]:
        minority = [k + 1 for k, v in iso.spin_assignment.items() if v == -1]
        print(f"  {iso.label}: minority at sites {minority}")


def run_filter(args):
    """Design a filtering funnel."""
    from .filtering import design_filtering_funnel
    from .models import CAS

    active_space = CAS(
        n_electrons=args.n_electrons,
        n_orbitals=args.n_orbitals,
    )

    plan = design_filtering_funnel(
        args.n_configs, active_space, args.n_isomers, style=args.style
    )

    print(f"Filtering funnel for ({args.n_electrons}e, {args.n_orbitals}o):")
    print(f"Total configurations: {plan.total_configs}")
    print()
    for level in plan.levels:
        print(f"  {level.method:>10}: {level.n_input:>8} → {level.n_output:>8}"
              f"  ({level.selection_criterion})"
              + (f"  {level.n_per_isomer}/isomer" if level.n_per_isomer else ""))
    print(f"\nFinal: {plan.levels[-1].n_output} configurations")


def run_fcidump(args):
    """Generate FCIDUMP files for a structure."""
    from apex_cas.structure_analyzer import parse_structure
    from apex_cas.CAS_builder_noncomputing import build_NC_CAS
    from .fcidump import generate_fcidump

    print(f"Generating FCIDUMP for: {args.structure}")
    cluster_info = parse_structure(args.structure, args.charge, args.spin)
    cases, _ = build_NC_CAS(cluster_info)
    active_space = cases["rule"]

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        struct_dir = os.path.dirname(os.path.abspath(args.structure))
        output_dir = os.path.join(struct_dir, "fcidump")

    # Try to auto-locate NPZ files from pipeline directory
    uhf_npz = args.uhf_npz
    dmrg_npz = args.dmrg_npz

    if args.pipeline_dir and (uhf_npz is None or dmrg_npz is None):
        import glob as glob_mod
        # Find UHF NPZ
        if uhf_npz is None:
            uhf_files = glob_mod.glob(os.path.join(args.pipeline_dir,
                                                     "level_*_UHF", "*_uhf.npz"))
            if uhf_files:
                uhf_npz = uhf_files[0]
                print(f"  Auto-located UHF NPZ: {uhf_npz}")

        # Find last DMRG NPZ
        if dmrg_npz is None:
            dmrg_files = sorted(glob_mod.glob(os.path.join(args.pipeline_dir,
                                                            "level_*_DMRG",
                                                            "*_dmrg_results.npz")))
            if dmrg_files:
                dmrg_npz = dmrg_files[-1]
                print(f"  Auto-located DMRG NPZ: {dmrg_npz}")

    # Override active space if manually specified
    if args.active_electrons is not None and args.active_orbitals is not None:
        active_space.n_electrons = args.active_electrons
        active_space.n_orbitals = args.active_orbitals

    print(f"  Active space: ({active_space.n_electrons}e, {active_space.n_orbitals}o)")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {output_dir}/")

    if uhf_npz is None:
        print("\n  No UHF NPZ provided. Running UHF calculation from scratch...")
        from .input_generator import generate_input
        # Run a quick UHF to get MO coefficients
        from pyscf import gto, scf
        import numpy as np
        geometry_lines = []
        for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
            geometry_lines.append(f"{elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
        geometry = "\n".join(geometry_lines)
        mol = gto.M(atom=geometry, basis=args.basis,
                     charge=args.charge, spin=int(round(2 * args.spin)),
                     verbose=0, symmetry=False)
        mf = scf.UHF(mol)
        mf.max_cycle = 2000
        mf.kernel()

        # Save temporary UHF NPZ
        tmp_dir = os.path.join(output_dir, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_npz = os.path.join(tmp_dir, "uhf.npz")
        np.savez(tmp_npz,
                 energy=mf.e_tot, converged=mf.converged,
                 mo_coeff_a=mf.mo_coeff[0], mo_coeff_b=mf.mo_coeff[1],
                 mo_occ_a=mf.mo_occ[0], mo_occ_b=mf.mo_occ[1],
                 mo_energy_a=mf.mo_energy[0], mo_energy_b=mf.mo_energy[1])
        uhf_npz = tmp_npz

    info = generate_fcidump(
        cluster_info, active_space, uhf_npz, output_dir,
        dmrg_npz=dmrg_npz, basis_set=args.basis,
        mode=args.mode,
    )
    print(f"\nFCIDUMP generation complete:")
    if "full_space" in info:
        print(f"  Full-space: {info['full_space']}")
    if "active_space" in info:
        print(f"  Active-space: {info['active_space']}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "spin": run_spin,
        "filter": run_filter,
        "fcidump": run_fcidump,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
