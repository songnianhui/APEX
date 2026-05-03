"""APEX_Filter CLI for the staged V1.0.0 workflow."""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from .steps_dmrg import step_dmrg, step_extrapolate_dmrg
from .steps_dmrg_basis import step_dmrg_basis
from .steps_enumeration import step_enumerate
from .steps_fno import step_cc_composite, step_fno_uccsdtq
from .steps_reference_uhf import step_uhf
from .steps_report import step_report
from .steps_setup import step_load
from .steps_ucc import step_ccsd, step_ccsd_t, step_ccsdt


def _parse_csv_ints(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid integer list '{value}'. Expected e.g. '500,1000,1500'."
        ) from exc


def _parse_yaml_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Could not parse value '{value}' as YAML/JSON.") from exc


def _add_session_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True, help="Path to the filter session directory.")


def _add_pick_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pick",
        default="all",
        help='Selection strategy, e.g. "all", "labels A,B", or "file /path/to/selection_worklist.csv".',
    )


def create_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="apex-filter",
        description="APEX_Filter staged electronic-structure workflow",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    load_cmd = subparsers.add_parser("load", help="Initialize a filter session from filter settings.")
    load_cmd.add_argument("--config", required=True, help="Path to the filter settings YAML.")
    _add_session_arg(load_cmd)

    enumerate_cmd = subparsers.add_parser("enumerate", help="Enumerate electronic configurations.")
    _add_session_arg(enumerate_cmd)
    enumerate_cmd.add_argument("--target-sz", type=float, default=None, help="Override target Sz.")
    enumerate_cmd.add_argument(
        "--forced-oxidation",
        type=_parse_yaml_value,
        default=None,
        help="Optional YAML/JSON oxidation constraint mapping.",
    )
    enumerate_cmd.add_argument("--max-configs", type=int, default=None, help="Limit the number of raw configs.")

    uhf_cmd = subparsers.add_parser("uhf", help="Run reference active-space UHF.")
    _add_session_arg(uhf_cmd)
    _add_pick_arg(uhf_cmd)
    uhf_cmd.add_argument("--conv-tol", type=float, default=1e-8)
    uhf_cmd.add_argument("--max-cycle", type=int, default=2000)
    uhf_cmd.add_argument("--stabilize-cycles", type=int, default=20)
    uhf_cmd.add_argument("--level-shift", type=float, default=0.3)
    uhf_cmd.add_argument("--damp", type=float, default=0.2)
    uhf_cmd.add_argument("--newton-refine", action="store_true")
    uhf_cmd.add_argument("--newton-max-cycle", type=int, default=8)

    ccsd_cmd = subparsers.add_parser("ccsd", help="Run active-space UCCSD.")
    _add_session_arg(ccsd_cmd)
    _add_pick_arg(ccsd_cmd)
    ccsd_cmd.add_argument("--code", choices=["pyscf"], default="pyscf")
    ccsd_cmd.add_argument("--basis-set", default="cc-pVDZ")

    ccsd_t_cmd = subparsers.add_parser("ccsd-t", help="Run active-space UCCSD(T).")
    _add_session_arg(ccsd_t_cmd)
    _add_pick_arg(ccsd_t_cmd)
    ccsd_t_cmd.add_argument("--code", choices=["pyscf"], default="pyscf")
    ccsd_t_cmd.add_argument("--basis-set", default="cc-pVDZ")
    ccsd_t_cmd.add_argument("--n-final", type=int, default=5)

    ccsdt_cmd = subparsers.add_parser("ccsdt", help="Run active-space UCCSDT.")
    _add_session_arg(ccsdt_cmd)
    _add_pick_arg(ccsdt_cmd)
    ccsdt_cmd.add_argument("--code", choices=["hast_ucc"], default="hast_ucc")
    ccsdt_cmd.add_argument("--basis-set", default="cc-pVDZ")
    ccsdt_cmd.add_argument("--n-final", type=int, default=5)
    ccsdt_cmd.add_argument("--conv-tol", type=float, default=1e-8)
    ccsdt_cmd.add_argument("--residual-tol", type=float, default=1e-6)
    ccsdt_cmd.add_argument("--max-cycle", type=int, default=2000)
    ccsdt_cmd.add_argument("--lambda-max-cycle", type=int, default=500)
    ccsdt_cmd.add_argument("--diis-space", type=int, default=6)
    ccsdt_cmd.add_argument("--diis-start-cycle", type=int, default=0)
    ccsdt_cmd.add_argument("--iterative-damping", type=float, default=1.0)
    ccsdt_cmd.add_argument("--level-shift", type=float, default=0.0)
    ccsdt_cmd.add_argument("--newton-krylov", action="store_true")

    dmrg_basis_cmd = subparsers.add_parser("dmrg-basis", help="Prepare orbital bases for DMRG.")
    _add_session_arg(dmrg_basis_cmd)
    _add_pick_arg(dmrg_basis_cmd)
    dmrg_basis_cmd.add_argument("--localization-method", default="pm")
    dmrg_basis_cmd.add_argument("--cc-conv-tol", type=float, default=1e-8)
    dmrg_basis_cmd.add_argument("--cc-max-cycle", type=int, default=2000)
    dmrg_basis_cmd.add_argument("--cc-diis-space", type=int, default=12)
    dmrg_basis_cmd.add_argument("--cc-direct", action="store_true")
    dmrg_basis_cmd.add_argument("--pm-pop-method", default="mulliken")
    dmrg_basis_cmd.add_argument("--pm-conv-tol", type=float, default=1e-6)
    dmrg_basis_cmd.add_argument("--pm-conv-tol-grad", type=float, default=None)
    dmrg_basis_cmd.add_argument("--pm-max-cycle", type=int, default=100)
    dmrg_basis_cmd.add_argument("--pm-exponent", type=int, default=2)
    dmrg_basis_cmd.add_argument("--pm-init-guess", default="atomic")
    dmrg_basis_cmd.add_argument("--boys-conv-tol", type=float, default=1e-6)
    dmrg_basis_cmd.add_argument("--boys-conv-tol-grad", type=float, default=None)
    dmrg_basis_cmd.add_argument("--boys-max-cycle", type=int, default=100)
    dmrg_basis_cmd.add_argument("--ordering-matrix-mode", default="exchange_proxy")
    dmrg_basis_cmd.add_argument("--exchange-proxy-max-orbitals", type=int, default=64)
    dmrg_basis_cmd.add_argument("--ga-generations", type=int, default=100)
    dmrg_basis_cmd.add_argument("--ga-population", type=int, default=50)
    dmrg_basis_cmd.add_argument("--ga-mutation-rate", type=float, default=0.1)
    dmrg_basis_cmd.add_argument("--ga-seed", type=int, default=17)

    dmrg_cmd = subparsers.add_parser("dmrg", help="Run active-space DMRG.")
    _add_session_arg(dmrg_cmd)
    _add_pick_arg(dmrg_cmd)
    dmrg_cmd.add_argument("--backend", default="pyblock2_sz")
    dmrg_cmd.add_argument("--basis-mode", default="step7_paired")
    dmrg_cmd.add_argument("--schedule-mode", default="workflow")
    dmrg_cmd.add_argument("--bond-dims", type=_parse_csv_ints, default=None)
    dmrg_cmd.add_argument("--n-sweeps", type=int, default=8)
    dmrg_cmd.add_argument("--convergence-tol", type=float, default=1e-8)
    dmrg_cmd.add_argument("--n-threads", type=int, default=4)
    dmrg_cmd.add_argument("--stack-mem", type=int, default=2 * 1024**3)
    dmrg_cmd.add_argument("--twosite-to-onesite", type=int, default=None)
    dmrg_cmd.add_argument("--dav-max-iter", type=int, default=None)
    dmrg_cmd.add_argument("--dav-def-max-size", type=int, default=None)
    dmrg_cmd.add_argument("--dav-rel-conv-thrd", type=float, default=None)
    dmrg_cmd.add_argument("--dav-type", default=None)

    extrapolate_cmd = subparsers.add_parser("extrapolate", help="Extrapolate DMRG energies.")
    _add_session_arg(extrapolate_cmd)

    report_cmd = subparsers.add_parser("report", help="Build the final ranked report.")
    _add_session_arg(report_cmd)

    fno_cmd = subparsers.add_parser("fno-uccsdtq", help="Run occupied-NO-freeze UCCSDT/UCCSDTQ.")
    _add_session_arg(fno_cmd)
    _add_pick_arg(fno_cmd)
    fno_cmd.add_argument("--freeze-occ", type=_parse_csv_ints, default=None)

    cc_composite_cmd = subparsers.add_parser("cc-composite", help="Build composite CC energies.")
    _add_session_arg(cc_composite_cmd)

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "load":
        step_load(args.config, args.session)
        return
    if args.command == "enumerate":
        step_enumerate(
            args.session,
            target_Sz=args.target_sz,
            forced_oxidation=args.forced_oxidation,
            max_configs=args.max_configs,
        )
        return
    if args.command == "uhf":
        step_uhf(
            args.session,
            pick=args.pick,
            conv_tol=args.conv_tol,
            max_cycle=args.max_cycle,
            stabilize_cycles=args.stabilize_cycles,
            level_shift=args.level_shift,
            damp=args.damp,
            newton_refine=args.newton_refine,
            newton_max_cycle=args.newton_max_cycle,
        )
        return
    if args.command == "ccsd":
        step_ccsd(args.session, pick=args.pick, code=args.code, basis_set=args.basis_set)
        return
    if args.command == "ccsd-t":
        step_ccsd_t(
            args.session,
            pick=args.pick,
            code=args.code,
            basis_set=args.basis_set,
            n_final=args.n_final,
        )
        return
    if args.command == "ccsdt":
        step_ccsdt(
            args.session,
            pick=args.pick,
            code=args.code,
            basis_set=args.basis_set,
            n_final=args.n_final,
            conv_tol=args.conv_tol,
            residual_tol=args.residual_tol,
            max_cycle=args.max_cycle,
            lambda_max_cycle=args.lambda_max_cycle,
            diis_space=args.diis_space,
            diis_start_cycle=args.diis_start_cycle,
            iterative_damping=args.iterative_damping,
            level_shift=args.level_shift,
            newton_krylov=args.newton_krylov,
        )
        return
    if args.command == "dmrg-basis":
        step_dmrg_basis(
            args.session,
            pick=args.pick,
            localization_method=args.localization_method,
            cc_conv_tol=args.cc_conv_tol,
            cc_max_cycle=args.cc_max_cycle,
            cc_diis_space=args.cc_diis_space,
            cc_direct=args.cc_direct,
            pm_pop_method=args.pm_pop_method,
            pm_conv_tol=args.pm_conv_tol,
            pm_conv_tol_grad=args.pm_conv_tol_grad,
            pm_max_cycle=args.pm_max_cycle,
            pm_exponent=args.pm_exponent,
            pm_init_guess=args.pm_init_guess,
            boys_conv_tol=args.boys_conv_tol,
            boys_conv_tol_grad=args.boys_conv_tol_grad,
            boys_max_cycle=args.boys_max_cycle,
            ordering_matrix_mode=args.ordering_matrix_mode,
            exchange_proxy_max_orbitals=args.exchange_proxy_max_orbitals,
            ga_generations=args.ga_generations,
            ga_population=args.ga_population,
            ga_mutation_rate=args.ga_mutation_rate,
            ga_seed=args.ga_seed,
        )
        return
    if args.command == "dmrg":
        step_dmrg(
            args.session,
            pick=args.pick,
            backend=args.backend,
            basis_mode=args.basis_mode,
            schedule_mode=args.schedule_mode,
            bond_dims=args.bond_dims,
            n_sweeps=args.n_sweeps,
            convergence_tol=args.convergence_tol,
            n_threads=args.n_threads,
            stack_mem=args.stack_mem,
            twosite_to_onesite=args.twosite_to_onesite,
            dav_max_iter=args.dav_max_iter,
            dav_def_max_size=args.dav_def_max_size,
            dav_rel_conv_thrd=args.dav_rel_conv_thrd,
            dav_type=args.dav_type,
        )
        return
    if args.command == "extrapolate":
        step_extrapolate_dmrg(args.session)
        return
    if args.command == "report":
        step_report(args.session)
        return
    if args.command == "fno-uccsdtq":
        step_fno_uccsdtq(args.session, pick=args.pick, freeze_occ=args.freeze_occ)
        return
    if args.command == "cc-composite":
        step_cc_composite(args.session)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    _dispatch(args)


if __name__ == "__main__":
    main()
