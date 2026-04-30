"""Fresh-process DMRG worker for step8 DMRG backends."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import time

import block2  # noqa: F401  # must precede pyscf/pyblock2 runtime usage
from pyscf.tools import fcidump as fcidump_mod

from .CAS_loader import load_fcidump
from .dmrg_integral_transform import build_spatial_basis, transform_integrals
from .reference_ucc import load_reference_mf_from_npz
from shared.dmrg_controls import build_dmrg_sweep_schedule, infer_pyblock2_benchmark_controls

def _prepare_working_fcidump(
    *,
    fcidump_path: str,
    dmrg_basis_npz_path: str | None,
    basis_mode: str,
    scratch: str,
):
    """Return (fcidump_data, working_fcidump_path, cleanup_path_or_none)."""
    os.makedirs(scratch, exist_ok=True)
    fcidump_data = load_fcidump(fcidump_path)
    mode = (basis_mode or "step7_paired").strip().lower()
    if mode == "original_identity":
        return fcidump_data, os.path.abspath(fcidump_path), None

    spatial_basis = build_spatial_basis(
        dmrg_basis_npz_path=dmrg_basis_npz_path,
        norb=fcidump_data.norb,
        basis_mode=mode,
    )
    h1e_t, h2e_t = transform_integrals(fcidump_data, spatial_basis)

    with tempfile.NamedTemporaryFile(
        prefix="apex_dmrg_basis_",
        suffix=".fcidump",
        dir=scratch,
        delete=False,
    ) as fh:
        transformed_fcidump = fh.name

    fcidump_mod.from_integrals(
        transformed_fcidump,
        h1e_t,
        h2e_t,
        fcidump_data.norb,
        fcidump_data.nelec,
        nuc=fcidump_data.ecore,
        ms=fcidump_data.ms2,
    )
    return fcidump_data, transformed_fcidump, transformed_fcidump


def _run_dmrg_pyblock2_sz(
    *,
    fcidump_data,
    working_fcidump: str,
    uhf_npz_path: str,
    basis_mode: str,
    bond_dim: int,
    n_sweeps: int,
    convergence_tol: float,
    schedule_mode: str,
    n_threads: int,
    stack_mem: int,
    scratch: str,
    twosite_to_onesite: int | None,
    dav_max_iter: int | None,
    dav_def_max_size: int | None,
    dav_rel_conv_thrd: float | None,
    dav_type: str | None,
):
    """Run one active-space DMRG solve via pyblock2 in the working basis."""
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    def _solve_pyblock2_stage(
        driver,
        mpo,
        ket,
        *,
        bond_dims_stage,
        noises_stage,
        thresholds_stage,
        tol_stage,
        dav_max_iter_stage,
        dav_def_max_size_stage,
        dav_rel_conv_thrd_stage,
        dav_type_stage,
        cutoff_stage,
    ):
        return driver.dmrg(
            mpo,
            ket,
            n_sweeps=len(bond_dims_stage),
            bond_dims=bond_dims_stage,
            noises=noises_stage,
            thrds=thresholds_stage,
            tol=tol_stage,
            iprint=1,
            twosite_to_onesite=None,
            dav_max_iter=int(dav_max_iter_stage),
            dav_def_max_size=int(dav_def_max_size_stage),
            dav_rel_conv_thrd=float(dav_rel_conv_thrd_stage),
            dav_type=dav_type_stage,
            cutoff=float(cutoff_stage),
        )

    def _solve_pyblock2_manual_switch(
        driver,
        mpo,
        ket,
        *,
        bond_dims_full,
        noises_full,
        thresholds_full,
        first_n,
        tol_final,
        dav_max_iter_stage,
        dav_def_max_size_stage,
        dav_rel_conv_thrd_stage,
        dav_type_stage,
        cutoff_stage,
    ):
        driver.dmrg(
            mpo,
            ket,
            n_sweeps=-1,
            bond_dims=bond_dims_full,
            noises=noises_full,
            thrds=thresholds_full,
            tol=tol_final,
            iprint=1,
            twosite_to_onesite=None,
            dav_max_iter=int(dav_max_iter_stage),
            dav_def_max_size=int(dav_def_max_size_stage),
            dav_rel_conv_thrd=float(dav_rel_conv_thrd_stage),
            dav_type=dav_type_stage,
            cutoff=float(cutoff_stage),
        )
        dmrg = driver._dmrg
        bw = driver.bw
        dmrg.me.init_environments(False)
        forward = ket.center == 0
        energy = dmrg.solve(first_n, forward, 0.0, 0)
        dmrg.bond_dims = bw.b.VectorUBond(bond_dims_full[first_n:])
        dmrg.noises = bw.VectorFP(noises_full[first_n:])
        dmrg.davidson_conv_thrds = bw.VectorFP(thresholds_full[first_n:])
        dmrg.me.dot = 1
        for ext_me in dmrg.ext_mes:
            ext_me.dot = 1
        energy = dmrg.solve(len(bond_dims_full) - first_n, ket.center == 0, tol_final)
        ket.dot = 1
        ket.save_data()
        if driver.clean_scratch:
            dmrg.me.remove_partition_files()
            for me in dmrg.ext_mes:
                me.remove_partition_files()
        ket.info.bond_dim = max(ket.info.bond_dim, bond_dims_full[-1])
        return energy

    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SZ,
        n_threads=n_threads,
        stack_mem=stack_mem,
    )
    driver.read_fcidump(working_fcidump, iprint=1)
    driver.initialize_system(
        n_sites=fcidump_data.norb,
        n_elec=fcidump_data.nelec,
        spin=fcidump_data.ms2,
        orb_sym=getattr(driver, "orb_sym", None),
    )
    mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore)
    bond_dims, noises, thresholds = build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=bond_dim,
        convergence_tol=convergence_tol,
        n_sweeps=n_sweeps,
    )
    inferred_controls = (
        infer_pyblock2_benchmark_controls(
            bond_dim=bond_dim,
            convergence_tol=convergence_tol,
        )
        if str(schedule_mode).strip().lower() == "benchmark"
        else {
            "twosite_to_onesite": None,
            "dav_max_iter": 4000,
            "dav_def_max_size": 50,
            "dav_rel_conv_thrd": 0.0,
            "dav_type": None,
        }
    )
    if twosite_to_onesite is None:
        twosite_to_onesite = inferred_controls["twosite_to_onesite"]
    if dav_max_iter is None:
        dav_max_iter = int(inferred_controls["dav_max_iter"])
    if dav_def_max_size is None:
        dav_def_max_size = int(inferred_controls["dav_def_max_size"])
    if dav_rel_conv_thrd is None:
        dav_rel_conv_thrd = float(inferred_controls["dav_rel_conv_thrd"])
    if dav_type is None:
        dav_type = inferred_controls["dav_type"]
    dmrg_cutoff = 1e-14 if str(schedule_mode).strip().lower() == "benchmark" else 1e-20

    use_manual_switch = (
        twosite_to_onesite is not None
        and 0 < int(twosite_to_onesite) < len(bond_dims)
    )
    ket = driver.get_random_mps(
        tag="GS",
        bond_dim=bond_dims[0],
        nroots=1,
        dot=2,
        orig_dot=bool(use_manual_switch),
        full_fci=False,
    )
    capture = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        if use_manual_switch:
            first_n = int(twosite_to_onesite)
            energy = _solve_pyblock2_manual_switch(
                driver,
                mpo,
                ket,
                bond_dims_full=bond_dims,
                noises_full=noises,
                thresholds_full=thresholds,
                first_n=first_n,
                tol_final=convergence_tol,
                dav_max_iter_stage=dav_max_iter,
                dav_def_max_size_stage=dav_def_max_size,
                dav_rel_conv_thrd_stage=dav_rel_conv_thrd,
                dav_type_stage=dav_type,
                cutoff_stage=dmrg_cutoff,
            )
        else:
            energy = _solve_pyblock2_stage(
                driver,
                mpo,
                ket,
                bond_dims_stage=bond_dims,
                noises_stage=noises,
                thresholds_stage=thresholds,
                tol_stage=convergence_tol,
                dav_max_iter_stage=dav_max_iter,
                dav_def_max_size_stage=dav_def_max_size,
                dav_rel_conv_thrd_stage=dav_rel_conv_thrd,
                dav_type_stage=dav_type,
                cutoff_stage=dmrg_cutoff,
            )
    wall_time = time.time() - t0
    solver_output = capture.getvalue()
    if solver_output:
        print(solver_output, end="" if solver_output.endswith("\n") else "\n")
    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)

    s_squared = None
    try:
        ssq_mpo = driver.get_spin_square_mpo()
        s_squared = float(driver.expectation(ket, ssq_mpo, ket) / driver.get_mpo_amp())
    except Exception:
        s_squared = None

    try:
        driver.finalize()
    except Exception:
        pass

    total_energy = float(energy + fcidump_data.ecore)
    converged = "ATTENTION: DMRG is not converged" not in solver_output

    return {
        "method": "DMRG",
        "energy": total_energy,
        "correlation_energy": float(total_energy - mf.e_tot),
        "converged": converged,
        "s_squared": s_squared,
        "uhf_energy": float(mf.e_tot),
        "backend": "pyblock2_sz",
        "basis_mode": str(basis_mode),
        "bond_dim": int(bond_dim),
        "n_sweeps": int(n_sweeps),
        "schedule_mode": str(schedule_mode),
        "bond_dims": [int(x) for x in bond_dims],
        "noises": [float(x) for x in noises],
        "thresholds": [float(x) for x in thresholds],
        "wall_time_s": float(wall_time),
        "twosite_to_onesite": None if twosite_to_onesite is None else int(twosite_to_onesite),
        "dav_max_iter": int(dav_max_iter),
        "dav_def_max_size": int(dav_def_max_size),
        "dav_rel_conv_thrd": float(dav_rel_conv_thrd),
        "dav_type": None if dav_type is None else str(dav_type),
    }


def _run_dmrg_pyscf_dmrgci_sz(
    *,
    fcidump_data,
    working_fcidump: str,
    uhf_npz_path: str,
    basis_mode: str,
    bond_dim: int,
    n_sweeps: int,
    schedule_mode: str,
    scratch: str,
    twosite_to_onesite: int | None,
    dav_max_iter: int | None,
    dav_def_max_size: int | None,
    dav_rel_conv_thrd: float | None,
    dav_type: str | None,
):
    """Run one active-space DMRG solve via PySCF DMRGCI in the working basis."""
    from apex_cas.CAS_tester import _run_sz_dmrg

    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    capture = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        e_act, _dmrg_1rdm, _wall_time_inner, s_squared = _run_sz_dmrg(
            working_fcidump,
            bond_dim,
            fcidump_data.norb,
            fcidump_data.nelec,
            fcidump_data.ms2,
            fcidump_data.ecore,
            scratch,
            schedule_mode=schedule_mode,
            n_sweeps=n_sweeps,
            max_iter=n_sweeps,
        )
    wall_time = time.time() - t0
    solver_output = capture.getvalue()
    if solver_output:
        print(solver_output, end="" if solver_output.endswith("\n") else "\n")
    energy = float(e_act + fcidump_data.ecore)
    return {
        "method": "DMRG",
        "energy": energy,
        "correlation_energy": float(energy - mf.e_tot),
        "converged": "ATTENTION: DMRG is not converged" not in solver_output,
        "s_squared": None if s_squared is None else float(s_squared),
        "uhf_energy": float(mf.e_tot),
        "backend": "pyscf_dmrgci_sz",
        "basis_mode": str(basis_mode),
        "bond_dim": int(bond_dim),
        "n_sweeps": int(n_sweeps),
        "schedule_mode": str(schedule_mode),
        "bond_dims": [],
        "noises": [],
        "thresholds": [],
        "wall_time_s": float(wall_time),
        "twosite_to_onesite": None,
        "dav_max_iter": None,
        "dav_def_max_size": None,
        "dav_rel_conv_thrd": None,
        "dav_type": None,
    }


def _run_dmrg(
    *,
    backend: str,
    fcidump_path: str,
    uhf_npz_path: str,
    dmrg_basis_npz_path: str,
    basis_mode: str,
    bond_dim: int,
    n_sweeps: int,
    convergence_tol: float,
    schedule_mode: str,
    n_threads: int,
    stack_mem: int,
    scratch: str,
    twosite_to_onesite: int | None,
    dav_max_iter: int | None,
    dav_def_max_size: int | None,
    dav_rel_conv_thrd: float | None,
    dav_type: str | None,
):
    """Run one active-space DMRG solve via the selected backend."""
    fcidump_data, working_fcidump, cleanup_path = _prepare_working_fcidump(
        fcidump_path=fcidump_path,
        dmrg_basis_npz_path=dmrg_basis_npz_path,
        basis_mode=basis_mode,
        scratch=scratch,
    )
    try:
        if backend == "pyscf_dmrgci_sz":
            return _run_dmrg_pyscf_dmrgci_sz(
                fcidump_data=fcidump_data,
                working_fcidump=working_fcidump,
                uhf_npz_path=uhf_npz_path,
                basis_mode=basis_mode,
                bond_dim=bond_dim,
                n_sweeps=n_sweeps,
                schedule_mode=schedule_mode,
                scratch=scratch,
                twosite_to_onesite=twosite_to_onesite,
                dav_max_iter=dav_max_iter,
                dav_def_max_size=dav_def_max_size,
                dav_rel_conv_thrd=dav_rel_conv_thrd,
                dav_type=dav_type,
            )
        if backend != "pyblock2_sz":
            raise ValueError(f"Unsupported DMRG backend: {backend}")
        return _run_dmrg_pyblock2_sz(
            fcidump_data=fcidump_data,
            working_fcidump=working_fcidump,
            uhf_npz_path=uhf_npz_path,
            basis_mode=basis_mode,
            bond_dim=bond_dim,
            n_sweeps=n_sweeps,
            convergence_tol=convergence_tol,
            schedule_mode=schedule_mode,
            n_threads=n_threads,
            stack_mem=stack_mem,
            scratch=scratch,
            twosite_to_onesite=twosite_to_onesite,
            dav_max_iter=dav_max_iter,
            dav_def_max_size=dav_def_max_size,
            dav_rel_conv_thrd=dav_rel_conv_thrd,
            dav_type=dav_type,
        )
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except OSError:
                pass


def _parse_args():
    parser = argparse.ArgumentParser(description="APEX_Filter DMRG worker")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--fcidump-path", required=True)
    parser.add_argument("--uhf-npz-path", required=True)
    parser.add_argument("--dmrg-basis-npz-path", required=True)
    parser.add_argument("--basis-mode", required=True)
    parser.add_argument("--bond-dim", type=int, required=True)
    parser.add_argument("--n-sweeps", type=int, required=True)
    parser.add_argument("--convergence-tol", type=float, required=True)
    parser.add_argument("--schedule-mode", required=True)
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--stack-mem", type=int, required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--twosite-to-onesite", type=int, default=None)
    parser.add_argument("--dav-max-iter", type=int, default=None)
    parser.add_argument("--dav-def-max-size", type=int, default=None)
    parser.add_argument("--dav-rel-conv-thrd", type=float, default=None)
    parser.add_argument("--dav-type", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    result = _run_dmrg(
        backend=args.backend,
        fcidump_path=args.fcidump_path,
        uhf_npz_path=args.uhf_npz_path,
        dmrg_basis_npz_path=args.dmrg_basis_npz_path,
        basis_mode=args.basis_mode,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        convergence_tol=args.convergence_tol,
        schedule_mode=args.schedule_mode,
        n_threads=args.n_threads,
        stack_mem=args.stack_mem,
        scratch=args.scratch,
        twosite_to_onesite=args.twosite_to_onesite,
        dav_max_iter=args.dav_max_iter,
        dav_def_max_size=args.dav_def_max_size,
        dav_rel_conv_thrd=args.dav_rel_conv_thrd,
        dav_type=args.dav_type,
    )
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
