"""Shared DMRG solver primitives used by staged APEX workflows."""

from __future__ import annotations

import multiprocessing
import os
import time

import numpy as np
from pyscf import ao2mo, dmrgscf, gto, scf
from pyscf.mcscf import casci as casci_mod
from pyscf.tools import fcidump as fd_mod

from .dmrg_controls import (
    build_dmrg_sweep_schedule as _build_dmrg_sweep_schedule,
    compress_dmrg_schedule_for_dmrgci as _compress_dmrg_schedule_for_dmrgci,
)


def resolve_dmrgci_twodot_to_onedot(default_switch: int | None, max_iter: int) -> int:
    """Return a BLOCK-safe twodot_to_onedot sweep index."""
    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if default_switch is None:
        return 0
    switch = int(default_switch)
    if switch == 0:
        return 0
    if switch >= max_iter:
        return max_iter - 1
    return switch


def run_sz_dmrg(
    fcidump_path,
    bond_dim,
    norb,
    nelec,
    ms2,
    ecore_real,
    dmrg_tmpdir,
    *,
    schedule_mode="benchmark",
    n_sweeps=None,
    max_iter=None,
):
    """Run SZ DMRG via PySCF DMRGCI and return active energy, 1-RDM, wall time, S^2."""
    data = fd_mod.read(fcidump_path, verbose=False)
    h1e = data["H1"]
    h2e_8fold = data["H2"]

    nalpha = (nelec + ms2) // 2
    nbeta = (nelec - ms2) // 2

    dummy_mol = gto.M()
    dummy_mol.nelectron = nelec
    dummy_mol.incore_anyway = True
    dummy_mol.spin = ms2
    dummy_mol.verbose = 4

    dummy_mf = scf.RHF(dummy_mol)
    dummy_mf.incore_anyway = True
    dummy_mf.verbose = 4
    dummy_mf.mo_coeff = np.eye(norb)
    dummy_mf.mo_occ = np.zeros(norb)
    dummy_mf.mo_occ[:nelec] = 2.0

    mc = casci_mod.CASCI(dummy_mf, norb, (nalpha, nbeta), ncore=0)
    mc.mo_coeff = np.eye(norb)

    eri_4idx = ao2mo.restore(1, h2e_8fold, norb)
    mc.fcisolver = dmrgscf.DMRGCI(dummy_mol, maxM=bond_dim, tol=1e-8)
    mc.fcisolver.memory = 8
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count()))
    mc.fcisolver.runtimeDir = dmrg_tmpdir
    mc.fcisolver.scratchDirectory = dmrg_tmpdir
    bond_dims, noises, thrds = _build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=bond_dim,
        convergence_tol=1e-8,
        n_sweeps=n_sweeps,
    )
    schedule_sweeps, schedule_max_ms, schedule_tols, schedule_noises = _compress_dmrg_schedule_for_dmrgci(
        bond_dims,
        noises,
        thrds,
    )
    mc.fcisolver.scheduleSweeps = schedule_sweeps
    mc.fcisolver.scheduleMaxMs = schedule_max_ms
    mc.fcisolver.scheduleTols = schedule_tols
    mc.fcisolver.scheduleNoises = schedule_noises
    effective_max_iter = int(max_iter) if max_iter is not None else len(bond_dims)
    mc.fcisolver.maxIter = effective_max_iter
    mc.fcisolver.twodot_to_onedot = resolve_dmrgci_twodot_to_onedot(
        getattr(mc.fcisolver, "twodot_to_onedot", 0),
        effective_max_iter,
    )

    t0 = time.time()
    result = mc.fcisolver.kernel(h1e, eri_4idx, norb, (nalpha, nbeta))
    e_act = result[0] if isinstance(result, tuple) else result
    t1 = time.time()

    dmrg_1rdm = mc.fcisolver.make_rdm1(0, norb, (nalpha, nbeta))
    return e_act, dmrg_1rdm, t1 - t0, None


def run_su2_dmrg(
    fcidump_path,
    bond_dim,
    norb,
    nelec,
    ms2,
    ecore_real,
    dmrg_tmpdir,
    *,
    stack_mem_gb=None,
    schedule_mode="benchmark",
    n_sweeps=None,
):
    """Run SU2 DMRG via pyblock2 and return active energy, 1-RDM, wall time, S^2."""
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError(
            "pyblock2 is required for SU2 DMRG calculations. "
            "Install it with: pip install pyblock2"
        )

    if stack_mem_gb is None:
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / 1024**3
            stack_mem_gb = max(1.0, min(avail_gb * 0.7, 12.0))
        except ImportError:
            stack_mem_gb = 4.0
    stack_mem_bytes = int(stack_mem_gb * 1024**3)

    spin = ms2
    driver = DMRGDriver(
        scratch=dmrg_tmpdir,
        symm_type=SymmetryTypes.SU2,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count())),
        stack_mem=stack_mem_bytes,
    )
    driver.read_fcidump(fcidump_path, iprint=1)
    driver.initialize_system(n_sites=norb, n_elec=nelec, spin=spin)

    mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=0.0)
    bond_dims, noises, thrds = _build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=bond_dim,
        convergence_tol=1e-8,
        n_sweeps=n_sweeps,
    )

    ket = driver.get_random_mps(tag="GS", bond_dim=bond_dims[0], nroots=1)
    t0 = time.time()
    e_dmrg = driver.dmrg(
        mpo,
        ket,
        n_sweeps=len(bond_dims),
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        tol=1e-8,
        iprint=1,
    )
    t1 = time.time()

    e_act = e_dmrg - driver.ecore
    dmrg_1rdm = driver.get_1pdm(ket)

    try:
        ssq_mpo = driver.get_spin_square_mpo()
        ssq_value = driver.expectation(ket, ssq_mpo, ket) / driver.get_mpo_amp()
    except Exception:
        ssq_value = None

    driver.finalize()
    return e_act, dmrg_1rdm, t1 - t0, ssq_value
