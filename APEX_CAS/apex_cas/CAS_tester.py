"""DMRG solver module: FCIDUMP → DMRG → 1-RDM → NOON → save results → plot.

Provides `run_dmrg_test()` as the main entry point, implementing
Step 6-9 of the APEX_CAS pipeline:

  Step 6  — DMRG solve on FCIDUMP integrals (block2 + pyscf.dmrgscf / pyblock2)
  Step 7  — Extract 1-RDM from DMRG wavefunction
  Step 8  — Diagonalize 1-RDM → natural occupation numbers (NOON)
  Step 9  — Generate NOON bar plot via orbital_visualizer

Supports two symmetry channels:
  - SZ (default): pyscf.dmrgscf.DMRGCI, determinant basis
  - SU2: pyblock2 DMRGDriver, spin-adapted CSF basis
"""

import logging
import multiprocessing
import os
import shutil
import tempfile
import time

import numpy as np

from shared.dmrg_controls import (
    build_dmrg_sweep_schedule,
    compress_dmrg_schedule_for_dmrgci,
)

logger = logging.getLogger(__name__)


def _resolve_dmrgci_twodot_to_onedot(default_switch: int | None, max_iter: int) -> int:
    """Return a BLOCK-safe twodot_to_onedot sweep index.

    PySCF's DMRGCI defaults may set ``twodot_to_onedot`` equal to the default
    ``maxIter`` at high bond dimensions (for example ``30`` when ``maxM>=2000``).
    This is valid for the default PySCF schedule because its ``maxIter`` is
    larger (e.g. ``38``), but it becomes invalid if APEX shortens the sweep
    count to exactly ``30``. BLOCK requires ``len(schedule) > twodot_to_onedot``.
    """
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


def _run_sz_dmrg(
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
    """SZ DMRG via pyscf.dmrgscf.DMRGCI (original path).

    Returns
    -------
    tuple
        (e_act, dmrg_1rdm, wall_time, spin_squared)
        spin_squared is always None for the SZ path.
    """
    from pyscf import ao2mo, gto, scf
    from pyscf import dmrgscf
    from pyscf.mcscf import casci as casci_mod
    from pyscf.tools import fcidump as fd_mod

    # Read FCIDUMP (SZ path needs pyscf format)
    data = fd_mod.read(fcidump_path, verbose=False)
    h1e = data["H1"]
    h2e_8fold = data["H2"]

    nalpha = (nelec + ms2) // 2
    nbeta = (nelec - ms2) // 2

    # Build dummy mol + mf for CASCI
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
    bond_dims, noises, thrds = build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=bond_dim,
        convergence_tol=1e-8,
        n_sweeps=n_sweeps,
    )
    schedule_sweeps, schedule_max_ms, schedule_tols, schedule_noises = (
        compress_dmrg_schedule_for_dmrgci(bond_dims, noises, thrds)
    )
    mc.fcisolver.scheduleSweeps = schedule_sweeps
    mc.fcisolver.scheduleMaxMs = schedule_max_ms
    mc.fcisolver.scheduleTols = schedule_tols
    mc.fcisolver.scheduleNoises = schedule_noises
    effective_max_iter = int(max_iter) if max_iter is not None else len(bond_dims)
    mc.fcisolver.maxIter = effective_max_iter
    mc.fcisolver.twodot_to_onedot = _resolve_dmrgci_twodot_to_onedot(
        getattr(mc.fcisolver, "twodot_to_onedot", 0),
        effective_max_iter,
    )

    # Run DMRG
    t0 = time.time()
    result = mc.fcisolver.kernel(h1e, eri_4idx, norb, (nalpha, nbeta))
    e_act = result[0] if isinstance(result, tuple) else result
    t1 = time.time()

    # Extract 1-RDM
    dmrg_1rdm = mc.fcisolver.make_rdm1(0, norb, (nalpha, nbeta))

    return e_act, dmrg_1rdm, t1 - t0, None  # (e_act, 1rdm, wall_time, spin_squared)


def _run_su2_dmrg(fcidump_path, bond_dim, norb, nelec, ms2, ecore_real, dmrg_tmpdir,
                  stack_mem_gb=None, schedule_mode="benchmark", n_sweeps=None):
    """SU2 DMRG via pyblock2 DMRGDriver with SymmetryTypes.SU2.

    Returns
    -------
    tuple
        (e_act, dmrg_1rdm, wall_time, spin_squared)
        dmrg_1rdm is spin-traced [norb, norb].
        spin_squared is <S²> or None if computation fails.
    """
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError(
            "pyblock2 is required for SU2 DMRG calculations. "
            "Install it with: pip install pyblock2"
        )

    # Auto-detect stack memory: use 70% of available RAM if not specified
    if stack_mem_gb is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1024**3
            stack_mem_gb = max(1.0, min(avail_gb * 0.7, 12.0))
        except ImportError:
            stack_mem_gb = 4.0
    stack_mem_bytes = int(stack_mem_gb * 1024**3)

    spin = ms2  # total spin S, initialize_system(spin=2*S)

    driver = DMRGDriver(
        scratch=dmrg_tmpdir,
        symm_type=SymmetryTypes.SU2,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count())),
        stack_mem=stack_mem_bytes,
    )

    # Read FCIDUMP (pyblock2 auto-converts to SU2 internal format)
    driver.read_fcidump(fcidump_path, iprint=1)

    # Override system info from FCIDUMP header
    driver.initialize_system(n_sites=norb, n_elec=nelec, spin=spin)

    # Build MPO (ecore=0 because we handle it in the caller)
    mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=0.0)

    bond_dims, noises, thrds = build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=bond_dim,
        convergence_tol=1e-8,
        n_sweeps=n_sweeps,
    )

    # Run DMRG
    ket = driver.get_random_mps(tag="GS", bond_dim=bond_dims[0], nroots=1)
    t0 = time.time()
    e_dmrg = driver.dmrg(
        mpo, ket,
        n_sweeps=len(bond_dims),
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        tol=1e-8,
        iprint=1,
    )
    t1 = time.time()

    # pyblock2 dmrg() returns e_active + ecore; subtract driver.ecore
    e_act = e_dmrg - driver.ecore

    # Extract spin-traced 1-RDM [norb, norb]
    dmrg_1rdm = driver.get_1pdm(ket)

    # Compute <S²>
    try:
        ssq_mpo = driver.get_spin_square_mpo()
        ssq_value = driver.expectation(ket, ssq_mpo, ket) / driver.get_mpo_amp()
    except Exception:
        ssq_value = None

    driver.finalize()

    return e_act, dmrg_1rdm, t1 - t0, ssq_value


def run_dmrg_test(
    fcidump_path: str,
    bond_dim: int = 500,
    output_dir: str = None,
    label: str = None,
    symm_type: str = "sz",
    stack_mem_gb: float = None,
    dmrg_mode: str = "benchmark",
    n_sweeps: int | None = None,
    max_iter: int | None = None,
) -> dict:
    """Run DMRG on an FCIDUMP file and return results including NOON.

    Pipeline: FCIDUMP → DMRG → 1-RDM → NOON → save results → plot

    Parameters
    ----------
    fcidump_path : str
        Absolute path to the FCIDUMP file.
    bond_dim : int
        Maximum bond dimension M for DMRG (default 500).
    output_dir : str or None
        Output directory for results.  Defaults to the FCIDUMP file's
        parent directory.
    label : str or None
        Label for output file naming.  Derived from the FCIDUMP filename
        if not provided.
    symm_type : str
        DMRG symmetry type: "sz" (default, determinant basis) or
        "su2" (spin-adapted CSF basis via pyblock2).
    stack_mem_gb : float or None
        Stack memory in GB for pyblock2 DMRG tensor operations.
        None = auto-detect (70% of available RAM, capped at 12 GB).
    dmrg_mode : str
        DMRG schedule profile: ``"workflow"`` or ``"benchmark"``.
    n_sweeps : int or None
        Optional override for total sweeps.
    max_iter : int or None
        Optional override for PySCF DMRGCI maxIter on the SZ path.

    Returns
    -------
    dict
        Keys: ``e_active``, ``e_core``, ``e_total``, ``noon``,
        ``dmrg_1rdm``, ``bond_dim``, ``n_orb``, ``n_elec``,
        ``h5_path``, ``noon_plot_path``, ``spin_squared`` (SU2 only).
    """
    # block2 must be imported before pyscf to avoid module conflicts
    try:
        import block2  # noqa: F401
    except ImportError:
        raise ImportError(
            "block2 is required for DMRG calculations. "
            "Install it with: pip install block2"
        )

    # ── Derive output paths ──────────────────────────────────
    if output_dir is None:
        output_dir = os.path.dirname(fcidump_path)
    dmrg_dir = os.path.join(output_dir, "dmrg")
    os.makedirs(dmrg_dir, exist_ok=True)

    fcidump_stem = os.path.basename(fcidump_path)
    # Strip "FCIDUMP." prefix if present
    if fcidump_stem.upper().startswith("FCIDUMP"):
        stem = label or fcidump_stem[len("FCIDUMP"):]
        stem = stem.lstrip(".")
    else:
        stem = label or os.path.splitext(fcidump_stem)[0]

    if not stem:
        stem = "dmrg"

    artifact_stem = f"{stem}_{str(symm_type).strip().lower()}_M{int(bond_dim)}"

    # ── Step 6a: Read FCIDUMP header info ─────────────────────
    from pyscf.tools import fcidump as fd_mod

    print(f"Reading FCIDUMP: {fcidump_path}")
    data = fd_mod.read(fcidump_path, verbose=False)

    h1e = data["H1"]
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data["MS2"])
    ecore_in_file = float(data["ECORE"])

    nalpha = (nelec + ms2) // 2
    nbeta = (nelec - ms2) // 2
    print(f"  NORB={norb}, NELEC={nelec}, MS2={ms2}, "
          f"(nalpha, nbeta)=({nalpha}, {nbeta})")

    # Read real ecore from sidecar file
    ecore_path = fcidump_path + ".ecore"
    if os.path.isfile(ecore_path):
        with open(ecore_path) as f:
            ecore_real = float(f.read().strip())
        print(f"  E_core (sidecar): {ecore_real:.12f}")
    else:
        ecore_real = ecore_in_file
        print(f"  E_core (FCIDUMP): {ecore_in_file:.12f}")

    # ── Step 6b+6c: Run DMRG solver ───────────────────────────
    dmrg_tmpdir = tempfile.mkdtemp(prefix="apex_dmrg_")
    print(f"\nStarting DMRG solver ({symm_type.upper()}): "
          f"CAS({nelec}e, {norb}o), M={bond_dim}")
    print("=" * 60)

    try:
        if symm_type == "su2":
            e_act, dmrg_1rdm, wall_time, spin_squared = _run_su2_dmrg(
                fcidump_path, bond_dim, norb, nelec, ms2, ecore_real, dmrg_tmpdir,
                stack_mem_gb=stack_mem_gb,
                schedule_mode=dmrg_mode,
                n_sweeps=n_sweeps,
            )
        else:
            e_act, dmrg_1rdm, wall_time, spin_squared = _run_sz_dmrg(
                fcidump_path, bond_dim, norb, nelec, ms2, ecore_real, dmrg_tmpdir,
                schedule_mode=dmrg_mode,
                n_sweeps=n_sweeps,
                max_iter=max_iter,
            )
    finally:
        # Clean up DMRG scratch directory
        try:
            shutil.rmtree(dmrg_tmpdir, ignore_errors=True)
            logger.info("Cleaned up DMRG scratch directory: %s", dmrg_tmpdir)
        except Exception:
            pass

    print(f"  DMRG completed in {wall_time:.1f} s")

    # ── Step 7: NOON from 1-RDM ──────────────────────────────
    print("\nExtracting 1-RDM from DMRG wavefunction ...")
    eigvals, eigvecs = np.linalg.eigh(dmrg_1rdm)
    noon = eigvals[::-1]  # descending order

    print(f"  NOON range: [{noon.min():.6f}, {noon.max():.6f}]")
    n_strongly_correlated = int(np.sum((noon > 0.02) & (noon < 1.98)))
    print(f"  Strongly correlated orbitals: {n_strongly_correlated}")

    # Build orbital labels
    noon_labels = [f"orb_{i}" for i in range(norb)]

    # ── Compute total energy ─────────────────────────────────
    e_total = e_act + ecore_real

    # ── Step 8b: Save results to HDF5 ────────────────────────
    import h5py

    h5_path = os.path.join(dmrg_dir, f"{artifact_stem}_dmrg_results.h5")
    h5_kwargs = dict(compression="gzip", compression_opts=9)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("dmrg_1rdm", data=dmrg_1rdm, **h5_kwargs)
        f.create_dataset("noon", data=noon, **h5_kwargs)
        f.create_dataset("noon_labels", data=[str(l) for l in noon_labels])

        meta = f.create_group("metadata")
        meta.attrs["e_active"] = e_act
        meta.attrs["e_core"] = ecore_real
        meta.attrs["e_total"] = e_total
        meta.attrs["bond_dim"] = bond_dim
        meta.attrs["n_orb"] = norb
        meta.attrs["n_elec"] = nelec
        meta.attrs["nalpha"] = nalpha
        meta.attrs["nbeta"] = nbeta
        meta.attrs["ms2"] = ms2
        meta.attrs["fcidump_path"] = fcidump_path
        meta.attrs["wall_time_s"] = wall_time
        meta.attrs["symm_type"] = symm_type
        if spin_squared is not None:
            meta.attrs["spin_squared"] = spin_squared

    print(f"  Results saved to: {h5_path}")

    # ── Step 9: Generate NOON plot ───────────────────────────
    from .orbital_visualizer import generate_noon_plot
    from .state_io import save_dmrg_summary

    noon_path = os.path.join(dmrg_dir, f"{artifact_stem}_noon_plot.png")
    noon_plot_path = generate_noon_plot(
        occupations=noon,
        labels=noon_labels,
        output_path=noon_path,
    )
    save_dmrg_summary(
        dmrg_dir,
        artifact_stem,
        fcidump_path=fcidump_path,
        h5_path=h5_path,
        noon_plot_path=noon_plot_path,
        bond_dim=bond_dim,
        symm_type=symm_type,
        n_orb=norb,
        n_elec=nelec,
        ms2=ms2,
        e_active=e_act,
        e_core=ecore_real,
        e_total=e_total,
        wall_time_s=wall_time,
        spin_squared=spin_squared,
        settings_payload={
            "bond_dim": int(bond_dim),
            "symm_type": symm_type,
            "stack_mem_gb": stack_mem_gb,
            "dmrg_mode": dmrg_mode,
            "n_sweeps": n_sweeps,
            "max_iter": max_iter,
        },
    )

    # ── Build return dict ────────────────────────────────────
    results = {
        "e_active": e_act,
        "e_core": ecore_real,
        "e_total": e_total,
        "noon": noon,
        "dmrg_1rdm": dmrg_1rdm,
        "bond_dim": bond_dim,
        "n_orb": norb,
        "n_elec": nelec,
        "h5_path": h5_path,
        "noon_plot_path": noon_plot_path,
        "wall_time": wall_time,
        "spin_squared": spin_squared,
    }

    return results
