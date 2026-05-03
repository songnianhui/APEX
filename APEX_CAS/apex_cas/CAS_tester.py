"""Internal DMRG/testcas helpers for the staged APEX_CAS workflow.

Provides `_run_dmrg_test()` as the canonical internal entry point, implementing
the FCIDUMP→DMRG→1-RDM→NOON→plot chain behind the ``apex-cas testcas``
workflow:

  Step 6  — DMRG solve on FCIDUMP integrals (block2 + pyscf.dmrgscf / pyblock2)
  Step 7  — Extract 1-RDM from DMRG wavefunction
  Step 8  — Diagonalize 1-RDM → natural occupation numbers (NOON)
  Step 9  — Generate NOON bar plot via orbital_visualizer

Supports two symmetry channels:
  - SZ (default): pyscf.dmrgscf.DMRGCI, determinant basis
  - SU2: pyblock2 DMRGDriver, spin-adapted CSF basis
"""

import logging
import os
import shutil
import tempfile

import h5py
import numpy as np
from pyscf.tools import fcidump as fd_mod

from .orbital_visualizer import _generate_noon_plot
from .state_io import _save_dmrg_summary
from shared.dmrg_solvers import run_sz_dmrg as _run_sz_dmrg, run_su2_dmrg as _run_su2_dmrg
from shared.settings_payloads import build_base_settings_payload as _build_base_settings_payload

logger = logging.getLogger(__name__)

def _run_dmrg_test(
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
    print(f"Reading FCIDUMP: {fcidump_path}")
    data = fd_mod.read(fcidump_path, verbose=False)

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
    h5_path = os.path.join(dmrg_dir, f"{artifact_stem}_dmrg_results.h5")
    h5_kwargs = dict(compression="gzip", compression_opts=9)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("dmrg_1rdm", data=dmrg_1rdm, **h5_kwargs)
        f.create_dataset("noon", data=noon, **h5_kwargs)
        f.create_dataset("noon_labels", data=[str(label) for label in noon_labels])

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
    noon_path = os.path.join(dmrg_dir, f"{artifact_stem}_noon_plot.png")
    noon_plot_path = _generate_noon_plot(
        occupations=noon,
        labels=noon_labels,
        output_path=noon_path,
    )
    _save_dmrg_summary(
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
        settings_payload=_build_base_settings_payload(
            None,
            bond_dim=int(bond_dim),
            symm_type=symm_type,
            stack_mem_gb=stack_mem_gb,
            dmrg_mode=dmrg_mode,
            n_sweeps=n_sweeps,
            max_iter=max_iter,
        ),
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
