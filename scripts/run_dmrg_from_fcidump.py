#!/usr/bin/env python3
"""Standalone DMRG solver: read FCIDUMP -> solve active space via block2 DMRG.

Usage:
    python run_dmrg_from_fcidump.py [fcidump_path] [bond_dim] [chan_ecore]

Arguments:
    fcidump_path  Path to our FCIDUMP file.
                  Default: examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh
    bond_dim      Maximum bond dimension M for DMRG. Default: 2000
    chan_ecore    Chan's E_core for comparison.
                  Default: -4976.26532397 (Fe2S2)

Examples:
    # Quick test with small bond dimension
    python run_dmrg_from_fcidump.py  500

    # Production run
    python run_dmrg_from_fcidump.py  4000

    # Custom FCIDUMP path
    python run_dmrg_from_fcidump.py /path/to/FCIDUMP 2000 -4976.26532397
"""
import sys
import os
import time

# block2 must be imported before pyscf to avoid module conflicts
import block2  # noqa: F401

import numpy as np
from pyscf import ao2mo, lib
from pyscf.tools import fcidump as fd_mod
from pyscf import dmrgscf
from pyscf.mcscf import casci as casci_mod

# ── Defaults (Fe2S2) ──────────────────────────────────────────────
CHAN_E_ACT   = -116.6056091
CHAN_E_CORE  = -4976.26532397
CHAN_E_TOTAL = CHAN_E_ACT + CHAN_E_CORE

DEFAULT_FCIDUMP = (
    "examples/fe2s2/outputs/fcidump/"
    "FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh"
)

# ── Parse args ────────────────────────────────────────────────────
fcidump_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FCIDUMP
bond_dim     = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
chan_ecore   = float(sys.argv[3]) if len(sys.argv) > 3 else CHAN_E_CORE

# ── Read FCIDUMP ──────────────────────────────────────────────────
print(f"Reading FCIDUMP: {fcidump_path}")
data = fd_mod.read(fcidump_path, verbose=False)

h1e = data["H1"]
h2e_8fold = data["H2"]
norb = int(data["NORB"])
nelec = int(data["NELEC"])
ms2 = int(data["MS2"])
ecore_in_file = float(data["ECORE"])

nalpha = (nelec + ms2) // 2
nbeta  = (nelec - ms2) // 2

print(f"  NORB={norb}, NELEC={nelec}, MS2={ms2}, (nalpha, nbeta)=({nalpha}, {nbeta})")

# Read real ecore from sidecar file
ecore_path = fcidump_path + ".ecore"
if os.path.isfile(ecore_path):
    with open(ecore_path) as f:
        ecore_real = float(f.read().strip())
    print(f"  E_core (sidecar): {ecore_real:.12f}")
else:
    ecore_real = ecore_in_file
    print(f"  E_core (FCIDUMP): {ecore_in_file:.12f}")

# ── Build a dummy mol + mf for CASCI ─────────────────────────────
# DMRGCI needs a CASCI-compatible object. We construct a minimal
# mol/mf pair whose only purpose is to carry the integrals.
from pyscf import gto, scf

# Dummy molecule: 1 atom, e- = nelec, same norb
# The actual integrals will be overridden in CASCI
dummy_mol = gto.M()
dummy_mol.nelectron = nelec
dummy_mol.incore_anyway = True
dummy_mol.spin = ms2
dummy_mol.verbose = 4

# Dummy RHF
dummy_mf = scf.RHF(dummy_mol)
dummy_mf.incore_anyway = True
dummy_mf.verbose = 4

# Dummy MO coefficients: identity matrix (integrals are already in MO basis)
dummy_mf.mo_coeff = np.eye(norb)
dummy_mf.mo_occ = np.zeros(norb)
dummy_mf.mo_occ[:nelec] = 2.0  # closed-shell guess (approximate)

# ── Build CASCI with dummy objects ────────────────────────────────
# ncore=0 because FCIDUMP integrals already have core folded in
mc = casci_mod.CASCI(dummy_mf, norb, (nalpha, nbeta), ncore=0)
mc.mo_coeff = np.eye(norb)

# Override integrals with FCIDUMP data
eri_4idx = ao2mo.restore(1, h2e_8fold, norb)
mc.fcisolver = dmrgscf.DMRGCI(dummy_mol, maxM=bond_dim, tol=1e-8)
mc.fcisolver.memory = 8  # GB
mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
mc.fcisolver.runtimeDir = lib.param.TMPDIR
mc.fcisolver.scratchDirectory = lib.param.TMPDIR

print(f"\nStarting DMRG solver: CAS({nelec}e, {norb}o), M={bond_dim}")
print("=" * 60)

t0 = time.time()

# Use get_h1eff/get_h2eff with overridden integrals
# Manually compute E_active via DMRG on the FCIDUMP integrals
from pyscf import fci as pyscf_fci

# DMRGCI.kernel expects (h1e, h2e, norb, nelec)
# We feed it directly from FCIDUMP
result = mc.fcisolver.kernel(h1e, eri_4idx, norb, (nalpha, nbeta))
e_act = result[0] if isinstance(result, tuple) else result

t1 = time.time()

# ── Results ───────────────────────────────────────────────────────
e_total = e_act + ecore_real

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  E_active  = {e_act:.12f} Hartree")
print(f"  E_core    = {ecore_real:.12f} Hartree")
print(f"  E_total   = {e_total:.12f} Hartree")
print(f"  Wall time = {t1 - t0:.1f} s")

# ── Compare with Chan ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON WITH CHAN")
print("=" * 60)
print(f"  {'':>12} {'Chan':>20} {'Ours':>20} {'Diff':>20}")
print(f"  {'E_active':>12} {CHAN_E_ACT:>20.12f} {e_act:>20.12f} {e_act - CHAN_E_ACT:>+20.12f}")
print(f"  {'E_core':>12} {chan_ecore:>20.12f} {ecore_real:>20.12f} {ecore_real - chan_ecore:>+20.12f}")
print(f"  {'E_total':>12} {CHAN_E_TOTAL:>20.12f} {e_total:>20.12f} {e_total - CHAN_E_TOTAL:>+20.12f}")
