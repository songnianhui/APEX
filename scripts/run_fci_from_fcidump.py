#!/usr/bin/env python3
"""Standalone FCI solver: read FCIDUMP -> solve CASCI -> compare with Chan.

Usage:
    python run_fci_from_fcidump.py [fcidump_path] [chan_ecore]

Arguments:
    fcidump_path  Path to our FCIDUMP file.
                  Default: examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh
    chan_ecore    Chan's E_core for comparison.
                  Default: -4976.26532397 (Fe2S2)
"""
import sys
import os
import time
import numpy as np
from pyscf import fci, ao2mo
from pyscf.tools import fcidump as fd_mod

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
chan_ecore   = float(sys.argv[2]) if len(sys.argv) > 2 else CHAN_E_CORE

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

print(f"  NORB={norb}, NELEC={nelec}, MS2={ms2}")
print(f"  nalpha={nalpha}, nbeta={nbeta}")

# Read real ecore from sidecar file
ecore_path = fcidump_path + ".ecore"
if os.path.isfile(ecore_path):
    with open(ecore_path) as f:
        ecore_real = float(f.read().strip())
    print(f"  E_core (sidecar): {ecore_real:.12f}")
else:
    ecore_real = ecore_in_file
    print(f"  E_core (FCIDUMP): {ecore_in_file:.12f}")

# ── Convert h2e to 4-index (no symmetry) ─────────────────────────
print("Converting h2e to 4-index ...")
eri = ao2mo.restore(1, h2e_8fold, norb)

# ── FCI dimension estimate ────────────────────────────────────────
from scipy.special import comb
ndet_a = int(comb(norb, nalpha, exact=True))
ndet_b = int(comb(norb, nbeta, exact=True))
ndet = ndet_a * ndet_b
mem_gb = ndet * 8 / 1e9
print(f"\nFCI dimension: C({norb},{nalpha}) x C({norb},{nbeta}) = "
      f"{ndet_a} x {ndet_b} = {ndet:,}")
print(f"CI vector memory estimate: {mem_gb:.2f} GB")

# ── Run FCI ───────────────────────────────────────────────────────
print(f"\nStarting FCI solver for CAS({nelec}e, {norb}o) ...")
print("=" * 60)

fs = fci.direct_spin1.FCI()
fs.max_cycle = 200
fs.conv_tol = 1e-8
fs.verbose = 5  # prints davidson iteration info

t0 = time.time()
e_act, ci_vec = fs.kernel(h1e, eri, norb, (nalpha, nbeta))
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
