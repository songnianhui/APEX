"""Shared primitives for active-space reference SCF containers and solvers."""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo, gto, scf


def _build_default_init_guess(mol_fake, norb):
    """Create a spin-balanced diagonal guess consistent with mol.spin."""
    dm = np.zeros((2, norb, norb))
    na = (mol_fake.nelectron + mol_fake.spin) // 2
    nb = mol_fake.nelectron - na

    for idx in range(min(na, norb)):
        dm[0, idx, idx] = 1.0
    for idx in range(min(nb, norb)):
        dm[1, idx, idx] = 1.0
    return dm


def _sanitize_ms2_for_nelec(nelec: int, ms2: int) -> int:
    """Clamp MS2 to a physically valid parity-compatible value for nelec."""
    ms2_int = int(round(ms2))
    max_ms2 = int(abs(nelec))
    if ms2_int > max_ms2:
        ms2_int = max_ms2
    if ms2_int < -max_ms2:
        ms2_int = -max_ms2
    if (nelec - ms2_int) % 2 != 0:
        if ms2_int > 0:
            ms2_int -= 1
        elif ms2_int < 0:
            ms2_int += 1
        else:
            ms2_int = 1 if nelec % 2 == 1 else 0
            if (nelec - ms2_int) % 2 != 0:
                ms2_int = -ms2_int
    return ms2_int


def build_fake_mol(norb: int, nelec: int, ms2: int, ecore: float = 0.0):
    """Build a minimal PySCF Mole container for active-space calculations."""
    mol = gto.M()
    mol.nelectron = nelec
    mol.spin = _sanitize_ms2_for_nelec(nelec, ms2)
    mol.incore_anyway = True
    mol._nao_nr = norb

    mol.atom = [["H", (0.0, 0.0, 0.0)]]
    mol.basis = "sto-3g"
    mol.build(False, False)

    mol.nao_nr = lambda *args, **kwargs: norb
    mol.nao = norb
    mol.energy_nuc = lambda *args, **kwargs: float(ecore)
    return mol


def build_reference_uhf_solver(fcidump_data, mol_fake, conv_tol=1e-8, max_cycle=2000):
    """Build a PySCF UHF object whose integrals are supplied by FCIDUMP."""
    norb = fcidump_data.norb
    h1e = fcidump_data.h1e
    h2e = fcidump_data.h2e

    mf = scf.UHF(mol_fake)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.get_hcore = lambda *args, **kwargs: h1e.copy()
    mf.get_ovlp = lambda *args, **kwargs: np.eye(norb)
    mf.get_init_guess = lambda *args, **kwargs: _build_default_init_guess(mol_fake, norb)

    if h2e.ndim == 4:
        mf._eri = ao2mo.restore(8, h2e, norb)
    elif h2e.ndim in (1, 2):
        mf._eri = h2e
    else:
        mf._eri = ao2mo.restore(8, h2e, norb)

    return mf
