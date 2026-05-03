"""Shared molecule-construction helpers for staged APEX workflows."""

from __future__ import annotations

from pyscf import gto

from .models import ComputationSettings as _ComputationSettings
from .setting_utils import build_basis_dict as _build_basis_dict


def build_mol_with_basis(cluster_info, settings: _ComputationSettings):
    """Build a PySCF Mole object with per-element basis support."""
    atoms = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        atoms.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    spin_2s = int(round(2 * cluster_info.target_spin))
    basis_dict = _build_basis_dict(cluster_info, settings)

    return gto.M(
        atom="\n".join(atoms),
        charge=cluster_info.total_charge,
        spin=spin_2s,
        basis=basis_dict,
        symmetry=False,
        verbose=1,
    )
