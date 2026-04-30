"""Alpha/beta orbital-pairing primitives."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def pair_alpha_beta_orbitals(mol, mo_alpha, mo_beta):
    """Find optimal alpha/beta orbital pairing by overlap maximization."""
    return pair_alpha_beta_orbitals_with_overlap(
        mol.intor_symmetric("int1e_ovlp"),
        mo_alpha,
        mo_beta,
    )


def pair_alpha_beta_orbitals_with_overlap(S_matrix, mo_alpha, mo_beta):
    """Pair alpha/beta orbitals given an AO overlap matrix."""
    overlap = mo_alpha.T @ S_matrix @ mo_beta
    row_ind, col_ind = linear_sum_assignment(-np.abs(overlap))
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def reorder_beta_to_match_alpha(pairs, mo_beta, n_orbitals):
    """Reorder beta orbitals to match alpha orbital ordering."""
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    beta_order = [p[1] for p in sorted_pairs[:n_orbitals]]
    return mo_beta[:, beta_order]

