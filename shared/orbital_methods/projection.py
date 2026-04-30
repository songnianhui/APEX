"""Projection-weight primitives."""

from __future__ import annotations

import numpy as np


def compute_projection_weights_for_targets(mol, mo_coeff, target_ao_indices):
    """Compute projection weight of every MO onto a given AO subset."""
    n_mo = mo_coeff.shape[1]
    if not target_ao_indices:
        return np.zeros(n_mo)

    target_ao = np.array(target_ao_indices)
    S = mol.intor_symmetric("int1e_ovlp")
    projections = np.zeros(n_mo)
    for i in range(n_mo):
        mo_i = mo_coeff[:, i]
        for ao_j in target_ao:
            e_j = np.zeros(mol.nao_nr())
            e_j[ao_j] = 1.0
            projections[i] += abs(np.dot(mo_i, np.dot(S, e_j))) ** 2
    return projections
