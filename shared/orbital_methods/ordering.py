"""Orbital-ordering primitives."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def fiedler_ordering(interaction_matrix):
    """Compute Fiedler ordering for orbital pairs."""
    n = interaction_matrix.shape[0]
    if n <= 2:
        return list(range(n))

    W = np.abs(np.asarray(interaction_matrix, dtype=float))
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vec = eigenvectors[:, 1] if len(eigenvalues) > 1 else eigenvectors[:, 0]
    return np.argsort(fiedler_vec).tolist()


def compute_ordering_matrix(
    mol,
    mo_coeff,
    *,
    mode: str = "exchange_proxy",
    exchange_proxy_max_orbitals: int = 64,
):
    """Build the matrix used to optimize DMRG chain ordering."""
    if mode == "exchange_proxy":
        nmo = mo_coeff.shape[1]
        if nmo > exchange_proxy_max_orbitals:
            logger.warning(
                "exchange_proxy ordering matrix not implemented beyond %d orbitals; "
                "falling back to overlap_proxy for nmo=%d",
                exchange_proxy_max_orbitals,
                nmo,
            )
            return compute_overlap_proxy_matrix(mol, mo_coeff)
        return compute_exchange_proxy_matrix(mol, mo_coeff)

    if mode == "overlap_proxy":
        return compute_overlap_proxy_matrix(mol, mo_coeff)

    raise ValueError(f"Unsupported DMRG ordering matrix mode: {mode}")


def compute_exchange_proxy_matrix(mol, mo_coeff):
    """Compute the explicit exchange-proxy matrix K_ij = |(ii|jj)|."""
    from pyscf import ao2mo

    nmo = mo_coeff.shape[1]
    eri_mo = ao2mo.full(mol, mo_coeff, compact=False)
    eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
    matrix = np.zeros((nmo, nmo), dtype=float)
    for i in range(nmo):
        for j in range(i, nmo):
            val = abs(float(eri_mo[i, i, j, j]))
            matrix[i, j] = val
            matrix[j, i] = val
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_overlap_proxy_matrix(mol, mo_coeff):
    """Compute an overlap-based proxy matrix for DMRG ordering."""
    S = mol.intor_symmetric("int1e_ovlp")
    orbital_overlap = mo_coeff.T @ S @ mo_coeff
    matrix = np.abs(orbital_overlap)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def chain_distance_cost(interaction_matrix, ordering):
    """Return sum_ij W_ij * |pos(i)-pos(j)| for a proposed chain ordering."""
    pos = np.empty(len(ordering), dtype=int)
    for chain_idx, orbital_idx in enumerate(ordering):
        pos[orbital_idx] = chain_idx
    distance = np.abs(pos[:, None] - pos[None, :])
    return float(np.sum(interaction_matrix * distance) / 2.0)


def genetic_algorithm_ordering(
    interaction_matrix,
    n_generations=100,
    population_size=50,
    mutation_rate=0.1,
    seed=None,
    *,
    objective: str = "distance_weighted",
):
    """Optimize orbital ordering using a genetic algorithm."""
    rng = np.random.default_rng(seed)

    n = interaction_matrix.shape[0]
    if n <= 3:
        return fiedler_ordering(interaction_matrix)

    if objective != "distance_weighted":
        raise ValueError(f"Unsupported GA objective: {objective}")

    baseline = fiedler_ordering(interaction_matrix)
    population = [baseline[:], list(reversed(baseline))]

    while len(population) < population_size:
        population.append(list(rng.permutation(n)))

    def cost(ordering):
        return chain_distance_cost(interaction_matrix, ordering)

    for _ in range(n_generations):
        scored = [(cost(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0])
        survivors = [ind for _, ind in scored[: max(2, population_size // 2)]]

        offspring = []
        while len(offspring) < population_size - len(survivors):
            p1, p2 = rng.choice(len(survivors), 2, replace=False)
            child = order_crossover(survivors[p1], survivors[p2], rng)
            if rng.random() < mutation_rate:
                i, j = rng.choice(n, 2, replace=False)
                child[i], child[j] = child[j], child[i]
            offspring.append(child)

        population = survivors + offspring

    final_scores = [(cost(ind), ind) for ind in population]
    final_scores.sort(key=lambda x: x[0])
    return final_scores[0][1]


def order_crossover(parent1, parent2, rng):
    """Order crossover (OX) for permutation GA."""
    n = len(parent1)
    child = [-1] * n

    start = int(rng.integers(0, n))
    end = int(rng.integers(start, n))
    child[start:end + 1] = parent1[start:end + 1]

    p2_filtered = [g for g in parent2 if g not in child[start:end + 1]]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_filtered[idx]
            idx += 1

    return child
