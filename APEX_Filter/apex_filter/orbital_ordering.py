"""Module 2c: Orbital Ordering

For DMRG, the orbital ordering along the 1D chain critically affects
convergence. Two steps:
  A. Alpha/beta orbital pairing by overlap maximization (Hungarian algorithm)
  B. Orbital pair reordering via Fiedler vector (spectral ordering) or
     genetic algorithm (combinatorial optimization)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import CAS


try:
    from pyscf import gto
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


# ──────────────────────────────────────────────────────────────────
# Step A: Alpha/Beta orbital pairing
# ──────────────────────────────────────────────────────────────────

def pair_alpha_beta_orbitals(mol, mo_alpha, mo_beta):
    """Find optimal alpha-beta orbital pairing by overlap maximization.

    Uses the Hungarian algorithm to find the assignment that maximizes
    the sum of |<alpha_i|beta_j>|.

    Args:
        mol: PySCF Mole object (needed for overlap matrix).
        mo_alpha: (nao, nmo) alpha MO coefficients.
        mo_beta: (nao, nmo) beta MO coefficients.

    Returns:
        List of (alpha_idx, beta_idx) pairs.
    """
    S = mol.intor_symmetric("int1e_ovlp")

    # Compute overlap matrix: overlap[i,j] = <alpha_i|beta_j>
    overlap = mo_alpha.T @ S @ mo_beta

    # Hungarian algorithm on -|overlap| (minimize = maximize |overlap|)
    row_ind, col_ind = linear_sum_assignment(-np.abs(overlap))

    return list(zip(row_ind.tolist(), col_ind.tolist()))


def pair_alpha_beta_orbitals_no_mol(S_matrix, mo_alpha, mo_beta):
    """Pair alpha/beta orbitals given a pre-computed overlap matrix.

    Args:
        S_matrix: AO overlap matrix (nao, nao).
        mo_alpha: (nao, nmo) alpha MO coefficients.
        mo_beta: (nao, nmo) beta MO coefficients.

    Returns:
        List of (alpha_idx, beta_idx) pairs.
    """
    overlap = mo_alpha.T @ S_matrix @ mo_beta
    row_ind, col_ind = linear_sum_assignment(-np.abs(overlap))
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def reorder_beta_to_match_alpha(pairs, mo_beta, n_orbitals):
    """Reorder beta orbitals to match alpha orbital ordering.

    Args:
        pairs: List of (alpha_idx, beta_idx) from pairing.
        mo_beta: (nao, nmo) beta MO coefficients.
        n_orbitals: Number of active orbitals.

    Returns:
        Reordered beta MO coefficients (nao, n_orbitals).
    """
    # Sort pairs by alpha index
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    beta_order = [p[1] for p in sorted_pairs[:n_orbitals]]

    return mo_beta[:, beta_order]


# ──────────────────────────────────────────────────────────────────
# Step B: Orbital pair reordering
# ──────────────────────────────────────────────────────────────────

def fiedler_ordering(interaction_matrix):
    """Compute Fiedler (spectral) ordering for orbital pairs.

    The Fiedler vector is the eigenvector corresponding to the second-smallest
    eigenvalue of the graph Laplacian. Sorting by Fiedler vector components
    gives a good 1D ordering that minimizes entanglement range.

    Args:
        interaction_matrix: (n, n) matrix of mutual information or
            exchange integrals between orbital pairs.

    Returns:
        List of orbital indices in the recommended DMRG ordering.
    """
    n = interaction_matrix.shape[0]
    if n <= 2:
        return list(range(n))

    # Build adjacency matrix (use absolute values)
    W = np.abs(interaction_matrix)
    np.fill_diagonal(W, 0)

    # Graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Fiedler vector = eigenvector for 2nd smallest eigenvalue
    # (first eigenvalue is ~0 with eigenvector [1,1,...,1])
    if len(eigenvalues) > 1:
        fiedler_vec = eigenvectors[:, 1]
    else:
        fiedler_vec = eigenvectors[:, 0]

    # Sort by Fiedler vector value
    ordering = np.argsort(fiedler_vec).tolist()

    return ordering


def compute_mutual_information_matrix(mol, mo_coeff, dm=None):
    """Compute a mutual-information-like matrix from orbital overlaps.

    This is a proxy for the true mutual information that would come from
    a DMRG calculation. Uses exchange-like integrals as an approximation.

    Args:
        mol: PySCF Mole object.
        mo_coeff: (nao, nmo) MO coefficients.
        dm: Optional density matrix for weighting.

    Returns:
        (nmo, nmo) interaction matrix.
    """
    nmo = mo_coeff.shape[1]

    # Compute two-electron integrals in MO basis (if feasible)
    # For large active spaces, use an overlap-based proxy
    try:
        eri_ao = mol.intor("int2e_sph", aosym="s1")
        # Transform to MO basis: (ij|kl) = sum_{abcd} C_ai C_bj C_ck C_dl (ab|cd)
        # This is expensive for large nmo, use truncated approach
        if nmo <= 50:
            return _compute_exchange_matrix(mol, mo_coeff, eri_ao)
    except Exception:
        pass

    # Fallback: use orbital spatial overlap as proxy
    return _compute_overlap_interaction(mol, mo_coeff)


def compute_exchange_integral_matrix(mol, mo_coeff):
    """Compute the exchange integral matrix K_ij = (ii|jj) in MO basis.

    This serves as a proxy for entanglement between orbital pairs.

    Args:
        mol: PySCF Mole object.
        mo_coeff: (nao, nmo) MO coefficients.

    Returns:
        (nmo, nmo) exchange integral matrix.
    """
    nmo = mo_coeff.shape[1]
    K = np.zeros((nmo, nmo))

    try:
        # Use PySCF's AO->MO transformation
        from pyscf import ao2mo
        eri_mo = ao2mo.full(mol, mo_coeff, compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)

        for i in range(nmo):
            for j in range(nmo):
                K[i, j] = abs(eri_mo[i, i, j, j])
    except Exception:
        # Fallback to overlap proxy
        K = _compute_overlap_interaction(mol, mo_coeff)

    return K


def genetic_algorithm_ordering(interaction_matrix, n_generations=100,
                                population_size=50, mutation_rate=0.1,
                                seed=None):
    """Optimize orbital ordering using a genetic algorithm.

    The fitness function is the sum of interaction_matrix[i][i+1] along
    the chain, which we want to maximize (keep strongly interacting
    orbitals adjacent).

    Args:
        interaction_matrix: (n, n) interaction/entanglement matrix.
        n_generations: Number of GA generations.
        population_size: Population size.
        mutation_rate: Probability of random swap mutation.
        seed: Random seed for reproducibility.

    Returns:
        List of orbital indices in the optimized ordering.
    """
    if seed is not None:
        np.random.seed(seed)

    n = interaction_matrix.shape[0]
    if n <= 3:
        return fiedler_ordering(interaction_matrix)

    # Initialize population with Fiedler ordering + random permutations
    fiedler = fiedler_ordering(interaction_matrix)
    population = [fiedler[:]]

    # Add random permutations
    while len(population) < population_size:
        perm = list(np.random.permutation(n))
        population.append(perm)

    def fitness(ordering):
        """Sum of nearest-neighbor interactions along the chain."""
        total = 0.0
        for k in range(len(ordering) - 1):
            i, j = ordering[k], ordering[k + 1]
            total += interaction_matrix[i, j]
        return total

    # Evolution
    for gen in range(n_generations):
        # Evaluate fitness
        scores = [(fitness(ind), ind) for ind in population]
        scores.sort(key=lambda x: -x[0])  # descending

        # Selection: keep top 50%
        survivors = [ind for _, ind in scores[:population_size // 2]]

        # Crossover: create offspring
        offspring = []
        while len(offspring) < population_size - len(survivors):
            p1, p2 = np.random.choice(len(survivors), 2, replace=False)
            child = _order_crossover(survivors[p1], survivors[p2])
            offspring.append(child)

        # Mutation
        for child in offspring:
            if np.random.random() < mutation_rate:
                i, j = np.random.choice(n, 2, replace=False)
                child[i], child[j] = child[j], child[i]

        population = survivors + offspring

    # Return best
    scores = [(fitness(ind), ind) for ind in population]
    scores.sort(key=lambda x: -x[0])
    return scores[0][1]


def reorder_orbitals(active_orbitals: CAS,
                      ordering: list) -> CAS:
    """Apply a given ordering to CAS.

    Args:
        active_orbitals: Original active orbitals.
        ordering: List of orbital indices in the new order.

    Returns:
        New CAS with reordered MO coefficients.
    """
    import copy
    result = copy.deepcopy(active_orbitals)


    if active_orbitals.mo_coeff_alpha is not None:
        result.mo_coeff_alpha = active_orbitals.mo_coeff_alpha[:, ordering]
    if active_orbitals.mo_coeff_beta is not None:
        result.mo_coeff_beta = active_orbitals.mo_coeff_beta[:, ordering]
    if active_orbitals.occupations is not None:
        result.occupations = active_orbitals.occupations[ordering]

    new_labels = [active_orbitals.orbital_labels[i] for i in ordering
                  if i < len(active_orbitals.orbital_labels)]
    result.orbital_labels = new_labels
    result.orbital_ordering = np.array(ordering)

    return result


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _compute_exchange_matrix(mol, mo_coeff, eri_ao):
    """Compute exchange integral matrix from AO ERI tensor."""
    nmo = mo_coeff.shape[1]
    K = np.zeros((nmo, nmo))

    # Transform (ii|jj) for each pair
    for i in range(nmo):
        for j in range(i, nmo):
            val = _transform_eri_element(eri_ao, mo_coeff, i, i, j, j)
            K[i, j] = abs(val)
            K[j, i] = K[i, j]

    return K


def _transform_eri_element(eri_ao, C, i, j, k, l):
    """Transform a single ERI element (ij|kl) to MO basis."""
    nao = C.shape[0]
    Ci = C[:, i]
    Cj = C[:, j]
    Ck = C[:, k]
    Cl = C[:, l]

    result = 0.0
    for a in range(nao):
        for b in range(nao):
            for c in range(nao):
                for d in range(nao):
                    result += Ci[a] * Cj[b] * Ck[c] * Cl[d] * eri_ao[a, b, c, d]
    return result


def _compute_overlap_interaction(mol, mo_coeff):
    """Compute interaction matrix from orbital spatial overlap.

    Proxy for mutual information: overlap_ij = sum_a |C_ai * C_aj| * w_a
    where w_a weights by AO distance to nearest metal center.
    """
    S = mol.intor_symmetric("int1e_ovlp")
    nmo = mo_coeff.shape[1]

    # Overlap of orbital densities
    interaction = np.zeros((nmo, nmo))
    for i in range(nmo):
        for j in range(i + 1, nmo):
            # Orbital density overlap
            rho_ij = mo_coeff[:, i] * mo_coeff[:, j]
            val = abs(np.sum(rho_ij))
            interaction[i, j] = val
            interaction[j, i] = val

    return interaction


def _order_crossover(parent1, parent2):
    """Order crossover (OX) for permutation GA."""
    n = len(parent1)
    child = [-1] * n

    # Select a segment from parent1
    start = np.random.randint(0, n)
    end = np.random.randint(start, n)

    # Copy segment
    child[start:end + 1] = parent1[start:end + 1]

    # Fill remaining from parent2 (in order)
    p2_filtered = [g for g in parent2 if g not in child[start:end + 1]]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_filtered[idx]
            idx += 1

    return child
