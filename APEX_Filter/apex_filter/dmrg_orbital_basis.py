"""DMRG orbital-basis preparation and ordering utilities.

This module owns the full "DMRG basis" algorithm layer:

- unrestricted UCCSD natural orbitals
- split localization
- alpha/beta pairing
- DMRG chain ordering

`steps_dmrg_basis.py` should remain orchestration-only.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
from shared.orbital_methods.localization import (
    localize_orbital_block as _shared_localize_block,
    split_localize_unrestricted as _shared_split_localize_unrestricted,
)
from shared.orbital_methods.metadata import DMRG_BASIS_SOURCE_METHOD
from shared.orbital_methods.natural_orbitals import (
    natural_orbitals_from_dm as _shared_natural_orbitals_from_dm,
)
from shared.orbital_methods.ordering import (
    chain_distance_cost,
    compute_ordering_matrix as compute_dmrg_ordering_matrix,
    compute_overlap_proxy_matrix,
    fiedler_ordering,
    genetic_algorithm_ordering,
)
from shared.orbital_methods.pairing import (
    pair_alpha_beta_orbitals,
    pair_alpha_beta_orbitals_with_overlap,
    reorder_beta_to_match_alpha,
)

from .models import CAS
from .hdf5_state_io import save_dmrg_basis_h5
from .reference_ucc import load_reference_mf_from_npz

logger = logging.getLogger(__name__)


@dataclass
class DMRGOrbitalBasisResult:
    """Prepared unrestricted orbital basis for DMRG."""

    mo_coeff_alpha: np.ndarray
    mo_coeff_beta: np.ndarray
    active_coeff_alpha: np.ndarray
    active_coeff_beta: np.ndarray
    alpha_no_occupations: np.ndarray
    beta_no_occupations: np.ndarray
    nocc_alpha: int
    nocc_beta: int
    pairs: list[tuple[int, int]]
    ordering: list[int]
    localization_method: str
    source_method: str
    ordering_matrix_mode: str
    ordering_objective: str
    pair_diag_overlap_min: float
    pair_diag_overlap_mean: float
    diag_dominant_fraction: float
    orth_err_alpha: float
    orth_err_beta: float
    ordering_is_permutation: bool
    ga_cost: float
    fiedler_cost: float


def build_dmrg_orbital_basis(
    mol,
    cas,
    fcidump_data,
    uhf_npz_path: str,
    *,
    localization_method: str = "pm",
    cc_conv_tol: float = 1e-8,
    cc_max_cycle: int = 2000,
    cc_diis_space: int = 12,
    cc_direct: bool = False,
    pm_pop_method: str = "mulliken",
    pm_conv_tol: float = 1e-6,
    pm_conv_tol_grad: float | None = None,
    pm_max_cycle: int = 100,
    boys_conv_tol: float = 1e-6,
    boys_conv_tol_grad: float | None = None,
    boys_max_cycle: int = 100,
    ordering_matrix_mode: str = "exchange_proxy",
    ordering_objective: str = "distance_weighted",
    exchange_proxy_max_orbitals: int = 64,
    ga_generations: int = 100,
    ga_population: int = 50,
    ga_mutation_rate: float = 0.1,
    ga_seed: int | None = 17,
):
    """Construct Chan-2026-style unrestricted DMRG orbitals.

    Pipeline:
    1. Rebuild the active-space UHF reference from step3 NPZ.
    2. Run active-space UCCSD on the same FCIDUMP Hamiltonian.
    3. Build unrestricted natural orbitals from alpha/beta 1-RDM blocks.
    4. Split-localize alpha/beta occupied/virtual blocks separately.
    5. Pair alpha/beta orbitals by overlap maximization.
    6. Reorder orbital pairs for the DMRG chain.
    """
    from pyscf import cc, lo

    if cas.mo_coeff_alpha is None or cas.mo_coeff_beta is None:
        raise ValueError("CAS must include AO-space mo_coeff_alpha/beta for DMRG orbital preparation")

    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = cc_conv_tol
    mycc.max_cycle = cc_max_cycle
    mycc.diis_space = cc_diis_space
    mycc.direct = cc_direct
    mycc.kernel()

    dm1a_mo, dm1b_mo = mycc.make_rdm1(ao_repr=False)

    occ_a, rot_a = _natural_orbitals_from_dm(dm1a_mo)
    occ_b, rot_b = _natural_orbitals_from_dm(dm1b_mo)

    ao_coeff_a = cas.mo_coeff_alpha @ rot_a
    ao_coeff_b = cas.mo_coeff_beta @ rot_b

    nocc_a = int(np.sum(mf.mo_occ[0] > 0))
    nocc_b = int(np.sum(mf.mo_occ[1] > 0))

    loc_a = _split_localize_unrestricted(
        mol,
        ao_coeff_a,
        nocc_a,
        method=localization_method,
        lo_module=lo,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )
    loc_b = _split_localize_unrestricted(
        mol,
        ao_coeff_b,
        nocc_b,
        method=localization_method,
        lo_module=lo,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )

    pairs = pair_alpha_beta_orbitals(mol, loc_a, loc_b)
    beta_paired = reorder_beta_to_match_alpha(pairs, loc_b, loc_b.shape[1])
    S = mol.intor_symmetric("int1e_ovlp")

    pair_overlap = np.abs(loc_a.T @ S @ beta_paired)
    pair_diag = np.diag(pair_overlap)
    pair_diag_overlap_min = float(pair_diag.min()) if pair_diag.size else 0.0
    pair_diag_overlap_mean = float(pair_diag.mean()) if pair_diag.size else 0.0
    diag_dominant_fraction = (
        float(np.mean(pair_diag >= pair_overlap.max(axis=1) - 1e-10))
        if pair_diag.size
        else 1.0
    )

    pair_average = 0.5 * (loc_a + beta_paired)
    interaction_matrix = compute_dmrg_ordering_matrix(
        mol,
        pair_average,
        mode=ordering_matrix_mode,
        exchange_proxy_max_orbitals=exchange_proxy_max_orbitals,
    )
    fiedler = fiedler_ordering(interaction_matrix)
    ordering = genetic_algorithm_ordering(
        interaction_matrix,
        n_generations=ga_generations,
        population_size=ga_population,
        mutation_rate=ga_mutation_rate,
        seed=ga_seed,
        objective=ordering_objective,
    )
    ga_cost = chain_distance_cost(interaction_matrix, ordering)
    fiedler_cost = chain_distance_cost(interaction_matrix, fiedler)
    ordering_is_permutation = sorted(ordering) == list(range(len(ordering)))

    active_loc_a = _project_into_active_space(cas.mo_coeff_alpha, loc_a, S)
    active_loc_b = _project_into_active_space(cas.mo_coeff_beta, beta_paired, S)
    final_alpha = loc_a[:, ordering]
    final_beta = beta_paired[:, ordering]
    orth_err_alpha = float(np.max(np.abs(final_alpha.T @ S @ final_alpha - np.eye(final_alpha.shape[1]))))
    orth_err_beta = float(np.max(np.abs(final_beta.T @ S @ final_beta - np.eye(final_beta.shape[1]))))

    return DMRGOrbitalBasisResult(
        mo_coeff_alpha=final_alpha,
        mo_coeff_beta=final_beta,
        active_coeff_alpha=active_loc_a[:, ordering],
        active_coeff_beta=active_loc_b[:, ordering],
        alpha_no_occupations=occ_a,
        beta_no_occupations=occ_b,
        nocc_alpha=nocc_a,
        nocc_beta=nocc_b,
        pairs=pairs,
        ordering=ordering,
        localization_method=localization_method,
        source_method=DMRG_BASIS_SOURCE_METHOD,
        ordering_matrix_mode=ordering_matrix_mode,
        ordering_objective=ordering_objective,
        pair_diag_overlap_min=pair_diag_overlap_min,
        pair_diag_overlap_mean=pair_diag_overlap_mean,
        diag_dominant_fraction=diag_dominant_fraction,
        orth_err_alpha=orth_err_alpha,
        orth_err_beta=orth_err_beta,
        ordering_is_permutation=ordering_is_permutation,
        ga_cost=ga_cost,
        fiedler_cost=fiedler_cost,
    )


def save_dmrg_orbital_basis(result: DMRGOrbitalBasisResult, npz_path: str):
    """Save prepared DMRG orbitals and metadata."""
    pair_array = np.asarray(result.pairs, dtype=int) if result.pairs else np.empty((0, 2), dtype=int)
    payload = {
        "mo_coeff_alpha": result.mo_coeff_alpha,
        "mo_coeff_beta": result.mo_coeff_beta,
        "active_coeff_alpha": result.active_coeff_alpha,
        "active_coeff_beta": result.active_coeff_beta,
        "alpha_no_occupations": result.alpha_no_occupations,
        "beta_no_occupations": result.beta_no_occupations,
        "nocc_alpha": result.nocc_alpha,
        "nocc_beta": result.nocc_beta,
        "pairs": pair_array,
        "ordering": np.asarray(result.ordering, dtype=int),
        "localization_method": np.array(result.localization_method),
        "source_method": np.array(result.source_method),
        "ordering_matrix_mode": np.array(result.ordering_matrix_mode),
        "ordering_objective": np.array(result.ordering_objective),
        "pair_diag_overlap_min": np.array(result.pair_diag_overlap_min),
        "pair_diag_overlap_mean": np.array(result.pair_diag_overlap_mean),
        "diag_dominant_fraction": np.array(result.diag_dominant_fraction),
        "orth_err_alpha": np.array(result.orth_err_alpha),
        "orth_err_beta": np.array(result.orth_err_beta),
        "ordering_is_permutation": np.array(result.ordering_is_permutation),
        "ga_cost": np.array(result.ga_cost),
        "fiedler_cost": np.array(result.fiedler_cost),
    }
    if npz_path.endswith(".npz"):
        save_dmrg_basis_h5(npz_path[:-4] + ".h5", payload)
    np.savez(npz_path, **payload)


def _natural_orbitals_from_dm(dm_mo: np.ndarray):
    """Diagonalize a 1-RDM block to obtain natural occupations/orbitals."""
    return _shared_natural_orbitals_from_dm(dm_mo)


def _project_into_active_space(active_coeff, localized_coeff, S):
    """Express localized AO-space orbitals in the original active-space basis."""
    return active_coeff.T @ S @ localized_coeff


def _split_localize_unrestricted(
    mol,
    ao_coeff: np.ndarray,
    nocc: int,
    *,
    method: str,
    lo_module,
    pm_pop_method: str,
    pm_conv_tol: float,
    pm_conv_tol_grad: float | None,
    pm_max_cycle: int,
    boys_conv_tol: float,
    boys_conv_tol_grad: float | None,
    boys_max_cycle: int,
):
    """Localize occupied and virtual blocks separately for one spin channel."""
    return _shared_split_localize_unrestricted(
        mol,
        ao_coeff,
        nocc,
        method=method,
        lo_module=lo_module,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )


def _localize_block(
    mol,
    mo_block: np.ndarray,
    *,
    method: str,
    lo_module,
    pm_pop_method: str,
    pm_conv_tol: float,
    pm_conv_tol_grad: float | None,
    pm_max_cycle: int,
    boys_conv_tol: float,
    boys_conv_tol_grad: float | None,
    boys_max_cycle: int,
):
    """Localize one orbital block with PM or Boys."""
    return _shared_localize_block(
        mol,
        mo_block,
        method=method,
        lo_module=lo_module,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )


def reorder_cas_orbital_coefficients(active_orbitals: CAS, ordering: list) -> CAS:
    """Reorder CAS orbital coefficient-like fields for a DMRG chain ordering."""
    result = copy.deepcopy(active_orbitals)

    if active_orbitals.mo_coeff_alpha is not None:
        result.mo_coeff_alpha = active_orbitals.mo_coeff_alpha[:, ordering]
    if active_orbitals.mo_coeff_beta is not None:
        result.mo_coeff_beta = active_orbitals.mo_coeff_beta[:, ordering]
    if active_orbitals.occupations is not None:
        result.occupations = active_orbitals.occupations[ordering]

    result.orbital_labels = [
        active_orbitals.orbital_labels[i]
        for i in ordering
        if i < len(active_orbitals.orbital_labels)
    ]
    result.orbital_ordering = np.array(ordering, dtype=int)
    return result
