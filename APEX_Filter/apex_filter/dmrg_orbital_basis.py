"""DMRG orbital-basis preparation and ordering utilities.

This module owns the full "DMRG basis" algorithm layer:

- unrestricted UCCSD natural orbitals
- split localization
- alpha/beta pairing
- DMRG chain ordering

`steps_dmrg_basis.py` should remain orchestration-only.
Result persistence is an internal concern exposed through
`_save_dmrg_orbital_basis(...)`; canonical callers should go through the step
orchestrator rather than treating this module as a general artifact writer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass as _dataclass

import numpy as np
from pyscf import cc, lo
from shared.orbital_methods.localization import (
    split_localize_unrestricted as _shared_split_localize_unrestricted,
)
from shared.orbital_methods.metadata import DMRG_BASIS_SOURCE_METHOD
from shared.orbital_methods.natural_orbitals import (
    natural_orbitals_from_dm as _shared_natural_orbitals_from_dm,
)
from shared.orbital_methods.ordering import (
    chain_distance_cost as _chain_distance_cost,
    compute_ordering_matrix as _compute_dmrg_ordering_matrix,
    fiedler_ordering as _fiedler_ordering,
    genetic_algorithm_ordering as _genetic_algorithm_ordering,
)
from shared.orbital_methods.pairing import (
    pair_alpha_beta_orbitals as _pair_alpha_beta_orbitals,
    reorder_beta_to_match_alpha as _reorder_beta_to_match_alpha,
)
from shared.reference_states import load_reference_mf_from_npz as _load_reference_mf_from_npz
from shared.settings_payloads import extend_settings_payload as _extend_settings_payload

from .hdf5_state_io import _save_dmrg_basis_h5

logger = logging.getLogger(__name__)


@_dataclass
class _DMRGOrbitalBasisResult:
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


def _build_dmrg_orbital_basis(
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
    pm_exponent: int = 2,
    pm_init_guess: str = "atomic",
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
    if cas.mo_coeff_alpha is None or cas.mo_coeff_beta is None:
        raise ValueError("CAS must include AO-space mo_coeff_alpha/beta for DMRG orbital preparation")

    mf = _load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = cc_conv_tol
    mycc.max_cycle = cc_max_cycle
    mycc.diis_space = cc_diis_space
    mycc.direct = cc_direct
    mycc.kernel()

    dm1a_mo, dm1b_mo = mycc.make_rdm1(ao_repr=False)

    occ_a, rot_a = _shared_natural_orbitals_from_dm(dm1a_mo)
    occ_b, rot_b = _shared_natural_orbitals_from_dm(dm1b_mo)

    ao_coeff_a = cas.mo_coeff_alpha @ rot_a
    ao_coeff_b = cas.mo_coeff_beta @ rot_b

    nocc_a = int(np.sum(mf.mo_occ[0] > 0))
    nocc_b = int(np.sum(mf.mo_occ[1] > 0))

    loc_a = _shared_split_localize_unrestricted(
        mol,
        ao_coeff_a,
        nocc_a,
        method=localization_method,
        lo_module=lo,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        pm_exponent=pm_exponent,
        pm_init_guess=pm_init_guess,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )
    loc_b = _shared_split_localize_unrestricted(
        mol,
        ao_coeff_b,
        nocc_b,
        method=localization_method,
        lo_module=lo,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        pm_exponent=pm_exponent,
        pm_init_guess=pm_init_guess,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )

    pairs = _pair_alpha_beta_orbitals(mol, loc_a, loc_b)
    beta_paired = _reorder_beta_to_match_alpha(pairs, loc_b, loc_b.shape[1])
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
    interaction_matrix = _compute_dmrg_ordering_matrix(
        mol,
        pair_average,
        mode=ordering_matrix_mode,
        exchange_proxy_max_orbitals=exchange_proxy_max_orbitals,
    )
    fiedler = _fiedler_ordering(interaction_matrix)
    ordering = _genetic_algorithm_ordering(
        interaction_matrix,
        n_generations=ga_generations,
        population_size=ga_population,
        mutation_rate=ga_mutation_rate,
        seed=ga_seed,
        objective=ordering_objective,
    )
    ga_cost = _chain_distance_cost(interaction_matrix, ordering)
    fiedler_cost = _chain_distance_cost(interaction_matrix, fiedler)
    ordering_is_permutation = sorted(ordering) == list(range(len(ordering)))

    active_loc_a = _project_into_active_space(cas.mo_coeff_alpha, loc_a, S)
    active_loc_b = _project_into_active_space(cas.mo_coeff_beta, beta_paired, S)
    final_alpha = loc_a[:, ordering]
    final_beta = beta_paired[:, ordering]
    orth_err_alpha = float(np.max(np.abs(final_alpha.T @ S @ final_alpha - np.eye(final_alpha.shape[1]))))
    orth_err_beta = float(np.max(np.abs(final_beta.T @ S @ final_beta - np.eye(final_beta.shape[1]))))

    return _DMRGOrbitalBasisResult(
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


def _save_dmrg_orbital_basis(
    result: _DMRGOrbitalBasisResult,
    npz_path: str,
    *,
    label: str | None = None,
    family: str | None = None,
    energy: float | None = None,
    reference_state_path: str | None = None,
    fcidump_path: str | None = None,
    settings=None,
    settings_payload=None,
    cluster_info=None,
    fcidump_data=None,
    cas=None,
):
    """Save prepared DMRG orbitals and metadata."""
    settings_payload = _extend_settings_payload(
        settings_payload,
        source_method=result.source_method,
    )
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
        _save_dmrg_basis_h5(
            npz_path[:-4] + ".h5",
            payload,
            label=label,
            family=family,
            energy=energy,
            reference_state_path=reference_state_path,
            fcidump_path=fcidump_path,
            settings=settings,
            settings_payload=settings_payload,
            cluster_info=cluster_info,
            fcidump_data=fcidump_data,
            cas=cas,
        )
    np.savez(npz_path, **payload)


def _project_into_active_space(active_coeff, localized_coeff, S):
    """Express localized AO-space orbitals in the original active-space basis."""
    return active_coeff.T @ S @ localized_coeff
