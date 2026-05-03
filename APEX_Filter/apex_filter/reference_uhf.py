"""Reference-state UHF on the active-space Hamiltonian defined by FCIDUMP.

This module anchors the canonical Step 3 route where broken-symmetry UHF is
solved directly on the active-space Hamiltonian rather than on the full
molecule. The intentional public surface here is:

- ``converge_reference_uhf(...)`` for Step 3 orchestration

All remaining helpers should be read as internal workflow support for initial
guesses, label interpretation, and broken-symmetry bookkeeping. Fake-molecule
and reference-UHF construction live in ``shared.active_space_reference`` and
are imported here only as internal workflow seams.
"""

import logging
from dataclasses import dataclass as _dataclass

import numpy as np

from shared.active_space_reference import (
    _sanitize_ms2_for_nelec,
    build_fake_mol as _build_fake_mol,
    build_reference_uhf_solver as _build_reference_uhf_solver,
)
from shared.chem_knowledge import (
    get_common_oxidation_states as _get_common_oxidation_states,
    get_local_spin as _get_local_spin,
)
from shared.cluster_info_labels import resolve_metal_site_label as _resolve_metal_site_label
from shared.final_state_signatures import (
    parse_orbital_metal_mapping as _parse_orbital_metal_mapping,
    summarize_final_state_from_dm as _summarize_final_state_from_dm,
)

from shared.models import CAS as _CAS, ClusterInfo as _ClusterInfo, ElectronicConfig as _ElectronicConfig

logger = logging.getLogger(__name__)


@_dataclass
class _ReferenceUHFResult:
    """Result of a reference-state UHF calculation in the active space."""
    config: _ElectronicConfig
    energy: float
    converged: bool
    s_squared: float
    mo_coeff: tuple  # (mo_a, mo_b) in FCIDUMP basis, each (norb, norb)
    mo_energy: tuple  # (eps_a, eps_b), each (norb,)
    dm: tuple         # (dm_a, dm_b) in FCIDUMP basis, each (norb, norb)
    mo_occ: tuple     # (occ_a, occ_b)
    diagnostics: dict | None = None


def _build_bs_initial_guess_active_space(
    cas: _CAS,
    config: _ElectronicConfig,
    fcidump_data,
    metal_orbital_map: dict,
) -> tuple:
    """Build BS density matrix in the active-space MO basis.

    Strategy:
    1. Use CAS.occupations as reference occupation numbers.
    2. For majority-spin metal d-orbitals: high alpha occupation.
    3. For minority-spin metal d-orbitals: high beta occupation (spin flip).
    4. For ligand/bridging orbitals: equal alpha/beta = occupation/2.
    5. D-orbital assignment: place extra electron on specified orbital.

    Parameters
    ----------
    cas : CAS
        Active space with orbital_labels and occupations.
    config : ElectronicConfig
        Electronic configuration specifying spin and oxidation.
    fcidump_data : FCIDUMPData
        Integral data (norb used for matrix dimensions).
    metal_orbital_map : dict
        {orb_idx: metal_site_idx} from _parse_orbital_metal_mapping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dm_a, dm_b) each of shape (norb, norb).
    """
    norb = fcidump_data.norb
    dm_a = np.zeros((norb, norb))
    dm_b = np.zeros((norb, norb))

    occupations = cas.occupations
    if occupations is None:
        # Fallback: equal distribution
        occupations = np.ones(norb) * fcidump_data.nelec / (2.0 * norb)

    spin_assignment = config.spin_assignment  # {site_idx: +1/-1}

    for i in range(norb):
        if i >= len(occupations):
            break

        occ = occupations[i]
        metal_site = metal_orbital_map.get(i)

        if metal_site is not None and metal_site in spin_assignment:
            # Metal-localized orbital: apply BS spin polarization
            spin_dir = spin_assignment[metal_site]

            if spin_dir == +1:
                # Majority spin: mostly alpha
                dm_a[i, i] = min(occ, 1.0) if occ > 0.5 else occ
                dm_b[i, i] = max(occ - 1.0, 0.0) if occ > 1.0 else 0.0
            else:
                # Minority spin: mostly beta (spin flip)
                dm_b[i, i] = min(occ, 1.0) if occ > 0.5 else occ
                dm_a[i, i] = max(occ - 1.0, 0.0) if occ > 1.0 else 0.0
        else:
            # Ligand/bridging orbital: equal alpha/beta
            dm_a[i, i] = occ / 2.0
            dm_b[i, i] = occ / 2.0

    # Apply d-orbital assignment: extra electron on specified orbital
    if config.d_orbital_assignments:
        for site_idx, d_orb_idx in config.d_orbital_assignments.items():
            spin_dir = spin_assignment.get(site_idx, +1)
            _apply_d_orbital_encoding(
                dm_a, dm_b, site_idx, d_orb_idx, spin_dir,
                metal_orbital_map, cas,
            )

    return dm_a, dm_b


def converge_reference_uhf(
    cas: _CAS,
    config: _ElectronicConfig,
    fcidump_data,
    cluster_info: _ClusterInfo,
    *,
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    stabilize_cycles: int = 20,
    level_shift: float = 0.3,
    damp: float = 0.2,
    newton_refine: bool = False,
    newton_max_cycle: int = 8,
) -> _ReferenceUHFResult:
    """Run BS-UHF within the active space.

    Steps:
    1. Build fake mol + SCF with FCIDUMP integrals
    2. Run high-spin UHF to convergence
    3. Build BS initial guess from high-spin density
    4. Run BS-UHF with stabilization (level_shift + damp)
    5. Tight convergence (no level_shift/damp)

    Parameters
    ----------
    cas : CAS
        Active space definition.
    config : ElectronicConfig
        Electronic configuration.
    fcidump_data : FCIDUMPData
        FCIDUMP integrals.
    cluster_info : ClusterInfo
        Cluster info.
    conv_tol : float
        SCF convergence tolerance.
    max_cycle : int
        Maximum SCF iterations.
    stabilize_cycles : int
        Number of BS-UHF stabilization iterations before the tight pass.
    level_shift : float
        Level shift applied during the stabilization pass.
    damp : float
        Damping factor applied during the stabilization pass.
    newton_refine : bool
        Whether to apply a short Newton/SOSCF refinement after the BS tight pass
        if the regular BS-UHF is still unconverged.
    newton_max_cycle : int
        Maximum Newton refinement cycles.

    Returns
    -------
    _ReferenceUHFResult
    """
    norb = fcidump_data.norb
    nelec = fcidump_data.nelec
    # Use the aligned local-spin estimate as the high-spin reference.
    ms2_highspin = _compute_high_spin_ms2(cluster_info, config)
    ms2_highspin = _sanitize_ms2_for_nelec(nelec, ms2_highspin)

    try:
        import pyscf  # noqa: F401
    except ImportError:
        return _ReferenceUHFResult(
            config=config, energy=0.0, converged=False,
            s_squared=0.0, mo_coeff=(None, None), mo_energy=(None, None),
            dm=(None, None),
            mo_occ=(None, None),
        )

    # Build orbital-metal mapping
    metal_orbital_map = _parse_orbital_metal_mapping(cas, cluster_info)
    _warn_if_spin_sites_unmapped(config, metal_orbital_map, cluster_info)

    # Step 1: High-spin UHF
    hs_history = []

    def _collect_cycle(history_store):
        def _callback(envs):
            history_store.append(
                {
                    "cycle": int(envs.get("cycle", -1)) + 1,
                    "energy": float(envs.get("e_tot", 0.0)),
                    "delta_e": float(envs.get("e_tot", 0.0) - envs.get("last_hf_e", 0.0)),
                    "norm_gorb": float(envs.get("norm_gorb", 0.0)),
                    "norm_ddm": float(envs.get("norm_ddm", 0.0)),
                }
            )
        return _callback

    mol_hs = _build_fake_mol(norb, nelec, ms2_highspin, ecore=fcidump_data.ecore)
    mf_hs = _build_reference_uhf_solver(fcidump_data, mol_hs, conv_tol, max_cycle)
    mf_hs.callback = _collect_cycle(hs_history)
    mf_hs.kernel()

    if not mf_hs.converged:
        logger.warning(
            "High-spin UHF did not converge for %s; falling back to occupation-based BS guess",
            config.label,
        )
        dm_hs_a, dm_hs_b = _build_bs_initial_guess_active_space(
            cas, config, fcidump_data, metal_orbital_map
        )
    else:
        # Step 2: Build BS initial guess from the converged high-spin density.
        dm_hs_a, dm_hs_b = mf_hs.make_rdm1()
        dm_hs_a = dm_hs_a.copy()
        dm_hs_b = dm_hs_b.copy()

        # Apply BS spin flip: swap alpha/beta for minority-metal d orbitals.
        for orb_idx, metal_site in metal_orbital_map.items():
            if metal_site is not None and metal_site in config.spin_assignment:
                if config.spin_assignment[metal_site] == -1:
                    _swap_orbital_spin(dm_hs_a, dm_hs_b, orb_idx)

        # Apply d-orbital assignment after the spin flip.
        if config.d_orbital_assignments:
            for site_idx, d_orb_idx in config.d_orbital_assignments.items():
                spin_dir = config.spin_assignment.get(site_idx, +1)
                _apply_d_orbital_encoding(
                    dm_hs_a, dm_hs_b, site_idx, d_orb_idx, spin_dir,
                    metal_orbital_map, cas,
                )

    # Step 3: BS-UHF with stabilization
    ms2_target = int(2 * config.spin_isomer.Sz) if config.spin_isomer else 0
    ms2_target = _sanitize_ms2_for_nelec(nelec, ms2_target)
    mol_bs = _build_fake_mol(norb, nelec, ms2_target, ecore=fcidump_data.ecore)
    mf_bs = _build_reference_uhf_solver(fcidump_data, mol_bs, conv_tol, stabilize_cycles)
    bs_stabilize_history = []
    bs_tight_history = []
    mf_bs.callback = _collect_cycle(bs_stabilize_history)
    mf_bs.level_shift = level_shift
    mf_bs.damp = damp
    mf_bs.kernel(dm0=(dm_hs_a, dm_hs_b))

    # Step 4: Tight convergence
    mf_bs.callback = _collect_cycle(bs_tight_history)
    mf_bs.max_cycle = max_cycle
    mf_bs.level_shift = 0.0
    mf_bs.damp = 0.0
    mf_bs.kernel(dm0=mf_bs.make_rdm1())

    mf_final = mf_bs
    newton_history = []
    if newton_refine and not mf_bs.converged and newton_max_cycle > 0:
        try:
            mf_newton = mf_bs.newton()
            mf_newton.conv_tol = conv_tol
            mf_newton.max_cycle = newton_max_cycle
            mf_newton.callback = _collect_cycle(newton_history)
            mf_newton.kernel(mf_bs.mo_coeff, mf_bs.mo_occ)
            mf_final = mf_newton
        except Exception as exc:  # pragma: no cover - refinement failure fallback
            logger.warning("Newton refinement failed for %s: %s", config.label, exc)

    try:
        s2 = mf_final.spin_square()[0]
    except Exception:  # pragma: no cover - defensive fallback for solver backends
        s2 = 0.0
    dm_final = mf_final.make_rdm1()
    active_history = newton_history or bs_tight_history or bs_stabilize_history
    final_summary = _summarize_final_state_from_dm(cas, config, cluster_info, dm_final)
    diagnostics = {
        "high_spin_ms2": ms2_highspin,
        "target_ms2": ms2_target,
        "hs_history": hs_history,
        "bs_stabilize_history": bs_stabilize_history,
        "bs_tight_history": bs_tight_history,
        "newton_used": bool(newton_history),
        "newton_history": newton_history,
        "final_delta_e": active_history[-1]["delta_e"] if active_history else None,
        "energy_tail": [entry["energy"] for entry in active_history[-5:]],
        **final_summary,
    }

    return _ReferenceUHFResult(
        config=config,
        energy=mf_final.e_tot,
        converged=mf_final.converged,
        s_squared=s2,
        mo_coeff=(mf_final.mo_coeff[0], mf_final.mo_coeff[1]),
        mo_energy=(mf_final.mo_energy[0], mf_final.mo_energy[1]),
        dm=(dm_final[0], dm_final[1]),
        mo_occ=(mf_final.mo_occ[0], mf_final.mo_occ[1]),
        diagnostics=diagnostics,
    )


def _swap_orbital_spin(dm_a, dm_b, orb_idx):
    """Swap alpha and beta contributions for orbital orb_idx.

    Swaps the row and column corresponding to orb_idx between dm_a and dm_b.
    """
    orig_a = dm_a.copy()
    orig_b = dm_b.copy()
    dm_a[orb_idx, :] = orig_b[orb_idx, :]
    dm_b[orb_idx, :] = orig_a[orb_idx, :]
    dm_a[:, orb_idx] = orig_b[:, orb_idx]
    dm_b[:, orb_idx] = orig_a[:, orb_idx]


def _apply_d_orbital_encoding(dm_a, dm_b, site_idx, d_orb_idx, spin_dir,
                               metal_orbital_map, cas):
    """Encode d-orbital assignment in the density matrix.

    Redistributes the existing minority-spin occupation within the
    target metal d-manifold so the selected d orbital carries the
    minority-spin electron. This preserves the total electron count
    of the initial guess, unlike adding a new electron by hand.
    """
    # Find the orbital indices belonging to this metal site
    metal_orbs = [i for i, s in metal_orbital_map.items() if s == site_idx]

    if not metal_orbs or d_orb_idx >= len(metal_orbs):
        return

    target_orb = metal_orbs[d_orb_idx]
    minority_dm = dm_b if spin_dir == +1 else dm_a

    block = minority_dm[np.ix_(metal_orbs, metal_orbs)]
    total_occ = float(np.trace(block))
    if total_occ <= 1e-10:
        return

    # Reset the local minority-spin occupation so the selected orbital
    # is the doubly occupied one for this broken-symmetry reference.
    for orb in metal_orbs:
        minority_dm[orb, orb] = 0.0
    minority_dm[target_orb, target_orb] = total_occ


def _compute_high_spin_ms2(cluster_info, config):
    """Compute maximum Ms for the high-spin reference state.

    All metal spins aligned in the same direction.
    Ms = sum of all local Si.
    """
    total_ms = 0
    for i, metal in enumerate(cluster_info.metals):
        # Use oxidation state from config if available
        if config.oxidation and i in config.oxidation.assignments:
            ox = config.oxidation.assignments[i]
        else:
            states = _get_common_oxidation_states(metal.element)
            ox = states[0] if states else 2
        S = _get_local_spin(metal.element, ox)
        total_ms += int(2 * S)  # 2*Ms = 2*S for fully aligned

    return total_ms


def _warn_if_spin_sites_unmapped(
    config: _ElectronicConfig,
    metal_orbital_map: dict,
    cluster_info: _ClusterInfo,
):
    """Warn when the BS pattern names metal sites absent from the active-space labels."""
    covered_sites = {site_idx for site_idx in metal_orbital_map.values() if site_idx is not None}
    missing_sites = sorted(
        site_idx for site_idx in config.spin_assignment
        if site_idx not in covered_sites and 0 <= site_idx < len(cluster_info.metals)
    )
    if not missing_sites:
        return

    missing_labels = [
        _resolve_metal_site_label(cluster_info, site_idx)
        for site_idx in missing_sites
    ]
    logger.warning(
        "Active-space orbital labels do not map spin-carrying orbitals for sites %s in %s; "
        "broken-symmetry initial guess may be incomplete.",
        ", ".join(missing_labels),
        config.label,
    )
