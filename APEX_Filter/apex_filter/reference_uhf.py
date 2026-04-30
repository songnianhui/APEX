"""Reference-state UHF on the active-space Hamiltonian defined by FCIDUMP.

Implements broken-symmetry UHF directly on the active-space integrals,
following the Chan 2026 workflow where all calculations (UHF, UCCSD, etc.)
are performed within the active space defined by the FCIDUMP.

Key difference from full-molecule UHF:
- Uses FCIDUMP one- and two-electron integrals as the "AO" integrals
- Orbital basis = FCIDUMP active-space MO basis (already orthogonal)
- BS initial guess built by manipulating MO occupations (not AO blocks)
- Orbital character from CAS.orbital_labels determines metal-localized MOs
"""

import logging
import re
from dataclasses import dataclass

import numpy as np

from shared.cluster_info_labels import resolve_metal_site_label

from .models import CAS, ClusterInfo, ElectronicConfig

logger = logging.getLogger(__name__)


@dataclass
class ReferenceUHFResult:
    """Result of a reference-state UHF calculation in the active space."""
    config: ElectronicConfig
    energy: float
    converged: bool
    s_squared: float
    mo_coeff: tuple  # (mo_a, mo_b) in FCIDUMP basis, each (norb, norb)
    mo_energy: tuple  # (eps_a, eps_b), each (norb,)
    dm: tuple         # (dm_a, dm_b) in FCIDUMP basis, each (norb, norb)
    mo_occ: tuple     # (occ_a, occ_b)
    diagnostics: dict | None = None


def build_fake_mol(norb: int, nelec: int, ms2: int, ecore: float = 0.0):
    """Build a minimal PySCF Mole for active-space calculations.

    The "fake" mol provides the electron count and spin multiplicity
    needed by PySCF's SCF solvers, without requiring real atomic structure.

    Parameters
    ----------
    norb : int
        Number of active-space orbitals.
    nelec : int
        Number of active-space electrons.
    ms2 : int
        2 * Sz for the reference state.

    Returns
    -------
    pyscf.gto.Mole
    """
    from pyscf import gto

    mol = gto.M()
    mol.nelectron = nelec
    mol.spin = _sanitize_ms2_for_nelec(nelec, ms2)
    mol.incore_anyway = True
    # Override nao_nr to return norb
    mol._nao_nr = norb

    # Build a minimal real molecule so PySCF internals have a valid basis
    # object. The actual one-/two-electron integrals are overridden below,
    # so this atom/basis choice is only a container for SCF machinery.
    mol.atom = [["H", (0.0, 0.0, 0.0)]]
    mol.basis = "sto-3g"
    mol.build(False, False)

    # Patch nao_nr after build
    mol.nao_nr = lambda *args, **kwargs: norb
    mol.nao = norb
    mol.energy_nuc = lambda *args, **kwargs: float(ecore)

    return mol


def build_reference_uhf_solver(fcidump_data, mol_fake, conv_tol=1e-8, max_cycle=2000):
    """Build a PySCF UHF object that uses FCIDUMP integrals.

    Parameters
    ----------
    fcidump_data : FCIDUMPData
        Parsed FCIDUMP with h1e, h2e integrals.
    mol_fake : pyscf.gto.Mole
        Fake mol from build_fake_mol().
    conv_tol : float
        SCF convergence tolerance.
    max_cycle : int
        Maximum SCF iterations.

    Returns
    -------
    pyscf.scf.UHF
    """
    from pyscf import ao2mo, scf

    norb = fcidump_data.norb
    h1e = fcidump_data.h1e
    h2e = fcidump_data.h2e

    mf = scf.UHF(mol_fake)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle

    # Override integral getters
    mf.get_hcore = lambda *args, **kwargs: h1e.copy()
    mf.get_ovlp = lambda *args, **kwargs: np.eye(norb)
    mf.get_init_guess = lambda *args, **kwargs: _build_default_init_guess(mol_fake, norb)

    # Store two-electron integrals in PySCF's internal packed format
    if h2e.ndim == 4:
        mf._eri = ao2mo.restore(8, h2e, norb)
    elif h2e.ndim == 2:
        mf._eri = h2e
    elif h2e.ndim == 1:
        mf._eri = h2e
    else:
        mf._eri = ao2mo.restore(8, h2e, norb)

    return mf


def parse_orbital_metal_mapping(cas: CAS, cluster_info: ClusterInfo) -> dict:
    """Map active-space orbital indices to metal site indices.

    Parses CAS.orbital_labels to determine which spin-carrying metal
    d orbital each active orbital belongs to.

    Parameters
    ----------
    cas : CAS
        Active space with orbital_labels like "Fe1_3d_xy", "S2_3p_x".
    cluster_info : ClusterInfo
        Cluster with metal centers.

    Returns
    -------
    dict
        {orb_idx: metal_site_idx} for metal d-like orbitals that should
        participate in the broken-symmetry spin pattern.
        Non-d metal orbitals and non-metal orbitals map to None.
    """
    if not cas.orbital_labels:
        return {}

    metal_label_map = _build_metal_label_map(cluster_info)
    metal_elements = {}
    for i, metal in enumerate(cluster_info.metals):
        metal_elements.setdefault(metal.element, []).append(i)

    mapping = {}
    for orb_idx, label in enumerate(cas.orbital_labels):
        metal_site = _parse_label_to_metal_site(label, metal_label_map, metal_elements)
        if metal_site is not None and not _is_spin_carrying_metal_orbital(label):
            metal_site = None
        mapping[orb_idx] = metal_site

    return mapping


def build_bs_initial_guess_active_space(
    cas: CAS,
    config: ElectronicConfig,
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
        {orb_idx: metal_site_idx} from parse_orbital_metal_mapping.

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
    cas: CAS,
    config: ElectronicConfig,
    fcidump_data,
    cluster_info: ClusterInfo,
    *,
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    stabilize_cycles: int = 20,
    level_shift: float = 0.3,
    damp: float = 0.2,
    newton_refine: bool = False,
    newton_max_cycle: int = 8,
) -> ReferenceUHFResult:
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
    ReferenceUHFResult
    """
    norb = fcidump_data.norb
    nelec = fcidump_data.nelec
    # Use the aligned local-spin estimate as the high-spin reference.
    ms2_highspin = _compute_high_spin_ms2(cluster_info, config)
    ms2_highspin = _sanitize_ms2_for_nelec(nelec, ms2_highspin)

    try:
        from pyscf import scf
    except ImportError:
        return ReferenceUHFResult(
            config=config, energy=0.0, converged=False,
            s_squared=0.0, mo_coeff=(None, None), mo_energy=(None, None),
            dm=(None, None),
            mo_occ=(None, None),
        )

    # Build orbital-metal mapping
    metal_orbital_map = parse_orbital_metal_mapping(cas, cluster_info)
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

    mol_hs = build_fake_mol(norb, nelec, ms2_highspin, ecore=fcidump_data.ecore)
    mf_hs = build_reference_uhf_solver(fcidump_data, mol_hs, conv_tol, max_cycle)
    mf_hs.callback = _collect_cycle(hs_history)
    mf_hs.kernel()

    if not mf_hs.converged:
        logger.warning(
            "High-spin UHF did not converge for %s; falling back to occupation-based BS guess",
            config.label,
        )
        dm_hs_a, dm_hs_b = build_bs_initial_guess_active_space(
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
    mol_bs = build_fake_mol(norb, nelec, ms2_target, ecore=fcidump_data.ecore)
    mf_bs = build_reference_uhf_solver(fcidump_data, mol_bs, conv_tol, stabilize_cycles)
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

    return ReferenceUHFResult(
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


def run_reference_uhf_batch(
    cas: CAS,
    configs: list,
    fcidump_data,
    cluster_info: ClusterInfo,
    *,
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    stabilize_cycles: int = 20,
    level_shift: float = 0.3,
    damp: float = 0.2,
    newton_refine: bool = False,
    newton_max_cycle: int = 8,
) -> list:
    """Run active-space UHF for a batch of configurations.

    Parameters
    ----------
    cas : CAS
    configs : list[ElectronicConfig]
    fcidump_data : FCIDUMPData
    cluster_info : ClusterInfo
    conv_tol : float
    max_cycle : int
    stabilize_cycles : int
    level_shift : float
    damp : float
    newton_refine : bool
    newton_max_cycle : int

    Returns
    -------
    list[ActiveSpaceSCFResult]
    """
    results = []
    for config in configs:
        logger.info("Running active-space UHF for %s", config.label)
        try:
            result = converge_reference_uhf(
                cas, config, fcidump_data, cluster_info,
                conv_tol=conv_tol,
                max_cycle=max_cycle,
                stabilize_cycles=stabilize_cycles,
                level_shift=level_shift,
                damp=damp,
                newton_refine=newton_refine,
                newton_max_cycle=newton_max_cycle,
            )
        except Exception as e:
            logger.warning("UHF failed for %s: %s", config.label, e)
            result = ReferenceUHFResult(
                config=config, energy=0.0, converged=False,
                s_squared=0.0, mo_coeff=(None, None), mo_energy=(None, None),
                dm=(None, None),
                mo_occ=(None, None),
            )
        results.append(result)
    return results


# Backward-compatible aliases during the refactor.
ActiveSpaceSCFResult = ReferenceUHFResult
build_active_space_scf = build_reference_uhf_solver
converge_active_space_uhf = converge_reference_uhf
run_active_space_uhf_batch = run_reference_uhf_batch


# ══════════════════════════════════════════════════════════════════
# Internal Helpers
# ══════════════════════════════════════════════════════════════════

def _build_metal_label_map(cluster_info: ClusterInfo) -> dict[str, int]:
    """Build exact metal-label aliases for robust orbital-to-site mapping."""
    label_map = {}
    element_counts = {}
    for site_idx, metal in enumerate(cluster_info.metals):
        if metal.label:
            label_map[metal.label] = site_idx
        element_counts[metal.element] = element_counts.get(metal.element, 0) + 1
        label_map[f"{metal.element}{element_counts[metal.element]}"] = site_idx
    return label_map


def _parse_label_to_metal_site(label: str, metal_label_map: dict, metal_elements: dict) -> int | None:
    """Parse an orbital label to determine which metal site it belongs to.

    Examples:
        "Fe1_3d_xy" -> site 0 (if Fe1 is the first Fe)
        "Fe2_3d_z2" -> site 1
        "S1_3p_x"   -> None (not a metal)
        "LIG_1"     -> None
    """
    if not label:
        return None

    token = label.split("_", 1)[0].split(":", 1)[-1].strip()
    if token in metal_label_map:
        return metal_label_map[token]

    # Fall back to the first element+number occurrence anywhere in the token.
    match = re.search(r"([A-Z][a-z]?)(\d+)", token)
    if not match:
        return None

    elem = match.group(1)
    site_num = int(match.group(2)) - 1  # 0-indexed

    if elem not in metal_elements:
        return None

    sites = metal_elements[elem]
    if site_num < len(sites):
        return sites[site_num]

    return None


def _is_spin_carrying_metal_orbital(label: str) -> bool:
    """Return True for metal d-like labels used in BS spin assignment."""
    if "_" not in label:
        return False
    orbital_part = label.split("_", 1)[1]
    return re.search(r"\d+d", orbital_part) is not None


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


def _summarize_final_state_from_dm(cas, config, cluster_info, dm):
    """Summarize the converged/unconverged final state from the DM.

    Returns user-facing descriptors for later ranking and analysis.
    """
    dm_a, dm_b = dm
    metal_orbital_map = parse_orbital_metal_mapping(cas, cluster_info)
    spin_diag = np.diag(dm_a - dm_b)

    site_spin_proxy = {}
    spin_tokens = []
    for site_idx, metal in enumerate(cluster_info.metals):
        orb_indices = [i for i, s in metal_orbital_map.items() if s == site_idx]
        site_spin = float(sum(spin_diag[i] for i in orb_indices))
        metal_label = resolve_metal_site_label(cluster_info, site_idx)
        site_spin_proxy[metal_label] = site_spin
        arrow = "↑" if site_spin >= 0 else "↓"
        spin_tokens.append(f"{metal_label}{arrow}")

    oxidation_tokens = []
    if config.oxidation:
        for site_idx, metal in enumerate(cluster_info.metals):
            if site_idx in config.oxidation.assignments:
                metal_label = resolve_metal_site_label(cluster_info, site_idx)
                oxidation_tokens.append(
                    f"{metal_label}({_to_roman(config.oxidation.assignments[site_idx])})"
                )
    oxidation_label = "+".join(oxidation_tokens) if oxidation_tokens else "ox:none"

    final_d_basin = {}
    d_tokens = []
    if config.d_orbital_assignments:
        for site_idx in sorted(config.d_orbital_assignments):
            metal_label = resolve_metal_site_label(cluster_info, site_idx)
            spin_dir = config.spin_assignment.get(site_idx, +1)
            minority_dm = dm_b if spin_dir == +1 else dm_a
            metal_orbs = [i for i, s in metal_orbital_map.items() if s == site_idx]
            if not metal_orbs:
                continue
            diag = [float(minority_dm[i, i]) for i in metal_orbs]
            target_orb = metal_orbs[int(np.argmax(diag))]
            basin = _short_d_label(cas.orbital_labels[target_orb])
            final_d_basin[metal_label] = basin
            d_tokens.append(f"{metal_label}:{basin}")

    return {
        "final_site_spin_proxy": site_spin_proxy,
        "final_d_basin": final_d_basin,
        "final_state_signature": f"{''.join(spin_tokens)}|{oxidation_label}|{'+'.join(d_tokens) if d_tokens else 'd:none'}",
    }


def _short_d_label(label: str) -> str:
    """Extract a compact user-facing d-orbital basin label from a CAS label."""
    if "_" not in label:
        return label
    orbital_part = label.split("_", 1)[1]
    m = re.search(r"\d+d(.+)$", orbital_part)
    if m:
        return f"d{m.group(1)}"
    return orbital_part


def _to_roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _compute_high_spin_ms2(cluster_info, config):
    """Compute maximum Ms for the high-spin reference state.

    All metal spins aligned in the same direction.
    Ms = sum of all local Si.
    """
    from shared.chem_knowledge import get_local_spin

    total_ms = 0
    for i, metal in enumerate(cluster_info.metals):
        # Use oxidation state from config if available
        if config.oxidation and i in config.oxidation.assignments:
            ox = config.oxidation.assignments[i]
        else:
            from .elec_spin_config_generator import get_common_oxidation_states
            states = get_common_oxidation_states(metal.element)
            ox = states[0] if states else 2
        S = get_local_spin(metal.element, ox)
        total_ms += int(2 * S)  # 2*Ms = 2*S for fully aligned

    return total_ms


def _build_default_init_guess(mol_fake, norb):
    """Build a diagonal UHF initial guess in the FCIDUMP orbital basis."""
    nelec = int(mol_fake.nelectron)
    ms2 = int(mol_fake.spin)
    nalpha = (nelec + ms2) // 2
    nbeta = (nelec - ms2) // 2

    dm_a = np.zeros((norb, norb))
    dm_b = np.zeros((norb, norb))
    dm_a[np.arange(min(nalpha, norb)), np.arange(min(nalpha, norb))] = 1.0
    dm_b[np.arange(min(nbeta, norb)), np.arange(min(nbeta, norb))] = 1.0
    return np.array((dm_a, dm_b))


def _sanitize_ms2_for_nelec(nelec: int, ms2: int) -> int:
    """Project an ms2 guess onto the nearest value compatible with nelec."""
    target = int(round(ms2))
    valid_ms2 = [m for m in range(-nelec, nelec + 1) if (nelec - m) % 2 == 0]
    if target in valid_ms2:
        return target

    sanitized = min(
        valid_ms2,
        key=lambda candidate: (
            abs(candidate - target),
            0 if (target == 0 or np.sign(candidate) == np.sign(target)) else 1,
            abs(candidate),
            0 if (target >= 0 and candidate >= 0) or (target < 0 and candidate <= 0) else 1,
        ),
    )
    logger.warning(
        "Adjusted incompatible ms2=%s to ms2=%s for nelec=%s",
        target,
        sanitized,
        nelec,
    )
    return sanitized


def _warn_if_spin_sites_unmapped(config: ElectronicConfig, metal_orbital_map: dict, cluster_info: ClusterInfo):
    """Warn when the BS pattern names metal sites absent from the active-space labels."""
    covered_sites = {site_idx for site_idx in metal_orbital_map.values() if site_idx is not None}
    missing_sites = sorted(
        site_idx for site_idx in config.spin_assignment
        if site_idx not in covered_sites and 0 <= site_idx < len(cluster_info.metals)
    )
    if not missing_sites:
        return

    missing_labels = [
        resolve_metal_site_label(cluster_info, site_idx)
        for site_idx in missing_sites
    ]
    logger.warning(
        "Active-space orbital labels do not map spin-carrying orbitals for sites %s in %s; "
        "broken-symmetry initial guess may be incomplete.",
        ", ".join(missing_labels),
        config.label,
    )
