"""Module 5: Initial Guess Generator

High-level orchestrator that generates all broken-symmetry UHF initial guesses
for a transition metal cluster.  Ties together the spin enumeration, electronic
configuration, BS density-matrix construction, UHF SCF convergence, FCIDUMP
generation, and active-space template loading into a single callable pipeline.

Public API
----------
generate_ini_guesses   -- main entry point (returns a list of GuessResult)
load_active_space_template -- load a CAS template from the knowledge base
converge_uhf_scf       -- converge a single UHF calculation with BS guess
generate_fcidump_mode  -- produce FCIDUMP files for a set of converged UHFs
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ComputationSettings,
    ElectronicConfig,
    SpinIsomer,
    SpinIsomerFamily,
)
from apex_cas.CAS_builder_noncomputing import (
    build_NC_CAS,
    get_common_oxidation_states,
    get_local_spin,
    _match_cluster_template,
)
from .electronic_config import (
    generate_all_configs_v2,
    reduce_configs_by_symmetry,
)
from .spin_config import (
    apply_symmetry_reduction,
    enumerate_spin_isomers,
    label_isomers,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class GuessResult:
    """Container for a single initial-guess result.

    Attributes
    ----------
    config : ElectronicConfig
        The electronic configuration for this guess.
    label : str
        Human-readable label (same as config.label).
    dm : numpy.ndarray or None
        Density matrix (2, nao, nao) if computed.
    energy : float or None
        UHF energy if SCF was converged.
    converged : bool or None
        Whether UHF SCF converged (None if SCF was not attempted).
    s_squared : float or None
        <S^2> expectation value.
    fcidump_path : str or None
        Path to generated FCIDUMP file (if requested).
    """
    config: ElectronicConfig
    label: str = ""
    dm: Optional[np.ndarray] = None
    energy: Optional[float] = None
    converged: Optional[bool] = None
    s_squared: Optional[float] = None
    fcidump_path: Optional[str] = None


@dataclass
class GuessSummary:
    """Summary of a ``generate_ini_guesses`` run.

    Attributes
    ----------
    cluster_info : ClusterInfo
        The cluster description used.
    active_space : CAS
        Active space specification.
    spin_isomers : list[SpinIsomer]
        All enumerated spin isomers satisfying the Sz constraint.
    families : list[SpinIsomerFamily]
        Symmetry-reduced families.
    configs : list[ElectronicConfig]
        All electronic configurations (Cartesian product).
    results : list[GuessResult]
        One GuessResult per configuration (may contain None SCF fields
        if SCF was not run).
    n_total : int
        Total number of configurations enumerated.
    """
    cluster_info: ClusterInfo
    active_space: CAS
    spin_isomers: list[SpinIsomer] = field(default_factory=list)
    families: list[SpinIsomerFamily] = field(default_factory=list)
    configs: list[ElectronicConfig] = field(default_factory=list)
    results: list[GuessResult] = field(default_factory=list)
    n_total: int = 0


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def generate_ini_guesses(
    cluster_info: ClusterInfo,
    *,
    # Active space
    active_space: CAS = None,
    active_space_level: str = "standard",
    # Spin / electronic config
    target_Sz: float = None,
    forced_oxidation: dict = None,
    max_configs: int = None,
    # Symmetry
    symmetry_group: str = None,
    # SCF options
    run_scf: bool = False,
    computation_settings: ComputationSettings = None,
    basis_set: str = "cc-pVDZ",
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    # FCIDUMP options
    fcidump_mode: str = "none",
    fcidump_output_dir: str = None,
    # Template loading
    template_name: str = None,
) -> GuessSummary:
    """Generate broken-symmetry UHF initial guesses for a metal cluster.

    Orchestrates the full workflow:
      1. Load or build the active space (rule-based or from template).
      2. Enumerate spin isomers satisfying the Sz constraint.
      3. Apply symmetry reduction and label families.
      4. Enumerate electronic configurations (oxidation x d-orbital choices).
      5. (Optional) Run UHF SCF for each configuration.
      6. (Optional) Generate FCIDUMP files.

    Parameters
    ----------
    cluster_info : ClusterInfo
        Complete cluster description with metals, charge, target_spin, etc.
    active_space : CAS, optional
        Pre-built active space.  Built automatically if not provided.
    active_space_level : str
        Active-space level if building automatically: "minimal", "standard",
        "extended".
    target_Sz : float, optional
        Override for target total Sz.  Defaults to cluster_info.target_spin.
    forced_oxidation : dict, optional
        ``{site_idx: oxidation_state}`` to pin specific metals.
    max_configs : int, optional
        Cap on total electronic configurations enumerated.
    symmetry_group : str, optional
        Override point group.  Defaults to cluster_info.symmetry_group.
    run_scf : bool
        Whether to actually converge UHF SCF for each config.
    computation_settings : ComputationSettings, optional
        SCF parameters.  Built from defaults if not provided.
    basis_set : str
        Basis set (used only when ``computation_settings`` is None).
    conv_tol : float
        SCF convergence tolerance.
    max_cycle : int
        Maximum SCF iterations.
    fcidump_mode : str
        "none", "full", "active", or "both".
    fcidump_output_dir : str, optional
        Directory for FCIDUMP output.
    template_name : str, optional
        Name of a knowledge-base cluster template to load.

    Returns
    -------
    GuessSummary
        Summary with all spin isomers, families, configs, and results.
    """
    # ── 0. Resolve defaults ────────────────────────────────────────
    if target_Sz is None:
        target_Sz = cluster_info.target_spin
    if symmetry_group is None:
        symmetry_group = cluster_info.symmetry_group

    # ── 1. Active space ────────────────────────────────────────────
    if active_space is None:
        active_space = _build_active_space(
            cluster_info, active_space_level, template_name,
        )

    # ── 2. Enumerate spin isomers ──────────────────────────────────
    oxidation_states = forced_oxidation  # may be None
    spin_isomers = enumerate_spin_isomers(
        cluster_info,
        target_Sz=target_Sz,
        oxidation_states=oxidation_states,
    )

    # ── 3. Symmetry reduction ──────────────────────────────────────
    metal_positions = np.array([m.position for m in cluster_info.metals]) \
        if cluster_info.metals else None
    families = apply_symmetry_reduction(
        spin_isomers, symmetry_group, metal_positions,
    )
    families = label_isomers(families)

    # ── 4. Electronic configurations ───────────────────────────────
    configs = generate_all_configs_v2(
        cluster_info,
        max_configs=max_configs,
        forced_oxidation=forced_oxidation,
    )
    configs = reduce_configs_by_symmetry(configs, cluster_info)

    # ── 5. Build GuessResult objects ───────────────────────────────
    results = []
    for cfg in configs:
        result = GuessResult(
            config=cfg,
            label=cfg.label,
        )
        results.append(result)

    # ── 6. (Optional) Run SCF ──────────────────────────────────────
    if run_scf:
        settings = computation_settings or ComputationSettings(
            basis_set_default=basis_set,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
        )
        results = _run_scf_batch(
            cluster_info, results, settings,
        )

    # ── 7. (Optional) FCIDUMP ──────────────────────────────────────
    if fcidump_mode != "none" and fcidump_output_dir is not None:
        results = _generate_fcidumps(
            cluster_info, active_space, results,
            fcidump_output_dir, fcidump_mode,
            basis_set=basis_set,
        )

    return GuessSummary(
        cluster_info=cluster_info,
        active_space=active_space,
        spin_isomers=spin_isomers,
        families=families,
        configs=configs,
        results=results,
        n_total=len(configs),
    )


# ──────────────────────────────────────────────────────────────────
# Active-space template loading
# ──────────────────────────────────────────────────────────────────

def load_active_space_template(
    template_name: str,
    cluster_info: ClusterInfo = None,
    level: str = "standard",
) -> CAS:
    """Load an active space CAS from a knowledge-base cluster template.

    Parameters
    ----------
    template_name : str
        Template key in the knowledge base (e.g., ``"FeMo_cofactor"``,
        ``"Fe4S4_cubane"``, ``"Fe2S2_dimer"``).
    cluster_info : ClusterInfo, optional
        Cluster info for matching.  If provided, the template's active-space
        parameters are cross-checked against the cluster composition.
    level : str
        Active-space level key within the template (e.g., ``"LLDUC_model"``).

    Returns
    -------
    CAS
        Active space specification loaded from the template.

    Raises
    ------
    ValueError
        If the template is not found or the level key is missing.
    """
    template = _match_cluster_template_by_name(template_name)
    if template is None:
        raise ValueError(
            f"Cluster template '{template_name}' not found in knowledge base."
        )

    as_data = template.get("active_space", {})
    if not as_data:
        raise ValueError(
            f"Template '{template_name}' has no active_space section."
        )

    # Try the requested level first, then fall back to the first available
    level_data = as_data.get(level)
    if level_data is None:
        # Fall back to first key
        for key, val in as_data.items():
            if isinstance(val, dict) and "n_electrons" in val:
                level_data = val
                level = key
                break
    if level_data is None:
        raise ValueError(
            f"Template '{template_name}' has no active space level '{level}'. "
            f"Available: {list(as_data.keys())}"
        )

    n_elec = level_data.get("n_electrons", 0)
    n_orb = level_data.get("n_orbitals", 0)
    description = level_data.get("description", f"Template: {template_name}/{level}")

    cas = CAS(
        n_electrons=n_elec,
        n_orbitals=n_orb,
        description=description,
    )

    # Cross-check if cluster_info provided
    if cluster_info is not None:
        _cross_check_template(template_name, template, cluster_info)

    return cas


def list_available_templates() -> list[str]:
    """Return names of all available cluster templates in the knowledge base."""
    import yaml
    from ._paths import data_file as _kb_file
    path = _kb_file("cluster_templates.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return [k for k in data.keys() if isinstance(data[k], dict)]


# ──────────────────────────────────────────────────────────────────
# UHF SCF convergence
# ──────────────────────────────────────────────────────────────────

def converge_uhf_scf(
    cluster_info: ClusterInfo,
    config: ElectronicConfig,
    *,
    computation_settings: ComputationSettings = None,
    basis_set: str = "cc-pVDZ",
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
) -> GuessResult:
    """Converge a single UHF SCF calculation with a broken-symmetry guess.

    Builds the BS density matrix from the ElectronicConfig, then runs
    PySCF UHF with level-shifting and damping to converge the BS state.

    Parameters
    ----------
    cluster_info : ClusterInfo
        Cluster description.
    config : ElectronicConfig
        Electronic configuration specifying spin assignment and d-orbital
        occupancy.
    computation_settings : ComputationSettings, optional
        Computation settings.  Built from simple args if not provided.
    basis_set : str
        Basis set.
    conv_tol : float
        SCF convergence tolerance.
    max_cycle : int
        Maximum SCF iterations.

    Returns
    -------
    GuessResult
        With energy, converged status, s_squared, and dm populated.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        return GuessResult(
            config=config,
            label=config.label,
            converged=False,
        )

    if computation_settings is None:
        computation_settings = ComputationSettings(
            basis_set_default=basis_set,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
        )

    # Build molecule
    mol = _build_pyscf_mol(cluster_info, computation_settings)

    # High-spin UHF first
    mf = scf.UHF(mol)
    mf.conv_tol = computation_settings.conv_tol
    mf.max_cycle = computation_settings.max_cycle
    mf.kernel()

    if not mf.converged:
        logger.warning("High-spin UHF did not converge for %s", config.label)

    # Build BS initial guess via spin flip
    dm_a, dm_b = mf.make_rdm1()
    dm_a = dm_a.copy()
    dm_b = dm_b.copy()

    # Flip spins on minority sites
    if config.spin_assignment:
        _apply_spin_flip(mol, dm_a, dm_b, config, cluster_info)

    # Run BS-UHF with stabilisation
    mf_bs = scf.UHF(mol)
    mf_bs.conv_tol = computation_settings.conv_tol
    mf_bs.max_cycle = 20
    mf_bs.level_shift = 0.3
    mf_bs.damp = 0.2
    mf_bs.kernel(dm0=(dm_a, dm_b))

    # Continue without level shift for tight convergence
    mf_bs.max_cycle = computation_settings.max_cycle
    mf_bs.level_shift = 0.0
    mf_bs.damp = 0.0
    mf_bs.kernel(dm0=mf_bs.make_rdm1())

    s2 = mf_bs.spin_square()[0] if mf_bs.converged else None

    return GuessResult(
        config=config,
        label=config.label,
        dm=np.array([dm_a, dm_b]),
        energy=mf_bs.e_tot,
        converged=mf_bs.converged,
        s_squared=s2,
    )


# ──────────────────────────────────────────────────────────────────
# FCIDUMP generation mode
# ──────────────────────────────────────────────────────────────────

def generate_fcidump_mode(
    cluster_info: ClusterInfo,
    active_space: CAS,
    results: list[GuessResult],
    output_dir: str,
    *,
    mode: str = "both",
    basis_set: str = "cc-pVDZ",
) -> list[GuessResult]:
    """Generate FCIDUMP files for converged UHF results.

    Parameters
    ----------
    cluster_info : ClusterInfo
        Cluster description.
    active_space : CAS
        Active space specification.
    results : list[GuessResult]
        Results from ``generate_ini_guesses`` or ``converge_uhf_scf``.
    output_dir : str
        Output directory for FCIDUMP files.
    mode : str
        "full", "active", or "both".
    basis_set : str
        Basis set used in the UHF calculations.

    Returns
    -------
    list[GuessResult]
        Updated results with fcidump_path populated.
    """
    from .fcidump import generate_fcidump as _gen_fci

    os.makedirs(output_dir, exist_ok=True)
    updated = []

    for result in results:
        if result.energy is None:
            updated.append(result)
            continue

        # Write temporary NPZ for the FCIDUMP generator
        tmp_npz = os.path.join(output_dir, f"_tmp_{result.label}.npz")
        _save_uhf_npz(cluster_info, result, tmp_npz, basis_set)

        try:
            info = _gen_fci(
                cluster_info, active_space, tmp_npz, output_dir,
                basis_set=basis_set, mode=mode, label=result.label,
            )
            fcidump_path = info.get("active_space") or info.get("full_space")
            result.fcidump_path = fcidump_path
        except Exception as exc:
            logger.warning("FCIDUMP failed for %s: %s", result.label, exc)

        updated.append(result)

    return updated


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _build_active_space(cluster_info, level, template_name=None):
    """Build active space from rule-based construction or template."""
    level_map = {
        "minimal": ActiveSpaceLevel.MINIMAL,
        "standard": ActiveSpaceLevel.STANDARD,
        "extended": ActiveSpaceLevel.EXTENDED,
    }
    as_level = level_map.get(level, ActiveSpaceLevel.STANDARD)

    if template_name is not None:
        try:
            return load_active_space_template(template_name, cluster_info)
        except ValueError:
            logger.warning("Template '%s' not found, falling back to rule-based.",
                           template_name)

    cases, _ = build_NC_CAS(cluster_info, as_level)
    return cases.get("combined") or list(cases.values())[0]


def _match_cluster_template_by_name(template_name):
    """Load a specific template from the knowledge base by name."""
    import yaml
    from ._paths import data_file as _kb_file
    path = _kb_file("cluster_templates.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get(template_name)


def _cross_check_template(template_name, template, cluster_info):
    """Cross-check a template against the actual cluster composition."""
    expected_metals = template.get("metal_centers", [])
    actual_elements = [m.element for m in cluster_info.metals]

    if expected_metals:
        expected_elements = [mc.get("element", "") for mc in expected_metals]
        if sorted(actual_elements) != sorted(expected_elements):
            logger.warning(
                "Template '%s' expects metals %s but cluster has %s.",
                template_name, expected_elements, actual_elements,
            )


def _build_pyscf_mol(cluster_info, settings):
    """Build a PySCF Mole object from ClusterInfo and settings."""
    from pyscf import gto

    atoms = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        atoms.append(f"{elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

    spin_2s = int(round(2 * cluster_info.target_spin))

    mol = gto.M(
        atom="\n".join(atoms),
        charge=cluster_info.total_charge,
        spin=spin_2s,
        basis=settings.get_basis("Fe") if settings.basis_set_per_element
        else settings.basis_set_default,
        symmetry=False,
        verbose=0,
    )
    mol.build()
    return mol


def _apply_spin_flip(mol, dm_a, dm_b, config, cluster_info):
    """Apply spin flip on minority-spin metal sites in-place."""
    minority_sites = config.minority_spin_sites
    if not minority_sites:
        return

    for site_idx in minority_sites:
        if site_idx >= len(cluster_info.metals):
            continue
        atom_idx = cluster_info.metals[site_idx].index
        aoslice = mol.aoslice_by_atom()[atom_idx]
        ao_s, ao_e = aoslice[2], aoslice[3]

        # Swap alpha <-> beta rows and columns
        tmp_a_rows = dm_a[ao_s:ao_e, :].copy()
        tmp_b_rows = dm_b[ao_s:ao_e, :].copy()
        dm_a[ao_s:ao_e, :] = tmp_b_rows
        dm_b[ao_s:ao_e, :] = tmp_a_rows

        tmp_a_cols = dm_a[:, ao_s:ao_e].copy()
        tmp_b_cols = dm_b[:, ao_s:ao_e].copy()
        dm_a[:, ao_s:ao_e] = tmp_b_cols
        dm_b[:, ao_s:ao_e] = tmp_a_cols


def _run_scf_batch(cluster_info, results, settings):
    """Run UHF SCF for a batch of GuessResults."""
    updated = []
    for result in results:
        try:
            scf_result = converge_uhf_scf(
                cluster_info, result.config,
                computation_settings=settings,
            )
            updated.append(scf_result)
        except Exception as exc:
            logger.warning("SCF failed for %s: %s", result.label, exc)
            result.converged = False
            updated.append(result)
    return updated


def _generate_fcidumps(cluster_info, active_space, results,
                        output_dir, mode, basis_set="cc-pVDZ"):
    """Generate FCIDUMP files for all results with converged SCF."""
    return generate_fcidump_mode(
        cluster_info, active_space, results, output_dir,
        mode=mode, basis_set=basis_set,
    )


def _save_uhf_npz(cluster_info, result, npz_path, basis_set):
    """Save UHF result data to NPZ for FCIDUMP generation.

    This is a lightweight saver that stores the information needed
    by the FCIDUMP module.  In production the UHF template saves
    full NPZ files; here we create a compatible stub.
    """
    # We need to actually run a UHF to get MO coefficients.
    # This function stores what we have; the FCIDUMP generator
    # will re-run SCF if needed.
    np.savez(
        npz_path,
        energy=result.energy or 0.0,
        converged=result.converged or False,
    )
