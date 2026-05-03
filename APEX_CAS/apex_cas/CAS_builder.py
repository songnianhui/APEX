"""Computation-driven CAS construction helpers.

This module hosts the canonical staged-build helpers behind the
``apex-cas scf`` / ``apex-cas buildcas`` workflow, with
``apex-cas compute`` retained as a convenience wrapper over the same stages.
The intended public surface is intentionally small:

- ``run_scf_initialization(...)``
- ``build_cas_from_mean_field(...)``

The many construction branches below are internal pipeline stages for UNO,
LUO, AVAS, and related orbital-manipulation routes.
"""

import dataclasses
import os
import warnings

import numpy as np
import pyscf  # PySCF: 量子化学计算框架
from pyscf import (  # PySCF: DFT模块、分子构建(gto)、轨道定域化(lo)、Hartree-Fock(scf)
    dft,
    scf,
)

from shared.models import (
    CAS as _CAS,
    ActiveSpaceLevel as _ActiveSpaceLevel,
    AVASConfig as _AVASConfig,
    ClusterInfo as _ClusterInfo,
    ComputationSettings as _ComputationSettings,
    OrbitalGroup as _OrbitalGroup,
)
from shared.element_data import ELEMENTS as _ELEMENTS, get_valence_shells as _get_valence_shells
from shared.molecule_builder import build_mol_with_basis as _shared_build_mol_with_basis
from shared.chem_knowledge import (
    get_ligands_db as _get_ligands_db,
    get_metals_db as _get_metals_db,
    get_valence_s_orbital as _valence_s_for_element,
)
from shared.cluster_info_labels import (
    resolve_explicit_label as _resolve_explicit_label,
    resolve_metal_site_label as _resolve_metal_site_label,
)
from shared.cluster_info_labels import require_authoritative_cluster_info as _require_authoritative_cluster_info
from shared.orbital_methods.localization import (
    build_localization_params_from_settings as _shared_build_localization_params_from_settings,
    localize_orbitals_with_params as _shared_localize_orbitals_with_params,
    split_localize_by_occupations as _shared_split_localize_by_occupations,
)
from shared.orbital_methods.metadata import (
    build_source_method_prefix as _shared_build_source_method_prefix,
)
from shared.orbital_methods.natural_orbitals import (
    compute_unos as _shared_compute_unos,
)
from shared.orbital_methods.projection import (
    compute_projection_weights_for_targets as _shared_compute_projection_weights_for_targets,
)

# ──────────────────────────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────────────────────────


def _build_cas_from_cluster(
    cluster_info: _ClusterInfo,
    computation_settings: _ComputationSettings = None,
    cpt_cas_type: str = "uno",
    localization_method: str = "boys",
    projection_threshold: float = 0.3,
    avas_config: _AVASConfig = None,
    save_dir: str = ".",
) -> tuple:
    """Build CAS directly from a cluster description.

    This is the preferred high-level entry point for computation-driven CAS
    construction. It initializes SCF, then dispatches to the requested
    orbital-construction method.

    Args:
        cluster_info: Cluster description.
        computation_settings: SCF parameters. None produces defaults.
        cpt_cas_type: One of "uno", "uno_type2", "luo", "avas", or "alpha_sl".
        localization_method: "boys" (Boys, default) or "pm" (Pipek-Mezey).
        projection_threshold: Min projection weight for alpha_sl selection.
        avas_config: Configuration for AVAS method (required if cpt_cas_type="avas").
        save_dir: Directory for checkpoint files.

    Returns:
        Tuple of (CAS, mol, mf, chkfile_path) where:
        - CAS: Active space result with MO coefficients and metadata
        - mol: PySCF Mole object
        - mf: Converged SCF object
        - chkfile_path: Path to saved checkpoint file
    """
    if computation_settings is None:
        computation_settings = _ComputationSettings()

    mol, mf, chkfile_path = run_scf_initialization(
        cluster_info,
        computation_settings,
        save_dir=save_dir,
    )
    cas = build_cas_from_mean_field(
        mol,
        mf,
        cluster_info,
        computation_settings=computation_settings,
        cpt_cas_type=cpt_cas_type,
        localization_method=localization_method,
        projection_threshold=projection_threshold,
        avas_config=avas_config,
    )
    return cas, mol, mf, chkfile_path


def _dispatch_computed_cas_builder(
    mol,
    mf,
    cluster_info,
    *,
    cpt_cas_type: str,
    localization_method: str,
    projection_threshold: float,
    source_prefix: str,
    loc_params,
    avas_config: _AVASConfig | None,
):
    """Dispatch to the requested computed-CAS construction pipeline."""
    if cpt_cas_type == "avas":
        avas_cfg = avas_config if avas_config is not None else _AVASConfig()
        cas, _expected_types = _construct_avas(mol, mf, cluster_info, avas_cfg)
        return cas
    if cpt_cas_type == "luo":
        return _construct_luo(
            mol,
            mf,
            cluster_info,
            localization_method=localization_method,
            projection_threshold=projection_threshold,
            source_prefix=source_prefix,
            loc_params=loc_params,
        )
    if cpt_cas_type == "alpha_sl":
        return _construct_alpha_sl(
            mol,
            mf,
            cluster_info,
            loc_method_occ=localization_method,
            loc_method_vir=localization_method,
            projection_threshold=projection_threshold,
            source_prefix=source_prefix,
            loc_params=loc_params,
        )
    if cpt_cas_type == "uno_type2":
        return _construct_uno_type2(
            mol,
            mf,
            cluster_info,
            source_prefix=source_prefix,
            localization_method=localization_method,
            loc_params=loc_params,
        )
    return _construct_uno(
        mol,
        mf,
        cluster_info,
        source_prefix=source_prefix,
        localization_method=localization_method,
        loc_params=loc_params,
    )


def build_cas_from_mean_field(
    mol,
    mf,
    cluster_info: _ClusterInfo,
    computation_settings: _ComputationSettings = None,
    cpt_cas_type: str = "uno",
    localization_method: str = "boys",
    projection_threshold: float = 0.3,
    avas_config: _AVASConfig = None,
):
    """Build CAS from an existing SCF mean-field object.

    This establishes a cleaner boundary between SCF initialization and
    active-space construction. It is the natural future entry point for the
    planned ``apex-cas buildcas`` command.
    """
    if computation_settings is None:
        computation_settings = _ComputationSettings()

    source_prefix = _shared_build_source_method_prefix(computation_settings)
    loc_params = _shared_build_localization_params_from_settings(
        computation_settings,
        localization_method,
    )
    return _dispatch_computed_cas_builder(
        mol,
        mf,
        cluster_info,
        cpt_cas_type=cpt_cas_type,
        localization_method=localization_method,
        projection_threshold=projection_threshold,
        source_prefix=source_prefix,
        loc_params=loc_params,
        avas_config=avas_config,
    )


# ──────────────────────────────────────────────────────────────────
# Initialization: build molecule + run SCF + save checkpoint
# ──────────────────────────────────────────────────────────────────


def run_scf_initialization(
    cluster_info: _ClusterInfo,
    computation_settings: _ComputationSettings,
    save_dir: str = ".",
) -> tuple:
    """Build molecule, run high-spin SCF, and save checkpoint.

    This is the standard initialization step for all computing-based
    CAS construction methods. It builds the PySCF molecule object,
    runs a high-spin SCF calculation, and saves the results to a
    checkpoint file for later reuse.

    Args:
        cluster_info: Cluster description from structure analysis.
        computation_settings: SCF computation parameters.
        save_dir: Directory to save the checkpoint file (default: current directory).

    Returns:
        Tuple of (mol, mf, chkfile_path) where:
        - mol: PySCF Mole object
        - mf: Converged SCF object
        - chkfile_path: Path to the saved checkpoint file
    """

    n_threads = int(os.environ.get("PYSCF_NUM_THREADS", 8))
    pyscf.lib.num_threads(
        n_threads
    )  # PySCF: 设置并行线程数（可通过环境变量 PYSCF_NUM_THREADS 覆盖）

    # If scf_spin is set, temporarily override cluster_info.target_spin for SCF.
    # NOTE: cluster_info.target_spin stores total spin S, but PySCF gto.M(spin=N)
    # expects N = 2*Sz (nalpha - nbeta). For SCF we always use maximal projection Sz=S,
    # so the value is numerically identical: 2*Sz = 2*S.
    original_spin = cluster_info.target_spin
    if computation_settings.scf_spin is not None:
        cluster_info.target_spin = computation_settings.scf_spin

    mol = _shared_build_mol_with_basis(cluster_info, computation_settings)

    # Restore original target_spin after building mol
    cluster_info.target_spin = original_spin

    # Set chkfile path before kernel so PySCF auto-saves
    filename = _make_chkfile_name(cluster_info, computation_settings)
    os.makedirs(save_dir, exist_ok=True)
    chkfile_path = os.path.join(save_dir, filename)

    # Run SCF
    mf = _run_high_spin_scf(mol, computation_settings, chkfile_path)

    return mol, mf, chkfile_path


# ──────────────────────────────────────────────────────────────────
# PySCF interface helpers
# ──────────────────────────────────────────────────────────────────
def _make_chkfile_name(cluster_info: _ClusterInfo, settings: _ComputationSettings) -> str:
    """Generate a descriptive checkpoint filename."""
    return _get_output_stem(cluster_info, settings) + ".chk"


def _get_output_stem(cluster_info: _ClusterInfo, settings: _ComputationSettings) -> str:
    """Generate the common naming stem for all output files.

    The stem follows the pattern: ``{formula}_{method}_{xc}_{basis}``.
    Spaces and pipe characters are replaced with underscores.

    Returns:
        Naming stem string (no file extension).
    """
    formula = getattr(cluster_info, "formula", "cluster")
    method = settings.scf_method
    xc = settings.xc_functional or "none"
    # Use basis_set_file name when available, fall back to basis_set_default
    if settings.basis_set_file:
        basis = os.path.splitext(os.path.basename(settings.basis_set_file))[0]
    else:
        basis = settings.basis_set_default or "unknown"
    stem = f"{formula}_{method}_{xc}_{basis}"
    return stem.replace(" ", "_").replace("|", "_")

def _run_high_spin_scf(mol, settings: _ComputationSettings, chkfile_path: str = None):
    """Run high-spin SCF calculation with two-stage convergence.

    Stage 1 (rough): looser conv_tol, guaranteed level_shift ≥ 0.1 for stability.
    Stage 2 (precise): user-specified conv_tol and level_shift, using Stage 1
    density as initial guess.
    Stage 3 (newton)： Newton-Raphson refinement

    Args:
        mol: PySCF Mole object.
        settings: ComputationSettings with scf parameters and convergence helpers.
        chkfile_path: Path to save checkpoint file.

    Returns:
        Converged SCF object.
    """
    # ── Build mf object ──
    mf = _build_mf_object(mol, settings)

    # ── Apply convergence helpers ──
    mf.init_guess = settings.init_guess
    if settings.scf_damp > 0:
        mf.damp = settings.scf_damp
    if settings.scf_level_shift > 0:
        mf.level_shift = settings.scf_level_shift
    mf.diis_space = settings.diis_space

    # Set chkfile before kernel so PySCF auto-saves
    if chkfile_path:
        mf.chkfile = chkfile_path

    # ── Stage 1: rough convergence (optional) ──
    if settings.scf_stage1_rough:
        stage1_tol = max(1e-4, settings.conv_tol * 100)
        stage1_cycle = min(settings.max_cycle // 2, 30)

        # 当 smearing/frac_occ 已启用时，不强制 level_shift：
        # smearing/frac_occ 本身已在稳定占据数，level_shift 会扭曲 Fermi 面，
        # 导致 shift 移除后 smearing 占据数剧变、密度崩溃。
        stage1_shift = max(settings.scf_level_shift, 0.3)  # 传统稳定化

        print(
            f"\n  SCF Stage 1: conv_tol={stage1_tol:.1e}, level_shift={stage1_shift:.2f}, "
            f"max_cycle={stage1_cycle}"
        )
        mf.conv_tol = stage1_tol
        mf.level_shift = stage1_shift
        mf.max_cycle = stage1_cycle
        mf.verbose = settings.scf_verbose
        mf.conv_check = False  # 禁用 Extra cycle，防止 level_shift 移除导致密度被破坏
        mf.kernel()  # PySCF: 执行 SCF 自洽场迭代（Stage 1）

        # ── Stage 2: precise convergence (only if Stage 1 didn't meet target tol) ──
        if not mf.converged or mf.conv_tol > settings.conv_tol:  # PySCF: 检查 SCF 收敛状态
            print(
                f"\n  SCF Stage 2: conv_tol={settings.conv_tol:.1e}, level_shift={settings.scf_level_shift:.2f}, "
                f"max_cycle={settings.max_cycle}"
            )
            mf.conv_tol = settings.conv_tol
            mf.level_shift = settings.scf_level_shift
            mf.max_cycle = settings.max_cycle
            mf.conv_check = (
                True  # Stage 2 重新启用 Extra cycle（level_shift 与用户设定一致）
            )
            # Keep damp and diis_space from Stage 1
            mf.kernel()  # PySCF: 执行 SCF 自洽场迭代（Stage 2）
    else:
        print("\n  SCF Stage 1: skipped (scf_stage1_rough=False)")
        # Skip Stage 1, run directly with user-specified settings
        mf.conv_tol = settings.conv_tol
        mf.level_shift = settings.scf_level_shift
        mf.max_cycle = settings.max_cycle
        mf.verbose = settings.scf_verbose
        mf.conv_check = True
        mf.kernel()  # PySCF: 执行 SCF 自洽场迭代

    # ── Stage 3: Newton-Raphson refinement ──
    # 剥离 smearing/frac_occ 装饰器，对 Stage 2 密度做二阶优化。
    # 当 scf_stage3_newton=False 时跳过（用于匹配 ORCA 等程序的 SCF 行为）。
    if settings.scf_stage3_newton:
        print("\n  SCF Stage 3: Newton-Raphson refinement")

        newton_settings = dataclasses.replace(
            settings, frac_occ=False, smearing_method="none"
        )
        mf_newton_base = _build_mf_object(mol, newton_settings)
        if chkfile_path:
            mf_newton_base.chkfile = chkfile_path
        mf_newton = mf_newton_base.newton()  # PySCF: Newton-Raphson (SOSCF) 装饰器

        mf_newton.conv_tol = settings.newton_conv_tol
        mf_newton.max_cycle = settings.newton_max_cycle
        mf_newton.verbose = settings.scf_verbose
        print(f"    conv_tol={settings.newton_conv_tol:.1e}, max_cycle={settings.newton_max_cycle}")

        dm0 = mf.make_rdm1()  # PySCF: 用 Stage 2 的密度矩阵作为初猜
        mf_newton.kernel(dm0=dm0)  # PySCF: 执行 Newton-Raphson SCF 迭代
        mf = mf_newton
    else:
        print("\n  SCF Stage 3: skipped (scf_stage3_newton=False)")

    # ── Check convergence ──
    if not mf.converged:  # PySCF: 检查 SCF 收敛状态
        print(f"\n{'!' * 60}")
        print("  WARNING: SCF did NOT converge!")
        print(f"  Energy: {mf.e_tot:.10f} Hartree")  # PySCF: SCF 总能量
        print(f"  Cycles: {mf.cycles}")  # PySCF: SCF 迭代次数
        print(f"{'!' * 60}")

        # 1) 如果配置允许，直接继续
        if settings.scf_allow_unconverged:
            print("  scf_allow_unconverged=True, continuing with unconverged result.")
        # 2) 否则明确失败，由配置控制是否允许继续
        else:
            raise RuntimeError(
                "SCF did not converge. "
                "Set scf_allow_unconverged: true in YAML to continue."
            )

    # ── Solvent energy traceability ──
    if hasattr(mf, "with_solvent"):
        e_solvent = mf.scf_summary.get("e_solvent", 0.0)
        print(f"  Solvent energy (ddCOSMO): {e_solvent:.10f} Hartree")
        print(f"  Gas-phase energy:         {mf.e_tot - e_solvent:.10f} Hartree")
    else:
        print("  No solvent model active.")

    return mf


def _build_mf_object(mol, settings: _ComputationSettings):
    """Build the SCF mean-field object with relativistic and solvation corrections.

    Returns the mf object without setting convergence parameters, so that the
    caller can configure two-stage convergence separately.
    """
    # Base SCF object
    if settings.scf_method == "uhf":
        mf = scf.UHF(mol)  # PySCF: 创建非限制性 Hartree-Fock 对象
    elif settings.scf_method == "uks":
        mf = dft.UKS(mol)  # PySCF: 创建非限制性 Kohn-Sham DFT 对象
        mf.xc = settings.xc_functional
    else:
        raise ValueError(f"Unknown scf_method: {settings.scf_method}")

    # Relativistic correction: spin-free X2C via sfx2c1e decorator
    # IMPORTANT: x2c.UKS/UHF uses 2-component spinor basis (wrong for UNO).
    # We use UKS/UHF.sfx2c1e() which keeps everything in regular AO basis.
    if settings.relativistic == "sf-x2c":
        mf = (
            mf.sfx2c1e()
        )  # PySCF: spin-free X2C 标量相对论修正装饰器，替换 get_hcore 为 X2C 修正版
    elif settings.relativistic == "dkh":
        # PySCF: mol.set(relativistic="DKH") 告知 PySCF 使用 DKH 标量相对论修正
        mol.set(relativistic="DKH")
        if settings.scf_method == "uks":
            mf = dft.UKS(mol)  # PySCF: DKH 分支重建 UKS
            mf.xc = settings.xc_functional
        else:
            mf = scf.UHF(mol)  # PySCF: DKH 分支重建 UHF

    # Solvation model (wraps mf)
    if settings.solvation_model == "ddcosmo":
        from pyscf import solvent

        mf = solvent.ddcosmo.ddcosmo_for_scf(mf)  # PySCF: ddCOSMO 溶剂模型装饰器
        mf.with_solvent.epsilon = settings.solvation_epsilon

    # Density fitting (RI-J/K)
    if settings.density_fit:
        df_kwargs = {}
        if settings.density_fit_auxbasis:
            df_kwargs["auxbasis"] = settings.density_fit_auxbasis
        if settings.density_fit_only_dfj:
            df_kwargs["only_dfj"] = True
        mf = mf.density_fit(**df_kwargs)  # PySCF: RI-J/K with optional auxbasis and only_dfj

    # DFT integration grid
    if settings.grids_level != 3:
        mf.grids.level = settings.grids_level  # PySCF: DFT grid fineness (0-9)
    if settings.grids_small_rho_cutoff != 1e-7:
        mf.grids.small_rho_cutoff = settings.grids_small_rho_cutoff  # PySCF: density pruning cutoff (0 = disable)
    if settings.grids_prune != "nwchem":
        from pyscf.dft import gen_grid
        _prune_map = {
            "nwchem": gen_grid.nwchem_prune,
            "sg1": gen_grid.sg1_prune,
            "treutler": gen_grid.treutler_prune,
            "none": None,
        }
        mf.grids.prune = _prune_map.get(settings.grids_prune, gen_grid.nwchem_prune)

    # Fractional occupation decorator
    if settings.frac_occ:
        mf = scf.addons.frac_occ(mf)  # PySCF: 分数占据装饰器

    # Fermi/Gauss smearing decorator
    if settings.smearing_method != "none":
        mf = scf.addons.smearing_(  # PySCF: smearing 装饰器（in-place）
            mf, sigma=settings.smearing_sigma, method=settings.smearing_method
        )

    return mf


# ──────────────────────────────────────────────────────────────────
# UNO pipeline
# ──────────────────────────────────────────────────────────────────
def _construct_uno(
    mol,
    mf,
    cluster_info,
    source_prefix="UHF",
    localization_method="boys",
    loc_params=None,
):
    """Restricted UNO pipeline: UKS → UNO → split-localize → select.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object.
        cluster_info: ClusterInfo for projection-based selection.
        source_prefix: Method string prefix (e.g. "UKS-B3LYP").
        localization_method: "boys" (default) or "pm".
        loc_params: Optional dict of PM localization parameters.
    """
    # Step 2: Compute UNOs from alpha+beta density matrix
    mo_coeff_uno, occ_uno = _shared_compute_unos(mol, mf)

    # Step 3: Split-localize by occupation blocks
    mo_coeff_loc, orbital_labels = _split_localize(
        mol, mo_coeff_uno, occ_uno, cluster_info,
        method=localization_method, loc_params=loc_params,
    )

    # Step 4: Select active orbitals — ALWAYS NOON-based
    occ_lo, occ_hi = 0.02, 1.98
    active_indices = [i for i in range(len(occ_uno)) if occ_lo <= occ_uno[i] <= occ_hi]
    n_electrons = int(round(sum(occ_uno[active_indices])))
    print(
        f"  [UNO] NOON-based selection: {len(active_indices)} orbitals, "
        f"{n_electrons} electrons (range=[{occ_lo}, {occ_hi}])"
    )

    # Compute projection weights for report (if cluster_info available)
    proj_weights = None
    proj_wt_metal = None
    proj_wt_bridging = None
    if cluster_info is not None:
        proj_weights, proj_wt_metal, proj_wt_bridging = _compute_all_projection_weights(
            mol, mo_coeff_loc, cluster_info
        )

    # Extract active orbitals
    mo_active = mo_coeff_loc[:, active_indices]
    occ_active = occ_uno[active_indices]
    labels_active = [orbital_labels[i] for i in active_indices]

    # Determine selection method label for metadata
    actual_selection_method = "noon"  # always NOON for UNO pipeline

    return _CAS(
        mo_coeff_alpha=mo_active.copy(),
        mo_coeff_beta=mo_active.copy(),  # Same for restricted
        occupations=occ_active,
        orbital_labels=labels_active,
        cpt_cas_type="uno",
        n_electrons=n_electrons,
        n_orbitals=len(active_indices),
        source_method=f"{source_prefix}/UNO",
        mo_coeff_full=mo_coeff_loc,
        occupations_full=occ_uno,
        orbital_labels_full=orbital_labels,
        active_indices=active_indices,
        selection_method=actual_selection_method,
        projection_weights=proj_weights,
        projection_weights_metal=proj_wt_metal,
        projection_weights_bridging=proj_wt_bridging,
    )


# ──────────────────────────────────────────────────────────────────
# UNO type2 pipeline (core+active merged localization)
# ──────────────────────────────────────────────────────────────────


def _construct_uno_type2(
    mol,
    mf,
    cluster_info,
    source_prefix="UHF",
    localization_method="boys",
    loc_params=None,
):
    """UNO pipeline with core+active merged localization (2-group).

    Identical to ``_construct_uno`` except that core and active orbitals
    are localized as a single block, producing only two groups
    (core+active, virtual).

    After merging core+active for localization, the original UNO occupations
    no longer correspond to the localized orbitals.  We recompute occupations
    by projecting the total density matrix onto the localized orbital basis.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object.
        cluster_info: ClusterInfo for projection-based selection.
        source_prefix: Method string prefix (e.g. "UKS-B3LYP").
        localization_method: "boys" (default) or "pm".
        loc_params: Optional dict of PM localization parameters.
    """
    # Step 2: Compute UNOs from alpha+beta density matrix
    mo_coeff_uno, occ_uno = _shared_compute_unos(mol, mf)

    # Step 3: Split-localize with core+active merged
    mo_coeff_loc, orbital_labels = _split_localize(
        mol, mo_coeff_uno, occ_uno, cluster_info,
        method=localization_method, loc_params=loc_params,
        merge_core_active=True,
    )

    # Step 3b: Recompute occupations for localized orbitals.
    # When core+active are merged, the localization unitary mixes orbitals
    # with very different UNO occupations (~2 vs ~1).  The original occ_uno
    # no longer matches the localized orbitals.  We recompute by projecting
    # the total density matrix onto the localized MO basis:
    #   occ_loc[i] = C_loc[:,i]^T @ S @ D_total @ S @ C_loc[:,i]
    S = mol.intor_symmetric("int1e_ovlp")
    dm = mf.make_rdm1()
    if isinstance(dm, (list, tuple)) and len(dm) == 2:
        dm_total = dm[0] + dm[1]
    elif isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm_total = dm[0] + dm[1]
    else:
        dm_total = np.asarray(dm)

    SDS = S @ dm_total @ S
    occ_loc = np.sum(mo_coeff_loc * (SDS @ mo_coeff_loc), axis=0)

    # Step 4: Select active orbitals — NOON-based on recomputed occupations
    occ_lo, occ_hi = 0.02, 1.98
    active_indices = [i for i in range(len(occ_loc)) if occ_lo <= occ_loc[i] <= occ_hi]
    n_electrons = int(round(sum(occ_loc[active_indices])))
    print(
        f"  [UNO type2] NOON-based selection (recomputed occ): {len(active_indices)} orbitals, "
        f"{n_electrons} electrons (range=[{occ_lo}, {occ_hi}])"
    )

    # Compute projection weights for report (if cluster_info available)
    proj_weights = None
    proj_wt_metal = None
    proj_wt_bridging = None
    if cluster_info is not None:
        proj_weights, proj_wt_metal, proj_wt_bridging = _compute_all_projection_weights(
            mol, mo_coeff_loc, cluster_info
        )

    # Extract active orbitals
    mo_active = mo_coeff_loc[:, active_indices]
    occ_active = occ_loc[active_indices]
    labels_active = [orbital_labels[i] for i in active_indices]

    # Determine selection method label for metadata
    actual_selection_method = "noon"  # always NOON for UNO pipeline

    return _CAS(
        mo_coeff_alpha=mo_active.copy(),
        mo_coeff_beta=mo_active.copy(),  # Same for restricted
        occupations=occ_active,
        orbital_labels=labels_active,
        cpt_cas_type="uno_type2",
        n_electrons=n_electrons,
        n_orbitals=len(active_indices),
        source_method=f"{source_prefix}/UNO_type2",
        mo_coeff_full=mo_coeff_loc,
        occupations_full=occ_loc,
        orbital_labels_full=orbital_labels,
        active_indices=active_indices,
        selection_method=actual_selection_method,
        projection_weights=proj_weights,
        projection_weights_metal=proj_wt_metal,
        projection_weights_bridging=proj_wt_bridging,
    )


def _construct_luo(
    mol, mf, cluster_info, localization_method: str = "boys",
    projection_threshold: float = 0.3, source_prefix="UHF",
    loc_params=None,
):
    """Unrestricted LUO pipeline: localize alpha and beta separately.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object (UHF or UKS).
        cluster_info: ClusterInfo for projection-based selection.
        localization_method: "boys" (Boys, default) or "pm" (Pipek-Mezey).
        projection_threshold: Min projection weight for active orbital selection.
        source_prefix: Method string prefix (e.g. "UHF").
        loc_params: Optional dict of PM localization parameters.
    """

    # Get alpha and beta MOs
    mo_alpha = mf.mo_coeff[0]  # PySCF: alpha 轨道系数矩阵
    mo_beta = mf.mo_coeff[1]  # PySCF: beta 轨道系数矩阵

    # Determine occupation
    n_alpha = mol.nelec[0]  # PySCF: alpha 电子数
    n_beta = mol.nelec[1]  # PySCF: beta 电子数

    n_occ_a = n_alpha
    n_occ_b = n_beta

    n_tot = mo_alpha.shape[1]

    occ_alpha = np.zeros(n_tot)
    occ_alpha[:n_occ_a] = 1.0
    occ_beta = np.zeros(n_tot)
    occ_beta[:n_occ_b] = 1.0

    # Localize alpha occupied
    loc_a = _shared_localize_orbitals_with_params(mol, mo_alpha[:, :n_occ_a], method=localization_method, loc_params=loc_params)
    # Localize alpha virtual
    loc_a_vir = _shared_localize_orbitals_with_params(
        mol, mo_alpha[:, n_occ_a:], method=localization_method, loc_params=loc_params
    )

    # Localize beta occupied
    loc_b = _shared_localize_orbitals_with_params(mol, mo_beta[:, :n_occ_b], method=localization_method, loc_params=loc_params)
    # Localize beta virtual
    loc_b_vir = _shared_localize_orbitals_with_params(
        mol, mo_beta[:, n_occ_b:], method=localization_method, loc_params=loc_params
    )

    # Select active subset from localized orbitals
    # Combine occupied and virtual, then select by character
    all_loc_a = np.hstack([loc_a, loc_a_vir])
    all_loc_b = np.hstack([loc_b, loc_b_vir])

    # Select active orbitals by projection onto metal-d + bridging-p
    if cluster_info is not None:
        active_idx_a = _select_by_projection_threshold(mol, all_loc_a, cluster_info, threshold=projection_threshold)
        active_idx_b = _select_by_projection_threshold(mol, all_loc_b, cluster_info, threshold=projection_threshold)
        luo_selection_method = "character"
    else:
        n_tot = all_loc_a.shape[1]
        active_idx_a = list(range(n_tot))
        active_idx_b = list(range(n_tot))
        luo_selection_method = "all"

    mo_active_a = all_loc_a[:, active_idx_a]
    mo_active_b = all_loc_b[:, active_idx_b]

    # Generate labels
    labels_a = [f"LUO_a_{i}" for i in range(len(active_idx_a))]
    labels = labels_a  # Use alpha labels as primary

    # Compute projection weights for report (if cluster_info available)
    proj_weights = None
    proj_wt_metal = None
    proj_wt_bridging = None
    if cluster_info is not None:
        proj_weights, proj_wt_metal, proj_wt_bridging = _compute_all_projection_weights(
            mol, all_loc_a, cluster_info
        )

    return _CAS(
        mo_coeff_alpha=mo_active_a.copy(),
        mo_coeff_beta=mo_active_b.copy(),
        occupations=None,  # LUO doesn't have UNO occupations
        orbital_labels=labels,
        cpt_cas_type="luo",
        n_electrons=mol.nelec[0] + mol.nelec[1],  # PySCF: alpha + beta 电子总数
        n_orbitals=len(active_idx_a),
        source_method=f"{source_prefix}/LUO",
        mo_coeff_full=all_loc_a,
        occupations_full=occ_alpha,
        orbital_labels_full=labels,
        active_indices=active_idx_a,
        selection_method=luo_selection_method,
        projection_weights=proj_weights,
        projection_weights_metal=proj_wt_metal,
        projection_weights_bridging=proj_wt_bridging,
    )


# ──────────────────────────────────────────────────────────────────
# Alpha split-localization pipeline (Chan 2014 style)
# ──────────────────────────────────────────────────────────────────


def _construct_alpha_sl(
    mol,
    mf,
    cluster_info,
    loc_method_occ: str = "pm",
    loc_method_vir: str = "pm",
    projection_threshold: float = 0.3,
    source_prefix: str = "UHF",
    loc_params=None,
):
    """Chan-style alpha split-localization pipeline.

    UKS high-spin -> alpha MO -> split-localize (occ/vir separately)
    -> select by projection.

    Based on Sharma, Sivalingam, Neese & Chan, arXiv:1408.5080 (2014).

    Args:
        mol, mf, cluster_info: 标准 APEX 参数
        loc_method_occ: 占据轨道定域化方法 ("pm", "boys")
        loc_method_vir: 虚轨道定域化方法 ("pm", "boys")
        projection_threshold: 活性轨道 projection 权重阈值
        source_prefix: 报告用前缀
        loc_params: Optional dict of PM localization parameters.
    """
    # 1. 提取 alpha MO
    mo_alpha = mf.mo_coeff[0]  # PySCF: alpha 轨道系数矩阵
    n_alpha = mol.nelec[0]  # PySCF: alpha 电子数

    # 2. Split-localize: occupied 和 virtual 使用各自的方法
    loc_a_occ = _shared_localize_orbitals_with_params(mol, mo_alpha[:, :n_alpha], method=loc_method_occ, loc_params=loc_params)
    loc_a_vir = _shared_localize_orbitals_with_params(mol, mo_alpha[:, n_alpha:], method=loc_method_vir, loc_params=loc_params)
    all_loc = np.hstack([loc_a_occ, loc_a_vir])

    # 3. 构建 occupation 数组：将总密度矩阵 D_α + D_β 投影到 localized alpha MO 基
    n_tot = all_loc.shape[1]
    S = mol.intor('int1e_ovlp')
    D_beta = mf.mo_coeff[1][:, :mol.nelec[1]] @ mf.mo_coeff[1][:, :mol.nelec[1]].T
    SDbetaS = S @ D_beta @ S
    beta_occ_diag = np.einsum('mi,mi->i', all_loc, SDbetaS @ all_loc)
    occ = np.zeros(n_tot)
    occ[:n_alpha] = 1.0                         # alpha 占据部分
    occ += np.clip(beta_occ_diag, 0.0, None)   # 加上 beta 贡献

    # 4. 通过 projection 权重选择活性轨道
    if cluster_info is not None:
        active_indices = _select_by_projection_threshold(
            mol, all_loc, cluster_info, threshold=projection_threshold
        )
    else:
        active_indices = list(range(n_tot))

    # 5. 计算 projection 权重（用于报告）
    proj_weights = None
    proj_wt_metal = None
    proj_wt_bridging = None
    if cluster_info is not None:
        proj_weights, proj_wt_metal, proj_wt_bridging = _compute_all_projection_weights(
            mol, all_loc, cluster_info
        )

    # 6. 生成标签
    n_occ = n_alpha
    n_vir = n_tot - n_alpha
    labels = [f"occ_{i}" for i in range(n_occ)] + [f"vir_{i}" for i in range(n_vir)]

    # Add atomic character labels if cluster_info available
    if cluster_info is not None:
        labels = _assign_character_labels(mol, all_loc, labels, cluster_info)

    # 7. 活性空间信息
    mo_active = all_loc[:, active_indices]
    # 电子数：直接用总占据数之和（alpha + beta）
    n_electrons = int(round(float(np.sum(occ[active_indices]))))

    print(
        f"  [alpha_sl] Projection-based selection: {len(active_indices)} orbitals, "
        f"{n_electrons} electrons (threshold={projection_threshold})"
    )

    return _CAS(
        mo_coeff_alpha=mo_active.copy(),
        mo_coeff_beta=mo_active.copy(),  # 同一组空间轨道
        occupations=occ[active_indices],
        orbital_labels=[labels[i] for i in active_indices],
        cpt_cas_type="alpha_sl",
        n_electrons=n_electrons,
        n_orbitals=len(active_indices),
        source_method=f"{source_prefix}/alpha_sl",
        mo_coeff_full=all_loc,
        occupations_full=occ,
        orbital_labels_full=labels,
        active_indices=active_indices,
        selection_method="character",
        projection_weights=proj_weights,
        projection_weights_metal=proj_wt_metal,
        projection_weights_bridging=proj_wt_bridging,
    )


# ──────────────────────────────────────────────────────────────────
# Core computational functions
# ──────────────────────────────────────────────────────────────────


def _split_localize(
    mol,
    mo_coeff,
    occupations,
    cluster_info=None,
    occ_threshold_core: float = 1.98,
    occ_threshold_virtual: float = 0.02,
    method: str = "boys",
    loc_params=None,
    merge_core_active: bool = False,
):
    """Split-localize UNOs by occupation blocks.

    Partitions orbitals into core (occ > 1.98), active (0.02 < occ < 1.98),
    and virtual (occ < 0.02) blocks. Applies localization separately
    to each block.

    When ``merge_core_active=True``, core and active are merged into a single
    block for localization, producing two groups (core+active, virtual)
    instead of three.

    Args:
        mol: PySCF Mole object.
        mo_coeff: MO coefficient matrix.
        occupations: UNO occupation numbers.
        cluster_info: Optional cluster info for labeling.
        occ_threshold_core: Occupation threshold for core orbitals.
        occ_threshold_virtual: Occupation threshold for virtual orbitals.
        method: Localization method: "boys" (Boys, default) or
            "pm" (Pipek-Mezey).
        loc_params: Optional dict of PM localization parameters.
        merge_core_active: If True, merge core and active into one block.

    Returns:
        (localized_mo_coeff, orbital_labels)
    """
    localized, labels = _shared_split_localize_by_occupations(
        mol,
        mo_coeff,
        occupations,
        occ_threshold_core=occ_threshold_core,
        occ_threshold_virtual=occ_threshold_virtual,
        method=method,
        loc_params=loc_params,
        merge_core_active=merge_core_active,
    )

    # Add atomic character labels if cluster_info available
    if cluster_info is not None:
        labels = _assign_character_labels(mol, localized, labels, cluster_info)

    return localized, labels
def _build_target_ao_subspace(mol, cluster_info):
    """Build target AO indices using role-filtered valence (n, l) shell matching.

    Uses ``ao_shell_analysis.get_valence_shells()`` to determine the valence shells
    for each target atom, then **filters by atom role** before matching AO
    basis functions:

    * **Metals**: keep d/f shells only (l >= 2) — e.g. Fe 3d, Ce 4f + 5d
    * **Bridging atoms**: keep p shells only (l == 1) — e.g. S 3p, O 2p

    This follows Chan's convention of projecting only onto metal d/f and
    bridging p orbitals.  Metal s-shells (e.g. Fe 4s, which participates in
    σ bonding) and bridging s-shells (e.g. S 3s, which behaves as core-like)
    are excluded from the projection subspace.

    Args:
        mol: PySCF Mole object.
        cluster_info: ClusterInfo with metals and bridging_atoms.

    Returns:
        dict with keys ``"all"``, ``"metal"``, ``"bridging"``, each mapping
        to a sorted list of target AO indices.  ``"all"`` is the union of
        ``"metal"`` and ``"bridging"``.
    """
    _require_authoritative_cluster_info(
        cluster_info,
        context="Projection target construction",
    )

    metal_indices = {
        m.index for m in cluster_info.metals
        if m.projection_role == "metal_df"
    }
    bridge_indices = {
        b.index for b in cluster_info.bridging_atoms
        if b.projection_role == "bridging_p"
    }

    L_CHAR_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3}

    # Build per-atom valence (n, l_int) sets, filtered by atom role
    valence_nl_map: dict[int, set[tuple[int, int]]] = {}
    for idx in metal_indices | bridge_indices:
        elem = mol.atom_symbol(idx)
        Z = _ELEMENTS.get(elem, 0)
        if Z > 0:
            val_shells = _get_valence_shells(Z)
            all_nl = {(n, L_CHAR_TO_INT[lc]) for n, lc in val_shells}
            # Filter by atom role: metals keep d/f only, bridges keep p only
            if idx in metal_indices:
                valence_nl_map[idx] = {
                    (principal_n, ang_mom)
                    for principal_n, ang_mom in all_nl
                    if ang_mom >= 2
                }
            else:  # bridging
                valence_nl_map[idx] = {
                    (principal_n, ang_mom)
                    for principal_n, ang_mom in all_nl
                    if ang_mom == 1
                }

    ao_loc = mol.ao_loc_nr()  # PySCF: per-shell AO offset array
    ao_labels = mol.ao_labels()  # PySCF: AO labels, e.g. "0 Fe 3dxy"
    metal_ao_indices = []
    bridging_ao_indices = []

    for ish in range(mol.nbas):
        ia = mol.bas_atom(ish)  # PySCF: atom index for this shell
        ang_mom = mol.bas_angular(ish)  # PySCF: angular momentum quantum number
        ao_start = ao_loc[ish]
        ao_end = ao_loc[ish + 1]

        valence_nl = valence_nl_map.get(ia)
        if valence_nl is None:
            continue

        # Parse principal quantum number n from the AO label
        # Label format: "0 Fe 3dxy" → parts[2] = "3dxy" → n = 3
        label = ao_labels[ao_start]
        parts = label.split()
        n_shell = None
        if len(parts) >= 3:
            shell_str = parts[2]
            i = 0
            while i < len(shell_str) and shell_str[i].isdigit():
                i += 1
            n_shell = int(shell_str[:i]) if i > 0 else None

        if n_shell is not None and (n_shell, ang_mom) in valence_nl:
            ao_range = list(range(ao_start, ao_end))
            if ia in metal_indices:
                metal_ao_indices.extend(ao_range)
            else:
                bridging_ao_indices.extend(ao_range)

    metal_ao_indices = sorted(set(metal_ao_indices))
    bridging_ao_indices = sorted(set(bridging_ao_indices))
    return {
        "all": sorted(set(metal_ao_indices) | set(bridging_ao_indices)),
        "metal": metal_ao_indices,
        "bridging": bridging_ao_indices,
    }


def _compute_all_projection_weights(mol, mo_coeff, cluster_info):
    """Compute projection weight of every MO onto metal d/f + bridging p subspace.

    Returns decomposed weights so the caller can distinguish metal vs bridging
    contributions.

    Returns:
        tuple of (proj_all, proj_metal, proj_bridging), each ndarray (nmo,).
    """
    target_dict = _build_target_ao_subspace(mol, cluster_info)
    n_mo = mo_coeff.shape[1]
    if not target_dict["all"]:
        return np.zeros(n_mo), np.zeros(n_mo), np.zeros(n_mo)

    proj_all = _shared_compute_projection_weights_for_targets(mol, mo_coeff, target_dict["all"])
    proj_metal = _shared_compute_projection_weights_for_targets(mol, mo_coeff, target_dict["metal"])
    proj_bridging = _shared_compute_projection_weights_for_targets(mol, mo_coeff, target_dict["bridging"])
    return proj_all, proj_metal, proj_bridging


def _select_by_projection(mol, mo_coeff, n_active, cluster_info):
    """Select active orbitals by projecting onto metal-d + bridging-p AO subsets."""
    target_dict = _build_target_ao_subspace(mol, cluster_info)
    target_ao_indices = target_dict["all"]

    if not target_ao_indices:
        return list(range(min(n_active, mo_coeff.shape[1])))

    projections = _shared_compute_projection_weights_for_targets(mol, mo_coeff, target_ao_indices)

    # Select top n_active by projection weight
    selected = np.argsort(-projections)[:n_active]
    return sorted(selected.tolist())


def _select_by_projection_threshold(
    mol, mo_coeff, cluster_info, threshold: float = 0.3
):
    """Select orbitals with projection weight > threshold onto metal-d + bridging-p.

    Unlike ``_select_by_projection`` which returns exactly *n_active* orbitals,
    this function returns all orbitals whose projection weight onto the target
    AO subspace exceeds *threshold*.

    Args:
        mol: PySCF Mole object.
        mo_coeff: MO coefficient matrix.
        cluster_info: ClusterInfo with metals and bridging_atoms.
        threshold: Minimum projection weight (default 0.05).

    Returns:
        Sorted list of orbital indices.
    """
    target_dict = _build_target_ao_subspace(mol, cluster_info)
    target_ao_indices = target_dict["all"]

    if not target_ao_indices:
        return list(range(mo_coeff.shape[1]))

    projections = _shared_compute_projection_weights_for_targets(mol, mo_coeff, target_ao_indices)

    selected = [int(i) for i in range(mo_coeff.shape[1]) if projections[i] > threshold]
    return sorted(selected)


def _assign_character_labels(mol, mo_coeff, base_labels, cluster_info):
    """Assign character labels (e.g., 'Fe1_3dxy', 'S3_3px') to localized orbitals.

    Uses PySCF's ``mol.ao_labels()`` to extract the dominant AO type for each
    localized orbital.
    """
    metal_map = {
        m.index: _resolve_metal_site_label(cluster_info, site_idx)
        for site_idx, m in enumerate(cluster_info.metals)
    }
    bridge_map = {
        b.index: _resolve_explicit_label(
            getattr(b, "label", ""),
            f"{b.element}{b.index + 1}",
            cluster_info=cluster_info,
            context=f"bridging atom {b.index}",
        )
        for b in cluster_info.bridging_atoms
    }
    target_atoms = dict(metal_map)
    target_atoms.update(bridge_map)

    aoslices = mol.aoslice_by_atom()  # PySCF: 获取各原子的 AO 起止索引
    ao_labels = mol.ao_labels()  # PySCF: 获取所有 AO 轨道标签

    for i in range(mo_coeff.shape[1]):
        mo_i = mo_coeff[:, i]

        # Find the atom with largest contribution
        best_atom = -1
        best_contrib = 0.0

        for atom_idx in target_atoms:
            if atom_idx < len(aoslices):
                _, _, ao_s, ao_e = aoslices[atom_idx]
                contrib = np.sum(mo_i[ao_s:ao_e] ** 2)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_atom = atom_idx

        if best_atom >= 0 and best_contrib > 0.1:
            # Get the dominant AO type from ao_labels
            _, _, ao_s, ao_e = aoslices[best_atom]
            local_coeffs = mo_i[ao_s:ao_e] ** 2
            dominant_local = np.argmax(local_coeffs)
            dominant_ao_idx = ao_s + dominant_local

            dominant_label = (
                ao_labels[dominant_ao_idx]
                if dominant_ao_idx < len(ao_labels) else ""
            )
            parts = dominant_label.split()
            orb_type = parts[-1] if len(parts) > 0 else ""

            atom_label = target_atoms.get(best_atom, f"atom{best_atom}")

            base_labels[i] = (
                f"{atom_label}_{orb_type}" if orb_type else f"{atom_label}_orb"
            )

    return base_labels


# ──────────────────────────────────────────────────────────────────
# AVAS (Automated Valence Active Space) section
# Based on Sayfutyarova et al., JCTC 2017.
# Uses PySCF's built-in AVAS module with manual fallback.
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# AVAS-based CAS construction (renamed from build_avas_active_space)
# ──────────────────────────────────────────────────────────────────


def _construct_avas(
    mol,
    mf,
    cluster_info: _ClusterInfo,
    config: _AVASConfig,
) -> tuple[_CAS, list[dict]]:
    """AVAS-based CAS construction.

    Args:
        mol: PySCF Mole object.
        mf: Converged SCF object.
        cluster_info: ClusterInfo describing the cluster.
        config: AVASConfig with AVAS parameters.

    Returns:
        Tuple of (CAS, expected_types).
    """
    # Extract mo_coeff from mf
    mo_coeff = mf.mo_coeff  # PySCF: SCF 轨道系数矩阵（UHF 为 [alpha, beta] 列表）
    if isinstance(mo_coeff, (list, tuple)):
        # Unrestricted: use alpha coefficients
        mo_coeff = mo_coeff[0]

    # 1. Determine target valence orbitals.
    if config.avas_valence_orbitals:
        valence_orbitals = config.avas_valence_orbitals
    else:
        valence_orbitals = _build_avas_valence_from_knowledge_base(cluster_info)

    # 2. Run AVAS selection.
    selected_indices, projection_weights = _avas_select(
        mol,
        mf,
        mo_coeff,
        valence_orbitals,
        threshold=config.avas_threshold,
    )

    n_orbitals = len(selected_indices)

    # 3. Estimate electron count from MO occupation analysis.
    n_electrons = _estimate_active_electrons(
        mol, mo_coeff, selected_indices, cluster_info
    )

    # 4. Build orbital groups from the valence_orbitals specification.
    orbital_groups = _build_orbital_groups(valence_orbitals, cluster_info)

    # 5. Build expected_types metadata.
    expected_types = [
        {"element": elem, "ao_types": list(ao_list)}
        for elem, ao_list in valence_orbitals.items()
    ]

    # 6. Construct CAS.
    description = (
        f"({n_electrons}e, {n_orbitals}o) AVAS-selected "
        f"[threshold={config.avas_threshold}]"
    )

    active_space = _CAS(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        orbital_groups=orbital_groups,
        level=_ActiveSpaceLevel.STANDARD,
        description=description,
        stage="computed",
        cpt_cas_type="avas",
        mo_coeff_full=None,
        occupations_full=None,
        orbital_labels_full=[],
    )

    return active_space, expected_types


def _build_avas_valence_from_knowledge_base(
    cluster_info: _ClusterInfo,
) -> dict[str, list[str]]:
    """Automatically infer AVAS target valence orbitals from the knowledge base.

    For each metal the active d orbitals are taken from the database, and the
    valence s orbital is added.  For each bridging atom the bridging p
    orbitals are taken from the ligand database.

    Args:
        cluster_info: A ``ClusterInfo`` object describing the cluster.

    Returns:
        dict mapping element symbol to list of AO type labels, e.g.
        ``{"Fe": ["3d", "4s"], "S": ["3p"], "Mo": ["4d"]}``.
    """
    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()
    result: dict[str, list[str]] = {}

    # Metals: use active_orbitals from the knowledge base + valence s.
    for metal in cluster_info.metals:
        elem = metal.element
        if elem in result:
            continue  # already processed this element type
        orbitals = []
        if elem in metals_db:
            active = metals_db[elem].get("active_orbitals", [])
            orbitals.extend(active)
            valence_s = _valence_s_for_element(elem)
            if valence_s and valence_s not in orbitals:
                orbitals.append(valence_s)
        if not orbitals:
            # Fallback: try to infer from the row field
            row = metals_db.get(elem, {}).get("row", "")
            if row:
                orbitals.append(row)  # e.g. "3d"
        if orbitals:
            result[elem] = orbitals

    # Bridging atoms: use bridging_orbitals from the ligand database.
    for bridge in cluster_info.bridging_atoms:
        elem = bridge.element
        if elem in result:
            continue
        if elem in ligands_db:
            bridging = ligands_db[elem].get(
                "bridging_orbitals", ligands_db[elem].get("active_orbitals", [])
            )
            if bridging:
                result[elem] = list(bridging)

    return result


# ──────────────────────────────────────────────────────────────────
# AVAS selection function
# ──────────────────────────────────────────────────────────────────
def _avas_select(
    mol,
    mf,
    mo_coeff: np.ndarray,
    valence_orbitals: dict[str, list[str]],
    threshold: float = 0.4,
) -> tuple[list[int], np.ndarray]:
    """Select active orbitals via AVAS projection.

    Projects MO coefficients onto a target atomic-orbital subspace defined by
    ``valence_orbitals`` and keeps MOs whose projection weight exceeds
    *threshold*.

    Args:
        mol: PySCF ``gto.Mole`` object.
        mf: Converged PySCF SCF object (required by avas.kernel).
        mo_coeff: MO coefficient matrix from SCF, shape ``(nao, nmo)``.
        valence_orbitals: Mapping of element symbol to list of AO type labels,
            e.g. ``{"Fe": ["3d", "4s"], "S": ["3p"], "Mo": ["4d"]}``.
        threshold: Minimum projection weight to include an orbital (default 0.4).

    Returns:
        selected_indices: List of MO indices selected for the active space.
        projection_weights: numpy array of projection weights for all MOs.
    """
    # Build AO label list from valence_orbitals dict.
    # Each entry becomes a string like "Fe 3d", "S 3p", etc.
    ao_labels = []
    for element, ao_types in valence_orbitals.items():
        for ao_type in ao_types:
            ao_labels.append(f"{element} {ao_type}")

    # Attempt to use PySCF's built-in AVAS implementation.
    try:
        from pyscf.mcscf import (
            avas as _pyscf_avas,  # PySCF: AVAS (Automated Valence Active Space) 模块
        )

        # PySCF: avas.kernel 第一个参数为 mf (SCF 对象)，不是 mol
        # avas.kernel returns (ncore, ncas, nelecas, mo_coeff_new)
        # We only need the orbital information.
        ncore, ncas, nelecas, mo_coeff_avas = _pyscf_avas.kernel(
            mf,
            mo_coeff,
            ao_labels=ao_labels,
            threshold=threshold,
        )
        # In PySCF's AVAS the active orbitals are mo_coeff_avas[:, ncore:ncore+ncas].
        # We map back to indices in the original mo_coeff.
        # For a simpler interface we compute projection weights ourselves so
        # the return value is always consistent.
        selected, weights = _manual_avas_projection(mol, mo_coeff, ao_labels, threshold)
        return selected, weights

    except (ImportError, AttributeError, Exception) as exc:
        warnings.warn(
            f"PySCF AVAS not available or failed ({exc}); "
            f"using manual projection fallback.",
            stacklevel=2,
        )
        selected, weights = _manual_avas_projection(mol, mo_coeff, ao_labels, threshold)
        return selected, weights


def _manual_avas_projection(
    mol,
    mo_coeff: np.ndarray,
    ao_labels: list[str],
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    """Manual AVAS projection when PySCF's avas module is unavailable.

    Algorithm:
      1. Build the projection operator P (diagonal mask) from target AOs.
      2. Compute overlap matrix S.
      3. For each MO i, compute weight:
            w_i = c_i^T P S P c_i  /  (c_i^T S c_i)
      4. Select MOs with w_i > threshold.
    """
    nao, nmo = mo_coeff.shape

    # 1. Build projection mask: 1 for target AOs, 0 otherwise.
    target_indices = set()
    for label in ao_labels:
        # mol.search_ao_label returns AO indices matching the label pattern.
        indices = mol.search_ao_label(label)  # PySCF: 搜索匹配标签的 AO 索引
        target_indices.update(
            indices.tolist() if hasattr(indices, "tolist") else indices
        )

    P = np.zeros(nao)
    for idx in target_indices:
        if 0 <= idx < nao:
            P[idx] = 1.0

    # 2. Overlap matrix
    S = mol.intor("int1e_ovlp")  # PySCF: 获取 AO 重叠积分（非对称化版本）

    # 3. Compute projection weights
    weights = np.zeros(nmo)
    for i in range(nmo):
        c_i = mo_coeff[:, i]
        denom = c_i @ S @ c_i
        if abs(denom) < 1e-14:
            weights[i] = 0.0
            continue
        PS = P * S  # broadcast: (nao,) * (nao, nao) -> only target rows kept
        PSP = PS[:, :] * P[np.newaxis, :]  # mask columns too
        weights[i] = (c_i @ PSP @ c_i) / denom

    # 4. Select MOs
    selected = [int(i) for i in range(nmo) if weights[i] > threshold]

    return selected, weights


def _estimate_active_electrons(
    mol,
    mo_coeff: np.ndarray,
    selected_indices: list[int],
    cluster_info: _ClusterInfo,
) -> int:
    """Estimate the number of active electrons in the selected orbitals.

    Uses occupation analysis: for a restricted calculation each spatial
    orbital carries 0 or 2 electrons.  For an initial estimate we assume
    all occupied MOs that fall into the active window contribute 2 electrons.
    """
    # Determine the number of occupied orbitals from the SCF density.
    # We use the cluster charge + nuclear charges to find the electron count,
    # then divide by 2 for restricted orbitals.
    nao, nmo = mo_coeff.shape  # PySCF: nao=AO基函数数, nmo=MO轨道数
    n_electrons_total = int(mol.nelectron)  # PySCF: 分子总电子数
    n_electrons_charge = cluster_info.total_charge
    n_electrons_scf = n_electrons_total - n_electrons_charge
    n_occ = n_electrons_scf // 2  # restricted: doubly occupied

    # Count how many selected orbitals are occupied.
    n_active_electrons = 0
    for idx in selected_indices:
        if idx < n_occ:
            n_active_electrons += 2  # doubly occupied in restricted SCF
        # Virtual orbitals contribute 0 electrons.

    return n_active_electrons


def _build_orbital_groups(
    valence_orbitals: dict[str, list[str]],
    cluster_info: _ClusterInfo,
) -> list[_OrbitalGroup]:
    """Build OrbitalGroup list from the valence_orbitals specification.

    Each unique element in valence_orbitals maps to one OrbitalGroup per
    site of that element in the cluster.
    """
    metals_db = _get_metals_db()
    ligands_db = _get_ligands_db()
    groups: list[_OrbitalGroup] = []

    # Count orbitals per AO type (s->1, p->3, d->5).
    _ao_capacity = {"s": 1, "p": 3, "d": 5, "f": 7}

    def _n_orbs_for_type(ao_type: str) -> int:
        # ao_type is like "3d", "4s", "3p".
        # Extract the angular momentum character (last character).
        char = ao_type[-1].lower() if ao_type else "d"
        return _ao_capacity.get(char, 5)

    # Metal groups.
    for site_idx, metal in enumerate(cluster_info.metals):
        elem = metal.element
        if elem not in valence_orbitals:
            continue
        ao_types = valence_orbitals[elem]
        n_orb = sum(_n_orbs_for_type(ao) for ao in ao_types)
        # Estimate electrons from the knowledge base.
        if elem in metals_db:
            # Use the most common oxidation state's d count as a rough estimate.
            ox_states = metals_db[elem].get("common_oxidation_states", [2])
            default_ox = ox_states[0] if ox_states else 2
            key = f"{elem}{abs(default_ox)}+" if default_ox > 0 else f"{elem}0"
            hs = metals_db[elem].get("high_spin_states", {})
            n_elec = hs.get(key, {}).get("d_count", 0)
            # Add s electrons if "4s"/"5s"/"6s" is in the target.
            for ao in ao_types:
                if ao.endswith("s"):
                    n_elec += 1  # rough: one s electron
        else:
            n_elec = 0

        groups.append(
            _OrbitalGroup(
                atom_label=_resolve_metal_site_label(cluster_info, site_idx),
                orbital_type="+".join(ao_types),
                n_orbitals=n_orb,
                n_electrons=n_elec,
            )
        )

    # Bridging atom groups.
    for bridge in cluster_info.bridging_atoms:
        elem = bridge.element
        if elem not in valence_orbitals:
            continue
        ao_types = valence_orbitals[elem]
        n_orb = sum(_n_orbs_for_type(ao) for ao in ao_types)

        # Estimate electrons from the ligand database.
        if elem in ligands_db:
            # Common estimate: 6 electrons for a filled p shell (S2-, O2-, etc.)
            n_elec = ligands_db[elem].get(
                f"electrons_as_{elem}2minus",
                ligands_db[elem].get("electrons_as_donor", 6),
            )
        else:
            n_elec = 6  # conservative default

        groups.append(
            _OrbitalGroup(
                atom_label=_resolve_explicit_label(
                    getattr(bridge, "label", ""),
                    f"{elem}{bridge.index}",
                    cluster_info=cluster_info,
                    context=f"bridging atom {bridge.index}",
                ),
                orbital_type="+".join(ao_types),
                n_orbitals=n_orb,
                n_electrons=n_elec,
            )
        )

    return groups
