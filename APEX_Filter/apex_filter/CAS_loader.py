"""CAS_loader -- Load CAS, FCIDUMP, and cluster data from APEX_CAS output.

Provides the entry point for APEX_Filter Step 1: loading all necessary
data produced by APEX_CAS (active space, integrals, cluster info).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import yaml
from shared.apex_cas_provenance import (
    build_effective_settings_from_apex_cas as _shared_build_effective_settings_from_apex_cas,
    load_apex_cas_provenance as _shared_load_apex_cas_provenance,
)
from shared.artifact_paths import (
    auto_detect_fcidump as _auto_detect_fcidump,
    candidate_structure_paths as _candidate_structure_paths,
    load_fcidump_summary as _load_fcidump_summary,
    load_json_if_exists as _load_json_if_exists,
    resolve_cluster_info_path as _resolve_cluster_info_path,
    resolve_fcidump_path as _resolve_fcidump_path,
    resolve_structure_path as _resolve_structure_path,
)

from .models import CAS, ClusterInfo, ComputationSettings, MetalCenter

if TYPE_CHECKING:
    from pyscf import gto, scf

logger = logging.getLogger(__name__)


@dataclass
class FCIDUMPData:
    """Parsed FCIDUMP integral file."""
    h1e: np.ndarray       # (norb, norb) one-electron integrals
    h2e: np.ndarray       # two-electron integrals (4-index or packed)
    ecore: float          # core energy
    norb: int             # number of active spatial orbitals
    nelec: int            # number of active electrons
    ms2: int              # 2 * total spin Sz


@dataclass
class FilterInputs:
    """Everything needed to start the filter pipeline (Steps 1-5)."""
    cas: CAS
    mol: "gto.Mole"
    mf: "scf.HF"
    cluster_info: ClusterInfo
    fcidump_data: FCIDUMPData
    settings: ComputationSettings
    case_dir: str
    fcidump_path: str
    config_raw: dict


def _load_apex_cas_provenance(case_dir: str) -> dict:
    """Thin local wrapper around the shared apex-cas provenance loader."""
    return _shared_load_apex_cas_provenance(case_dir)


def _extract_effective_settings(config_raw: dict, case_dir: str) -> tuple[ComputationSettings, dict]:
    """Build effective settings, preferring apex-cas stage sidecars over defaults."""
    return _shared_build_effective_settings_from_apex_cas(
        config_raw=config_raw,
        case_dir=case_dir,
        settings_cls=ComputationSettings,
        provenance_loader=_load_apex_cas_provenance,
    )


def _reconcile_cas_with_fcidump_selection(cas: CAS, fcidump_path: str, fcid: FCIDUMPData) -> CAS:
    """Treat the final FCIDUMP/selection as the authoritative active space.

    APEX_CAS may auto-build a compact CAS first and later expand the active
    space through manual edits to ``selection.txt`` before FCIDUMP export.
    APEX_Filter should therefore load the final FCIDUMP dimensions instead of
    assuming the original auto-selected CAS is still authoritative.
    """
    summary = _load_fcidump_summary(fcidump_path)
    if not summary:
        return cas

    selection_file = summary.get("selection_file")
    selected_indices = None
    selected_nelec = summary.get("n_electrons", fcid.nelec)
    selected_norb = summary.get("n_orbitals", fcid.norb)

    if selection_file and os.path.isfile(selection_file):
        from apex_cas.selection_io import load_active_selection

        selected_indices, selected_nelec = load_active_selection(selection_file)
        selected_norb = len(selected_indices)

    cas.n_electrons = int(selected_nelec)
    cas.n_orbitals = int(selected_norb)
    cas.n_qubits = 2 * cas.n_orbitals

    if selected_indices is not None:
        cas.active_indices = list(selected_indices)
        if cas.orbital_labels_full:
            cas.orbital_labels = [cas.orbital_labels_full[i] for i in selected_indices]
        if cas.occupations_full is not None:
            cas.occupations = np.asarray([cas.occupations_full[i] for i in selected_indices])
        if cas.mo_coeff_full is not None:
            cas.mo_coeff_alpha = cas.mo_coeff_full[:, selected_indices]
            cas.mo_coeff_beta = cas.mo_coeff_full[:, selected_indices]
        cas.selection_method = "selection_txt"

    return cas


def load_cas_state_from_apex_cas(case_dir: str) -> tuple:
    """Load (cas, mol, mf) from an APEX_CAS output directory.

    Wraps apex_cas.state_io.load_cas_state but bypasses
    the interactive chkfile selection prompt by auto-selecting
    the largest .chk file.

    Parameters
    ----------
    case_dir : str
        Path to APEX_CAS case directory containing outputs/.

    Returns
    -------
    tuple[CAS, Mole, SCF]
    """
    from apex_cas.state_io import load_cas_state as _load_cas_state

    # Temporarily patch input() to auto-select the first (largest) chkfile
    import builtins
    original_input = builtins.input

    def auto_input(prompt=""):
        # Auto-select default (empty string = index 0)
        return ""

    try:
        builtins.input = auto_input
        cas, mol, mf = _load_cas_state(case_dir)
    finally:
        builtins.input = original_input

    return cas, mol, mf


def load_fcidump(fcidump_path: str) -> FCIDUMPData:
    """Parse a standard FCIDUMP text file.

    Uses pyscf.tools.fcidump.read internally.
    Also attempts to load a sibling .ecore sidecar file.

    Parameters
    ----------
    fcidump_path : str
        Path to the FCIDUMP file.

    Returns
    -------
    FCIDUMPData
    """
    from pyscf.tools import fcidump as fcidump_mod

    data = fcidump_mod.read(fcidump_path)
    h1e = data["H1"]
    h2e = data["H2"]
    ecore = float(data["ECORE"])
    norb = data["NORB"]
    nelec = data["NELEC"]
    ms2 = data["MS2"]

    # Try to load ecore sidecar
    ecore_path = fcidump_path + ".ecore"
    if os.path.isfile(ecore_path):
        try:
            with open(ecore_path, "r") as f:
                ecore_from_sidecar = float(f.read().strip())
            if abs(ecore) < 1e-10 and abs(ecore_from_sidecar) > 1e-10:
                logger.info("Using ecore from sidecar: %.6f", ecore_from_sidecar)
                ecore = ecore_from_sidecar
        except (ValueError, IOError):
            pass

    return FCIDUMPData(
        h1e=h1e, h2e=h2e, ecore=ecore,
        norb=norb, nelec=nelec, ms2=ms2,
    )


def load_cluster_info_from_mol(
    mol,
    charge: int = 0,
    spin: float = 0.0,
    symmetry_group: str = "C1",
    reduction_symmetry: str = None,
    family_scheme: str = "",
    benchmark_profile: str = "",
    config_reduction_mode: str = "none",
) -> ClusterInfo:
    """Reconstruct ClusterInfo from a PySCF Mole object.

    Detects metal centers from atom elements. For rich cluster info
    (bridging atoms, terminal ligands), use parse_structure from APEX_CAS.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Restored mol object from APEX_CAS checkpoint.
    charge : int
        Total charge.
    spin : float
        Target total spin S.
    symmetry_group : str
        Approximate point group.
    reduction_symmetry : str
        Downstream reduction symmetry consumed by APEX_Filter.

    Returns
    -------
    ClusterInfo
    """
    # Metal elements (3d, 4d, 5d transition metals)
    METAL_ELEMENTS = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }

    metals = []
    all_elements = []
    all_positions = []

    for i, (symbol, coord) in enumerate(mol._atom):
        elem = symbol.capitalize()
        pos = np.array(coord)
        all_elements.append(elem)
        all_positions.append(pos)

        if elem in METAL_ELEMENTS:
            metals.append(MetalCenter(
                element=elem,
                index=i,
                position=pos,
                label=f"{elem}{len(metals) + 1}",
            ))

    positions = np.array(all_positions) if all_positions else None

    return ClusterInfo(
        metals=metals,
        bridging_atoms=[],
        terminal_ligands=[],
        all_elements=all_elements,
        all_positions=positions,
        formula=mol.formula if hasattr(mol, "formula") else "",
        total_charge=charge,
        target_spin=spin,
        symmetry_group=symmetry_group,
        reduction_symmetry=reduction_symmetry or symmetry_group,
        family_scheme=family_scheme or "",
        benchmark_profile=benchmark_profile or "",
        config_reduction_mode=config_reduction_mode or "none",
        annotation_source="mol_fallback",
    )


def load_cluster_info(
    *,
    case_dir: str,
    config_dir: str | None = None,
    config_raw: dict,
    mol,
    charge: int,
    spin: float,
    symmetry_group: str,
    reduction_symmetry: str = None,
    family_scheme: str = "",
    benchmark_profile: str = "",
    config_reduction_mode: str = "none",
):
    """Load ClusterInfo, preferring structure parsing over metals-only fallback."""
    structure_path = _resolve_structure_path(config_raw, case_dir)
    cluster_info_path = _resolve_cluster_info_path(config_raw, case_dir, config_dir)
    explicit_cluster_info_requested = cluster_info_path is not None

    if structure_path is not None:
        try:
            from apex_cas import parse_structure

            cluster_info = parse_structure(
                structure_path,
                charge=charge,
                target_spin=spin,
                cluster_info_path=cluster_info_path,
                family_scheme=family_scheme,
                benchmark_profile=benchmark_profile,
                config_reduction_mode=config_reduction_mode,
            )
            if symmetry_group:
                cluster_info.symmetry_group = symmetry_group
            if reduction_symmetry:
                cluster_info.reduction_symmetry = reduction_symmetry
            if family_scheme:
                cluster_info.family_scheme = family_scheme
            if benchmark_profile:
                cluster_info.benchmark_profile = benchmark_profile
            if config_reduction_mode:
                cluster_info.config_reduction_mode = config_reduction_mode
            return cluster_info
        except Exception as exc:
            if explicit_cluster_info_requested:
                raise RuntimeError(
                    "Failed to reconstruct ClusterInfo from the explicit "
                    f"cluster_info_path {cluster_info_path!r}: {exc}"
                ) from exc
            logger.warning(
                "Failed to reconstruct ClusterInfo from structure file %s; "
                "falling back to mol-based reconstruction: %s",
                structure_path,
                exc,
            )

    return load_cluster_info_from_mol(
        mol, charge, spin, symmetry_group, reduction_symmetry,
        family_scheme, benchmark_profile, config_reduction_mode,
    )


def load_filter_inputs(config_yaml_path: str) -> FilterInputs:
    """Top-level loader: read YAML config, load all APEX_CAS outputs.

    Parameters
    ----------
    config_yaml_path : str
        Path to the filter settings YAML file.

    Returns
    -------
    FilterInputs
    """
    # Load YAML
    with open(config_yaml_path, "r") as f:
        config_raw = yaml.safe_load(f) or {}
    config_dir = os.path.dirname(os.path.abspath(config_yaml_path))

    # Required field
    case_dir = config_raw.get("apex_cas_case_dir")
    if not case_dir:
        raise ValueError("YAML config must specify 'apex_cas_case_dir'")
    case_dir = os.path.abspath(case_dir)

    # Cluster info params
    charge = config_raw.get("charge", 0)
    spin = config_raw.get("spin", 0.0)
    symmetry_group = config_raw.get("symmetry_group", "C1")
    reduction_symmetry = config_raw.get("reduction_symmetry", symmetry_group)
    family_scheme = config_raw.get("family_scheme", "")
    benchmark_profile = config_raw.get("benchmark_profile", "")
    config_reduction_mode = config_raw.get("config_reduction_mode", "none")

    # Load CAS state from APEX_CAS
    logger.info("Loading CAS state from %s ...", case_dir)
    cas, mol, mf = load_cas_state_from_apex_cas(case_dir)
    logger.info("  CAS: (%de, %do)", cas.n_electrons, cas.n_orbitals)

    # Reconstruct ClusterInfo, preferring the original structure file when available.
    cluster_info = load_cluster_info(
        case_dir=case_dir,
        config_dir=config_dir,
        config_raw=config_raw,
        mol=mol,
        charge=charge,
        spin=spin,
        symmetry_group=symmetry_group,
        reduction_symmetry=reduction_symmetry,
        family_scheme=family_scheme,
        benchmark_profile=benchmark_profile,
        config_reduction_mode=config_reduction_mode,
    )

    # Load FCIDUMP
    fcidump_path = _resolve_fcidump_path(config_raw, case_dir, config_dir)
    if fcidump_path is None:
        raise FileNotFoundError(
            f"No FCIDUMP found in {case_dir}/outputs/fcidump/. "
            "Specify 'fcidump_path' in the YAML config."
        )
    logger.info("Loading FCIDUMP: %s", fcidump_path)
    fcidump_data = load_fcidump(fcidump_path)
    cas = _reconcile_cas_with_fcidump_selection(cas, fcidump_path, fcidump_data)

    settings, apex_cas_provenance = _extract_effective_settings(config_raw, case_dir)

    return FilterInputs(
        cas=cas,
        mol=mol,
        mf=mf,
        cluster_info=cluster_info,
        fcidump_data=fcidump_data,
        settings=settings,
        case_dir=case_dir,
        fcidump_path=fcidump_path,
        config_raw={**config_raw, "_apex_cas_provenance": apex_cas_provenance},
    )
