"""Step 1 loading helpers for APEX_Filter.

This module provides internal Step 1 orchestration helpers for reconstructing
the authoritative CAS, FCIDUMP, and finalized cluster-info view produced by
``APEX_CAS``. The one remaining cross-package dependency on
``apex_cas.state_io.load_cas_state(...)`` is intentional: Step 1 consumes the
authoritative APEX_CAS state-restoration boundary rather than reimplementing
it in shared code.
"""

import logging
import os
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING

import numpy as np
import yaml

# Intentional Step 1 consumer boundary: APEX_Filter reconstructs canonical
# APEX_CAS state through the staged readback entry point instead of copying
# restoration logic into shared.
from apex_cas.state_io import load_cas_state as _load_cas_state
from shared.apex_cas_provenance import (
    build_effective_settings_from_apex_cas as _build_effective_settings_from_apex_cas,
    load_apex_cas_provenance as _load_apex_cas_provenance,
)
from shared.artifact_paths import (
    load_fcidump_summary as _load_fcidump_summary,
    resolve_cluster_info_path as _resolve_cluster_info_path,
    resolve_fcidump_path as _resolve_fcidump_path,
    resolve_structure_path as _resolve_structure_path,
)
from shared.cluster_info_labels import require_authoritative_cluster_info as _require_authoritative_cluster_info
from shared.fcidump_io import FCIDUMPData as _FCIDUMPData, load_fcidump as _load_fcidump
from shared.selection_io import load_active_selection as _load_active_selection
from shared.structure_parser import parse_structure as _parse_structure

from shared.models import CAS as _CAS, ClusterInfo as _ClusterInfo, ComputationSettings as _ComputationSettings

if TYPE_CHECKING:
    from pyscf import gto, scf

logger = logging.getLogger(__name__)


@_dataclass
class _FilterInputs:
    """Everything needed to bootstrap the staged filter workflow from Step 1 onward."""
    cas: _CAS
    mol: "gto.Mole"
    mf: "scf.HF"
    cluster_info: _ClusterInfo
    fcidump_data: _FCIDUMPData
    settings: _ComputationSettings
    case_dir: str
    fcidump_path: str
    config_raw: dict


def _reconcile_cas_with_fcidump_selection(cas: _CAS, fcidump_path: str, fcid: _FCIDUMPData) -> _CAS:
    """Internal helper that treats final FCIDUMP/selection as authoritative.

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
        selected_indices, selected_nelec = _load_active_selection(selection_file)
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


def _load_cluster_info(
    *,
    case_dir: str,
    config_dir: str | None = None,
    config_raw: dict,
    charge: int,
    spin: float,
    symmetry_group: str,
    reduction_symmetry: str = None,
    family_scheme: str = "",
    benchmark_profile: str = "",
    config_reduction_mode: str = "none",
):
    """Rebuild authoritative ``ClusterInfo`` from finalized authority files."""
    structure_path = _resolve_structure_path(config_raw, case_dir)
    cluster_info_path = _resolve_cluster_info_path(config_raw, case_dir, config_dir)

    if structure_path is None:
        raise FileNotFoundError(
            "APEX_Filter requires the original structure file to reload the "
            "authoritative ClusterInfo. Set 'structure_path' explicitly or "
            "place a unique structure file under case_dir/inputs/."
        )
    if cluster_info_path is None:
        raise FileNotFoundError(
            "APEX_Filter requires a finalized cluster_info.yaml authority file. "
            "Run 'apex-cas prepare ... --finalize' first, or set "
            "'cluster_info_path' explicitly in the filter config."
        )

    try:
        cluster_info = _parse_structure(
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
        _require_authoritative_cluster_info(
            cluster_info,
            context="APEX_Filter Step 1 load",
        )
        return cluster_info
    except Exception as exc:
        raise RuntimeError(
            "Failed to reconstruct ClusterInfo from the finalized structure "
            f"and cluster_info authority files ({structure_path!r}, "
            f"{cluster_info_path!r}): {exc}"
        ) from exc


def _load_filter_inputs(config_yaml_path: str) -> _FilterInputs:
    """Top-level Step 1 loader for canonical filter inputs.

    Parameters
    ----------
    config_yaml_path : str
        Path to the filter settings YAML file.

    Returns
    -------
    _FilterInputs
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
    cas, mol, mf = _load_cas_state(case_dir)
    logger.info("  CAS: (%de, %do)", cas.n_electrons, cas.n_orbitals)

    # Reconstruct ClusterInfo, preferring the original structure file when available.
    cluster_info = _load_cluster_info(
        case_dir=case_dir,
        config_dir=config_dir,
        config_raw=config_raw,
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
    fcidump_data = _load_fcidump(fcidump_path)
    cas = _reconcile_cas_with_fcidump_selection(cas, fcidump_path, fcidump_data)

    settings, apex_cas_provenance = _build_effective_settings_from_apex_cas(
        config_raw=config_raw,
        case_dir=case_dir,
        settings_cls=_ComputationSettings,
        provenance_loader=_load_apex_cas_provenance,
    )

    return _FilterInputs(
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
