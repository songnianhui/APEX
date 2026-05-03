"""Standalone post-SCF observable analysis for APEX_Filter states.

This module is intentionally decoupled from the main workflow so that
additional observables can be added without entangling the core step logic.
The first target is the Fe-site local spin moment used in the Chan benchmarks.
Public analysis entry points remain focused on
``analyze_active_space_spin_observables(...)`` and
``analyze_step3_uhf_observables(...)``; benchmark comparison helpers are
internal implementation details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path as _Path

import h5py
import numpy as np
from pyscf import scf

from shared.ao_basis import get_d_ao_indices as _get_d_ao_indices
from shared.cluster_info_labels import require_authoritative_cluster_info as _require_authoritative_cluster_info
from shared.comparison import compare_two_sz_with_benchmark as _compare_two_sz_with_benchmark
from shared.molecule_builder import build_mol_with_basis as _build_mol_with_basis
from shared.models import ComputationSettings as _ComputationSettings
from shared.setting_utils import (
    apply_overrides as _apply_overrides,
    load_cas_settings_file as _load_cas_settings_file,
    settings_from_preset as _settings_from_preset,
)
from shared.spin_metrics import compute_two_s_from_s2 as _compute_two_s_from_s2
from shared.structure_parser import parse_structure as _parse_structure


def _load_settings(cas_settings_path: str) -> tuple[_ComputationSettings, dict]:
    raw = _load_cas_settings_file(cas_settings_path)
    preset_name = raw.get("preset", "default")
    settings = _settings_from_preset(preset_name)
    allowed = set(_ComputationSettings.__dataclass_fields__.keys())
    overrides = {k: v for k, v in raw.items() if k in allowed}
    if (
        "basis_set_default" in raw
        and "basis_set_per_element" not in raw
        and "basis_set_file" not in raw
    ):
        overrides["basis_set_per_element"] = {}
    if overrides:
        settings = _apply_overrides(settings, **overrides)
    return settings, raw


def _load_step3_state(step3_h5_path: str) -> dict:
    with h5py.File(step3_h5_path, "r") as h5:
        metadata = h5["metadata"].attrs
        payload = {
            "label": str(metadata.get("label", "")),
            "family": str(metadata.get("family", "")),
            "energy": float(metadata["energy"]),
            "s2": float(metadata["spin_sq"]),
            "final_state_signature": str(metadata.get("final_state_signature", "")),
            "dm_a": h5["density_matrices/dm_a"][:],
            "dm_b": h5["density_matrices/dm_b"][:],
            "active_indices": h5["active_space_mapping/active_indices"][:].astype(int),
        }
    return payload


def _load_active_mo_coefficients(cas_data_h5_path: str, active_indices: np.ndarray) -> np.ndarray:
    with h5py.File(cas_data_h5_path, "r") as h5:
        mo_coeff_full = h5["mo_coeff_full"][:]
    return mo_coeff_full[:, active_indices]


def _build_active_ao_density_matrices(
    dm_a: np.ndarray,
    dm_b: np.ndarray,
    active_mo_coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project active-space alpha/beta density matrices back to the AO basis."""
    coeff = np.asarray(active_mo_coeff, dtype=float)
    dm_a_ao = coeff @ np.asarray(dm_a, dtype=float) @ coeff.T
    dm_b_ao = coeff @ np.asarray(dm_b, dtype=float) @ coeff.T
    return dm_a_ao, dm_b_ao


def _compute_metal_d_two_sz(mol, cluster_info, ao_spin_density: np.ndarray) -> dict[str, float]:
    """Compute 2Sz on each metal by projecting onto the metal-d AO subspace.

    In a non-orthogonal AO basis, the local spin moment is evaluated as the
    population-like trace on the d-block: Tr[D_spin * S]_(metal d block).
    """
    overlap = mol.intor_symmetric("int1e_ovlp")
    result: dict[str, float] = {}
    for metal in cluster_info.metals:
        d_indices = _get_d_ao_indices(mol, metal.index)
        if not d_indices:
            result[metal.label] = 0.0
            continue
        sub_density = ao_spin_density[np.ix_(d_indices, d_indices)]
        sub_overlap = overlap[np.ix_(d_indices, d_indices)]
        result[metal.label] = float(np.sum(sub_density * sub_overlap))
    return result


def _compute_meta_lowdin_atomic_two_sz(
    mol,
    cluster_info,
    dm_a_ao: np.ndarray,
    dm_b_ao: np.ndarray,
) -> dict[str, float]:
    """Compute Fe-site local spin from meta-Lowdin atomic populations.

    This uses the standard PySCF meta-Lowdin orthogonalized AO population
    analysis on the alpha and beta density matrices separately, then sums the
    per-AO spin populations over each Fe atom.
    """
    overlap = mol.intor_symmetric("int1e_ovlp")
    old_verbose = getattr(mol, "verbose", 0)
    mol.verbose = 0
    try:
        pop_a, _ = scf.hf.mulliken_meta(mol, dm_a_ao, s=overlap, verbose=0)
        pop_b, _ = scf.hf.mulliken_meta(mol, dm_b_ao, s=overlap, verbose=0)
    finally:
        mol.verbose = old_verbose

    aoslices = mol.aoslice_by_atom()
    result: dict[str, float] = {}
    for metal in cluster_info.metals:
        _, _, ao_s, ao_e = aoslices[metal.index]
        result[metal.label] = float(np.sum(pop_a[ao_s:ao_e] - pop_b[ao_s:ao_e]))
    return result


def _compute_meta_lowdin_d_two_sz(
    mol,
    cluster_info,
    dm_a_ao: np.ndarray,
    dm_b_ao: np.ndarray,
) -> dict[str, float]:
    """Compute Fe-site local spin from meta-Lowdin populations on Fe d AOs only."""
    overlap = mol.intor_symmetric("int1e_ovlp")
    old_verbose = getattr(mol, "verbose", 0)
    mol.verbose = 0
    try:
        pop_a, _ = scf.hf.mulliken_meta(mol, dm_a_ao, s=overlap, verbose=0)
        pop_b, _ = scf.hf.mulliken_meta(mol, dm_b_ao, s=overlap, verbose=0)
    finally:
        mol.verbose = old_verbose

    result: dict[str, float] = {}
    for metal in cluster_info.metals:
        d_indices = _get_d_ao_indices(mol, metal.index)
        if not d_indices:
            result[metal.label] = 0.0
            continue
        result[metal.label] = float(np.sum(pop_a[d_indices] - pop_b[d_indices]))
    return result


def analyze_active_space_spin_observables(
    *,
    dm_a_active: np.ndarray,
    dm_b_active: np.ndarray,
    active_indices: np.ndarray,
    xyz_path: str,
    cluster_info_path: str,
    cas_settings_path: str,
    cas_data_h5_path: str,
    label: str = "",
    family: str = "",
    energy_hartree: float | None = None,
    s2: float | None = None,
    final_state_signature: str = "",
    chan_benchmark_json: str | None = None,
    theory: str = "UHF",
) -> dict:
    """Analyze local-spin observables for any active-space state."""
    settings, settings_raw = _load_settings(cas_settings_path)
    cluster_info = _parse_structure(
        xyz_path,
        charge=int(settings_raw["charge"]),
        target_spin=float(settings_raw["spin"]),
        cluster_info_path=cluster_info_path,
    )
    _require_authoritative_cluster_info(
        cluster_info,
        context="APEX_Filter post-SCF observables",
    )
    mol = _build_mol_with_basis(cluster_info, settings)

    active_coeff = _load_active_mo_coefficients(cas_data_h5_path, np.asarray(active_indices, dtype=int))
    ao_dm_a, ao_dm_b = _build_active_ao_density_matrices(dm_a_active, dm_b_active, active_coeff)
    ao_spin_density = ao_dm_a - ao_dm_b

    two_sz_methods = {
        "ao_projected_fe_d": {
            "two_sz_by_metal_label": _compute_metal_d_two_sz(mol, cluster_info, ao_spin_density),
            "definition": (
                "Projected local spin moment on the Fe d-AO subspace, "
                "computed as Tr[(D_alpha-D_beta) S] over the metal-d AO block "
                "after back-projecting the active-space spin density to the AO basis."
            ),
        },
        "meta_lowdin_atomic": {
            "two_sz_by_metal_label": _compute_meta_lowdin_atomic_two_sz(mol, cluster_info, ao_dm_a, ao_dm_b),
            "definition": (
                "Meta-Lowdin orthogonalized AO population analysis applied separately to "
                "alpha and beta AO densities; Fe-site 2Sz obtained by summing (pop_alpha-pop_beta) "
                "over all orthogonalized AOs belonging to each Fe atom."
            ),
        },
        "meta_lowdin_fe_d": {
            "two_sz_by_metal_label": _compute_meta_lowdin_d_two_sz(mol, cluster_info, ao_dm_a, ao_dm_b),
            "definition": (
                "Meta-Lowdin orthogonalized AO population analysis applied separately to "
                "alpha and beta AO densities; Fe-site 2Sz obtained by summing (pop_alpha-pop_beta) "
                "only over Fe d-like orthogonalized AOs."
            ),
        },
    }

    result = {
        "label": label,
        "family": family,
        "energy_hartree": energy_hartree,
        "s2": s2,
        "two_s": _compute_two_s_from_s2(s2) if s2 is not None else None,
        "final_state_signature": final_state_signature,
        "two_sz_by_metal_label": two_sz_methods["ao_projected_fe_d"]["two_sz_by_metal_label"],
        "definition": {
            "primary_two_sz_method": "ao_projected_fe_d",
        },
        "two_sz_methods": two_sz_methods,
        "provenance": {
            "cas_data_h5_path": str(_Path(cas_data_h5_path).resolve()),
            "xyz_path": str(_Path(xyz_path).resolve()),
            "cluster_info_path": str(_Path(cluster_info_path).resolve()),
            "cas_settings_path": str(_Path(cas_settings_path).resolve()),
        },
    }
    if chan_benchmark_json:
        result["chan_benchmark_comparison"] = _compare_two_sz_with_benchmark(
            two_sz_methods["ao_projected_fe_d"]["two_sz_by_metal_label"],
            chan_benchmark_json,
            theory=theory,
        )
        for payload in two_sz_methods.values():
            payload["chan_benchmark_comparison"] = _compare_two_sz_with_benchmark(
                payload["two_sz_by_metal_label"],
                chan_benchmark_json,
                theory=theory,
            )
    return result


def analyze_step3_uhf_observables(
    *,
    step3_h5_path: str,
    cas_data_h5_path: str,
    xyz_path: str,
    cluster_info_path: str,
    cas_settings_path: str,
    chan_benchmark_json: str | None = None,
) -> dict:
    state = _load_step3_state(step3_h5_path)
    result = analyze_active_space_spin_observables(
        dm_a_active=state["dm_a"],
        dm_b_active=state["dm_b"],
        active_indices=state["active_indices"],
        xyz_path=xyz_path,
        cluster_info_path=cluster_info_path,
        cas_settings_path=cas_settings_path,
        cas_data_h5_path=cas_data_h5_path,
        label=state["label"],
        family=state["family"],
        energy_hartree=state["energy"],
        s2=state["s2"],
        final_state_signature=state["final_state_signature"],
        chan_benchmark_json=chan_benchmark_json,
        theory="UHF",
    )
    result["provenance"]["step3_h5_path"] = str(_Path(step3_h5_path).resolve())
    return result


def _default_output_path(step3_h5_path: str) -> str:
    p = _Path(step3_h5_path)
    return str(p.with_name(p.stem + "_post_scf_observables.json"))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze post-SCF observables from step3 UHF HDF5.")
    parser.add_argument("--step3-h5", required=True)
    parser.add_argument("--cas-data-h5", required=True)
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--cluster-info", required=True)
    parser.add_argument("--cas-settings", required=True)
    parser.add_argument("--chan-benchmark-json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    result = analyze_step3_uhf_observables(
        step3_h5_path=args.step3_h5,
        cas_data_h5_path=args.cas_data_h5,
        xyz_path=args.xyz,
        cluster_info_path=args.cluster_info,
        cas_settings_path=args.cas_settings,
        chan_benchmark_json=args.chan_benchmark_json,
    )
    output_path = args.output or _default_output_path(args.step3_h5)
    _Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
