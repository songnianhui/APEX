"""HDF5 helpers for large stateful APEX_Filter artifacts.

This module introduces structured HDF5 containers for step outputs that carry
enough state to support future derived calculations without rerunning SCF.
"""

from __future__ import annotations

import dataclasses
import json
import os

import h5py
import numpy as np


_H5_KWARGS = dict(compression="gzip", compression_opts=9, shuffle=True)


def _write_dataset(group, key: str, value):
    arr = np.asarray(value)
    if arr.shape == ():
        group.create_dataset(key, data=arr)
    else:
        group.create_dataset(key, data=arr, **_H5_KWARGS)


def _attr_set(group, key: str, value):
    if value is None:
        return
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (dict, list, tuple)):
        group.attrs[key] = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (str, bytes, np.str_, np.bytes_)):
        group.attrs[key] = str(value)
    else:
        group.attrs[key] = value


def _xyz_text(elements, positions) -> str:
    if not elements or positions is None or len(elements) != len(positions):
        return ""
    lines = [str(len(elements)), "Generated from cluster_info.all_elements/all_positions"]
    for elem, xyz in zip(elements, positions):
        x, y, z = [float(v) for v in xyz]
        lines.append(f"{elem} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines)


def save_uhf_state_h5(
    h5_path: str,
    payload: dict,
    *,
    label: str | None = None,
    family: str | None = None,
    settings=None,
    cluster_info=None,
    fcidump_data=None,
    cas=None,
):
    """Persist a step3 UHF result payload as structured HDF5."""
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["artifact_type"] = "apex_filter_step3_uhf_state"
        _attr_set(meta, "label", label)
        _attr_set(meta, "family", family)

        for key in ("energy", "converged", "spin_sq", "final_delta_e"):
            if key in payload:
                _attr_set(meta, key, payload[key])

        for key in ("final_state_signature", "final_d_basin_json", "final_site_spin_proxy_json"):
            if key in payload:
                _attr_set(meta, key, payload[key])

        if settings is not None:
            if dataclasses.is_dataclass(settings):
                settings_payload = dataclasses.asdict(settings)
            else:
                settings_payload = dict(settings)
            _attr_set(meta, "settings_json", settings_payload)

        orbitals = f.create_group("orbitals")
        for key in ("mo_coeff_a", "mo_coeff_b", "mo_occ_a", "mo_occ_b", "mo_energy_a", "mo_energy_b"):
            if key in payload:
                _write_dataset(orbitals, key, payload[key])

        density = f.create_group("density_matrices")
        for key in ("dm_a", "dm_b"):
            if key in payload:
                _write_dataset(density, key, payload[key])

        diagnostics = f.create_group("diagnostics")
        for key in (
            "bs_stabilize_energy_history",
            "bs_stabilize_delta_e_history",
            "bs_tight_energy_history",
            "bs_tight_delta_e_history",
            "newton_energy_history",
            "newton_delta_e_history",
        ):
            if key in payload:
                _write_dataset(diagnostics, key, payload[key])

        molecule = f.create_group("molecule")
        if cluster_info is not None:
            _attr_set(molecule, "charge", getattr(cluster_info, "total_charge", None))
            _attr_set(molecule, "target_spin", getattr(cluster_info, "target_spin", None))
            all_elements = getattr(cluster_info, "all_elements", None)
            all_positions = getattr(cluster_info, "all_positions", None)
            if all_elements:
                dt = h5py.string_dtype("utf-8")
                molecule.create_dataset("atom_symbols", data=np.array([str(x) for x in all_elements], dtype=dt))
            if all_positions is not None:
                _write_dataset(molecule, "atom_positions", np.asarray(all_positions, dtype=float))
            xyz_text = _xyz_text(all_elements, all_positions)
            if xyz_text:
                _attr_set(molecule, "serialized_xyz", xyz_text)
        if settings is not None:
            _attr_set(molecule, "basis_set_default", getattr(settings, "basis_set_default", None))
            _attr_set(molecule, "basis_set_per_element_json", getattr(settings, "basis_set_per_element", None))
            _attr_set(molecule, "scf_method", getattr(settings, "scf_method", None))
            _attr_set(molecule, "xc_functional", getattr(settings, "xc_functional", None))
        if fcidump_data is not None:
            _attr_set(molecule, "active_norb", getattr(fcidump_data, "norb", None))
            _attr_set(molecule, "active_nelec", getattr(fcidump_data, "nelec", None))
            _attr_set(molecule, "active_ms2", getattr(fcidump_data, "ms2", None))
            _attr_set(molecule, "ecore", getattr(fcidump_data, "ecore", None))
        if "mo_occ_a" in payload and "mo_occ_b" in payload:
            solver_ms2 = int(round(float(np.sum(payload["mo_occ_a"]) - np.sum(payload["mo_occ_b"]))))
            _attr_set(molecule, "solver_ms2", solver_ms2)
        _attr_set(
            molecule,
            "serialized_solver_mol",
            {
                "container_atom": [["H", [0.0, 0.0, 0.0]]],
                "container_basis": "sto-3g",
                "notes": "Fake PySCF Mole used by active-space UHF; actual Hamiltonian comes from FCIDUMP integrals.",
            },
        )

        if cas is not None:
            mapping = f.create_group("active_space_mapping")
            if getattr(cas, "active_indices", None) is not None:
                _write_dataset(mapping, "active_indices", np.asarray(cas.active_indices, dtype=int))
            if getattr(cas, "orbital_labels", None):
                dt = h5py.string_dtype("utf-8")
                mapping.create_dataset(
                    "orbital_labels",
                    data=np.array([str(x) for x in cas.orbital_labels], dtype=dt),
                )
            if getattr(cas, "orbital_labels_full", None):
                dt = h5py.string_dtype("utf-8")
                mapping.create_dataset(
                    "orbital_labels_full",
                    data=np.array([str(x) for x in cas.orbital_labels_full], dtype=dt),
                )


def load_uhf_state_h5(h5_path: str) -> dict:
    """Load a step3 UHF HDF5 state into a dict compatible with old NPZ keys."""
    payload = {}
    with h5py.File(h5_path, "r") as f:
        meta = f.get("metadata")
        if meta is not None:
            for key in ("energy", "converged", "spin_sq", "final_delta_e", "label", "family", "settings_json"):
                if key in meta.attrs:
                    payload[key] = meta.attrs[key]
            for key in ("final_state_signature", "final_d_basin_json", "final_site_spin_proxy_json"):
                if key in meta.attrs:
                    payload[key] = meta.attrs[key]

        for group_name, keys in (
            (
                "orbitals",
                ("mo_coeff_a", "mo_coeff_b", "mo_occ_a", "mo_occ_b", "mo_energy_a", "mo_energy_b"),
            ),
            ("density_matrices", ("dm_a", "dm_b")),
            ("active_space_mapping", ("active_indices", "orbital_labels", "orbital_labels_full")),
            (
                "diagnostics",
                (
                    "bs_stabilize_energy_history",
                    "bs_stabilize_delta_e_history",
                    "bs_tight_energy_history",
                    "bs_tight_delta_e_history",
                    "newton_energy_history",
                    "newton_delta_e_history",
                ),
            ),
        ):
            group = f.get(group_name)
            if group is None:
                continue
            for key in keys:
                if key in group:
                    payload[key] = group[key][()]
    return payload


def save_dmrg_basis_h5(h5_path: str, payload: dict):
    """Persist a step7 DMRG-basis payload as structured HDF5."""
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["artifact_type"] = "apex_filter_step7_dmrg_basis"

        orbitals = f.create_group("orbitals")
        for key in (
            "mo_coeff_alpha",
            "mo_coeff_beta",
            "active_coeff_alpha",
            "active_coeff_beta",
            "alpha_no_occupations",
            "beta_no_occupations",
            "pairs",
            "ordering",
        ):
            if key in payload:
                _write_dataset(orbitals, key, payload[key])

        for key in (
            "nocc_alpha",
            "nocc_beta",
            "localization_method",
            "source_method",
            "ordering_matrix_mode",
            "ordering_objective",
            "pair_diag_overlap_min",
            "pair_diag_overlap_mean",
            "diag_dominant_fraction",
            "orth_err_alpha",
            "orth_err_beta",
            "ordering_is_permutation",
            "ga_cost",
            "fiedler_cost",
        ):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, (str, bytes, np.str_, np.bytes_)):
                meta.attrs[key] = str(value)
            else:
                meta.attrs[key] = value


def load_dmrg_basis_h5(h5_path: str) -> dict:
    """Load a step7 DMRG-basis HDF5 state into a dict compatible with NPZ keys."""
    payload = {}
    with h5py.File(h5_path, "r") as f:
        meta = f.get("metadata")
        if meta is not None:
            for key, value in meta.attrs.items():
                payload[key] = value
        orbitals = f.get("orbitals")
        if orbitals is not None:
            for key in orbitals.keys():
                payload[key] = orbitals[key][()]
    return payload


def _copy_group_contents(src_group, dst_group):
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value
    for key in src_group.keys():
        src_obj = src_group[key]
        if isinstance(src_obj, h5py.Dataset):
            data = src_obj[()]
            if getattr(data, "shape", ()) == ():
                dst_group.create_dataset(key, data=data)
            else:
                kwargs = {}
                if src_obj.compression is not None:
                    kwargs["compression"] = src_obj.compression
                if src_obj.compression_opts is not None:
                    kwargs["compression_opts"] = src_obj.compression_opts
                if src_obj.shuffle:
                    kwargs["shuffle"] = True
                dst_group.create_dataset(key, data=data, **kwargs)
        elif isinstance(src_obj, h5py.Group):
            child = dst_group.create_group(key)
            _copy_group_contents(src_obj, child)


def save_hast_state_h5(
    h5_path: str,
    *,
    result,
    reference_state_path: str | None = None,
    reference_state_payload: dict | None = None,
):
    """Persist a step6 HAST-UCC result as structured HDF5.

    The step6 file intentionally reuses the step3 HDF5 molecule/orbital
    provenance when available, and layers correlated observables plus density
    matrices on top.
    """
    reference_state_payload = reference_state_payload or {}
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["artifact_type"] = "apex_filter_step6_hast_ucc_state"
        _attr_set(meta, "method", result.method)
        _attr_set(meta, "nominal_backend", result.nominal_backend)
        _attr_set(meta, "energy", result.energy)
        _attr_set(meta, "correlation_energy", result.correlation_energy)
        _attr_set(meta, "converged", result.converged)
        _attr_set(meta, "s_squared", result.s_squared)
        _attr_set(meta, "t1_norm", result.t1_norm)
        _attr_set(meta, "uhf_energy", result.uhf_energy)
        _attr_set(meta, "two_s", result.two_s)
        _attr_set(meta, "two_sz_fe1", result.two_sz_fe1)
        _attr_set(meta, "two_sz_fe2", result.two_sz_fe2)
        _attr_set(meta, "observables_complete", getattr(result, "observables_complete", None))
        _attr_set(meta, "lambda_converged", getattr(result, "lambda_converged", None))
        _attr_set(meta, "observable_error", getattr(result, "observable_error", None))
        if result.post_scf_observables is not None:
            _attr_set(meta, "post_scf_observables_json", result.post_scf_observables)

        if reference_state_path and reference_state_path.endswith(".h5") and os.path.isfile(reference_state_path):
            with h5py.File(reference_state_path, "r") as src:
                if "molecule" in src:
                    _copy_group_contents(src["molecule"], f.create_group("molecule"))
                if "active_space_mapping" in src:
                    _copy_group_contents(src["active_space_mapping"], f.create_group("active_space_mapping"))
        else:
            mapping = f.create_group("active_space_mapping")
            if "active_indices" in reference_state_payload:
                _write_dataset(mapping, "active_indices", np.asarray(reference_state_payload["active_indices"], dtype=int))
            for key in ("orbital_labels", "orbital_labels_full"):
                if key in reference_state_payload:
                    dt = h5py.string_dtype("utf-8")
                    mapping.create_dataset(
                        key,
                        data=np.array([str(x) for x in reference_state_payload[key]], dtype=dt),
                    )

        orbitals = f.create_group("orbitals")
        for key in ("mo_coeff_a", "mo_coeff_b", "mo_occ_a", "mo_occ_b", "mo_energy_a", "mo_energy_b"):
            if key in reference_state_payload:
                _write_dataset(orbitals, key, reference_state_payload[key])

        density = f.create_group("density_matrices")
        for key in ("dm1a_mo", "dm1b_mo", "dm1a_active", "dm1b_active"):
            value = getattr(result, key, None)
            if value is not None:
                _write_dataset(density, key, value)

        amplitudes = f.create_group("amplitudes")
        for key in ("tamps_vector", "lamps_vector"):
            value = getattr(result, key, None)
            if value is not None:
                _write_dataset(amplitudes, key, np.asarray(value, dtype=float))

        diagnostics = f.create_group("diagnostics")
        _attr_set(diagnostics, "t1_norm", result.t1_norm)


def save_dmrg_state_h5(
    h5_path: str,
    *,
    result,
    reference_state_path: str | None = None,
    basis_state_path: str | None = None,
    scratch_dir: str | None = None,
):
    """Persist a step8 DMRG result as structured HDF5."""
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["artifact_type"] = "apex_filter_step8_dmrg_state"
        for key in (
            "method",
            "backend",
            "basis_mode",
            "schedule_mode",
            "bond_dim",
            "n_sweeps",
            "energy",
            "correlation_energy",
            "converged",
            "s_squared",
            "uhf_energy",
            "twosite_to_onesite",
            "dav_max_iter",
            "dav_def_max_size",
            "dav_rel_conv_thrd",
            "dav_type",
            "wall_time_s",
            "log_path",
        ):
            _attr_set(meta, key, getattr(result, key, None))

        schedule = f.create_group("schedule")
        _write_dataset(schedule, "bond_dims", np.asarray(result.bond_dims, dtype=int))
        _write_dataset(schedule, "noises", np.asarray(result.noises, dtype=float))
        _write_dataset(schedule, "thresholds", np.asarray(result.thresholds, dtype=float))

        if reference_state_path and reference_state_path.endswith(".h5") and os.path.isfile(reference_state_path):
            with h5py.File(reference_state_path, "r") as src:
                if "molecule" in src:
                    _copy_group_contents(src["molecule"], f.create_group("molecule"))
                if "active_space_mapping" in src:
                    _copy_group_contents(src["active_space_mapping"], f.create_group("active_space_mapping"))

        if basis_state_path and basis_state_path.endswith(".h5") and os.path.isfile(basis_state_path):
            with h5py.File(basis_state_path, "r") as src:
                basis_group = f.create_group("basis_state")
                if "metadata" in src:
                    _copy_group_contents(src["metadata"], basis_group.create_group("metadata"))
                if "orbitals" in src:
                    _copy_group_contents(src["orbitals"], basis_group.create_group("orbitals"))

        dmrg_diag = f.create_group("dmrg_diagnostics")
        _attr_set(dmrg_diag, "scratch_dir", scratch_dir)
        if scratch_dir:
            node0 = os.path.join(scratch_dir, "node0")
            for name, dtype in (
                ("E_dmrg.npy", float),
                ("bond_dims.npy", int),
                ("discarded_weights.npy", float),
            ):
                path = os.path.join(node0, name)
                if os.path.isfile(path):
                    arr = np.load(path, allow_pickle=True)
                    _write_dataset(dmrg_diag, name[:-4], np.asarray(arr, dtype=dtype))
            for name, dtype in (
                ("1pdm.npy", float),
                ("2pdm.npy", float),
            ):
                path = os.path.join(node0, name)
                if os.path.isfile(path):
                    arr = np.load(path, allow_pickle=True)
                    density = f.require_group("density_matrices")
                    _write_dataset(density, name[:-4], np.asarray(arr, dtype=dtype))
