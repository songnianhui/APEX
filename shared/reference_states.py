"""Shared loaders for saved active-space reference states."""

from __future__ import annotations

import h5py
import numpy as np

from .active_space_reference import (
    build_fake_mol as _build_fake_mol,
    build_reference_uhf_solver as _build_reference_uhf_solver,
)


def load_reference_state_payload(uhf_state_path: str) -> dict:
    """Load a saved Step 3 reference state from NPZ or HDF5."""
    if uhf_state_path.endswith(".h5"):
        payload = {}
        with h5py.File(uhf_state_path, "r") as f:
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

    npz = np.load(uhf_state_path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_reference_mf_from_npz(fcidump_data, uhf_state_path: str):
    """Rebuild a UHF object on the FCIDUMP Hamiltonian from saved Step 3 data."""
    data = load_reference_state_payload(uhf_state_path)

    mo_occ = (data["mo_occ_a"], data["mo_occ_b"])
    ms2 = int(round(float(np.sum(mo_occ[0]) - np.sum(mo_occ[1]))))
    mol = _build_fake_mol(
        fcidump_data.norb,
        fcidump_data.nelec,
        ms2,
        ecore=fcidump_data.ecore,
    )
    mf = _build_reference_uhf_solver(fcidump_data, mol)
    mf.mo_coeff = (data["mo_coeff_a"], data["mo_coeff_b"])
    mf.mo_occ = mo_occ
    mf.mo_energy = (data["mo_energy_a"], data["mo_energy_b"])
    mf.e_tot = float(data["energy"])
    mf.converged = bool(data["converged"])
    return mf
