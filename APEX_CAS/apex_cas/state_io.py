"""State persistence helpers for SCF/CAS workflows.

The staged CLI uses internal ``_save_*`` helpers to persist summaries and CAS
artifacts. ``load_cas_state(...)`` remains the public readback entry point used
by downstream workflow stages.
"""

import json
import logging
import os
import warnings
from types import SimpleNamespace as _SimpleNamespace

import h5py
import numpy as np
from pyscf import lib
from pyscf import scf as _scf

from shared.chkfiles import find_chkfile as _find_chkfile
from shared.models import CAS as _CAS
from shared.settings_payloads import (
    build_effective_localization_payload as _build_effective_localization_payload,
    build_effective_parameter_payload as _build_effective_parameter_payload,
    build_effective_selection_payload as _build_effective_selection_payload,
    build_requested_cas_payload as _build_requested_cas_payload,
    normalize_settings_payload as _normalize_settings_payload,
)

from .CAS_builder import _build_mf_object

logger = logging.getLogger(__name__)


def _decode_h5_string(value):
    """Decode HDF5 string scalars/bytes into plain Python str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _write_string_dataset(group, key: str, values):
    """Write a UTF-8 string dataset."""
    dt = h5py.string_dtype("utf-8")
    arr = np.array([str(v) for v in values], dtype=dt)
    group.create_dataset(key, data=arr)


def _extract_mol_symbols_positions(mol):
    """Return (symbols, positions) from a PySCF-like Mole object when available."""
    if mol is None:
        return None, None

    symbols = None
    positions = None

    try:
        symbols = list(mol.elements)
    except Exception:
        try:
            symbols = [str(atom[0]) for atom in getattr(mol, "_atom", [])]
        except Exception:
            symbols = None

    try:
        positions = np.asarray(mol.atom_coords(), dtype=float)
    except Exception:
        try:
            positions = np.asarray([atom[1] for atom in getattr(mol, "_atom", [])], dtype=float)
        except Exception:
            positions = None

    if symbols is not None and positions is not None and len(symbols) == len(positions):
        return symbols, positions
    return symbols, positions


def _serialize_solver_mol(mol):
    """Serialize the solver Mole when possible for provenance/restart support."""
    if mol is None:
        return None
    try:
        return mol.dumps()
    except Exception:
        return None


def _build_mf_settings_namespace(mapping: dict):
    """Build the settings payload needed to reconstruct the SCF object."""
    if not mapping:
        return None
    return _SimpleNamespace(
        scf_method=str(mapping.get("scf_method", "uks")),
        xc_functional=str(mapping.get("xc_functional", "B3LYP")),
        relativistic=str(mapping.get("relativistic", "none")),
        solvation_model=str(mapping.get("solvation_model", "none")),
        solvation_epsilon=float(mapping.get("solvation_epsilon", 4.0)),
        density_fit=bool(mapping.get("density_fit", False)),
        density_fit_auxbasis=mapping.get("density_fit_auxbasis"),
        density_fit_only_dfj=bool(mapping.get("density_fit_only_dfj", False)),
        grids_level=int(mapping.get("grids_level", 3)),
        grids_small_rho_cutoff=float(mapping.get("grids_small_rho_cutoff", 1e-7)),
        grids_prune=str(mapping.get("grids_prune", "nwchem")),
        frac_occ=bool(mapping.get("frac_occ", False)),
        smearing_method=str(mapping.get("smearing_method", "none")),
        smearing_sigma=float(mapping.get("smearing_sigma", 0.01)),
    )


def _build_scf_settings_payload(settings) -> dict:
    """Build grouped SCF-stage settings for JSON persistence."""
    payload = {
        "scf_spin": getattr(settings, "scf_spin", None),
        "scf_method": getattr(settings, "scf_method", "uks"),
        "xc_functional": getattr(settings, "xc_functional", "B3LYP"),
        "basis_set_default": getattr(settings, "basis_set_default", "def2-TZVP"),
        "basis_set_per_element": dict(getattr(settings, "basis_set_per_element", {})),
        "basis_set_file": getattr(settings, "basis_set_file", None),
        "relativistic": getattr(settings, "relativistic", "none"),
        "solvation_model": getattr(settings, "solvation_model", "none"),
        "density_fit": getattr(settings, "density_fit", False),
        "density_fit_auxbasis": getattr(settings, "density_fit_auxbasis", None),
        "density_fit_only_dfj": getattr(settings, "density_fit_only_dfj", False),
        "grids_level": getattr(settings, "grids_level", 3),
        "grids_small_rho_cutoff": getattr(settings, "grids_small_rho_cutoff", 1e-7),
        "grids_prune": getattr(settings, "grids_prune", "nwchem"),
        "conv_tol": getattr(settings, "conv_tol", 1e-8),
        "max_cycle": getattr(settings, "max_cycle", 2000),
        "scf_verbose": getattr(settings, "scf_verbose", 4),
        "init_guess": getattr(settings, "init_guess", "atom"),
        "scf_damp": getattr(settings, "scf_damp", 0.0),
        "scf_level_shift": getattr(settings, "scf_level_shift", 0.0),
        "diis_space": getattr(settings, "diis_space", 8),
        "frac_occ": getattr(settings, "frac_occ", False),
        "smearing_method": getattr(settings, "smearing_method", "none"),
        "smearing_sigma": getattr(settings, "smearing_sigma", 0.01),
        "scf_stage1_rough": getattr(settings, "scf_stage1_rough", False),
        "scf_stage3_newton": getattr(settings, "scf_stage3_newton", False),
        "newton_max_cycle": getattr(settings, "newton_max_cycle", 10),
        "newton_conv_tol": getattr(settings, "newton_conv_tol", 1e-10),
        "scf_allow_unconverged": getattr(settings, "scf_allow_unconverged", False),
    }
    if getattr(settings, "solvation_model", "none") != "none":
        payload["solvation_epsilon"] = getattr(settings, "solvation_epsilon", 4.0)
    return payload


def _merge_nested_dict(dst: dict, src: dict):
    """Recursively merge src into dst in-place."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_nested_dict(dst[key], value)
        else:
            dst[key] = value


def _update_cas_summary(
    output_dir: str,
    stem: str,
    *,
    generated_files: dict | None = None,
    effective_method: dict | None = None,
) -> str:
    """Patch the saved CAS summary with later-stage artifact paths or method metadata."""
    cas_info_path = os.path.join(output_dir, "scf", f"{stem}_cas_info.json")
    if not os.path.isfile(cas_info_path):
        raise FileNotFoundError(f"CAS summary not found: {cas_info_path}")

    with open(cas_info_path) as f:
        payload = json.load(f)

    if generated_files:
        payload.setdefault("results", {}).setdefault("generated_files", {})
        payload["results"]["generated_files"].update(
            {k: v for k, v in generated_files.items() if v}
        )
    if effective_method:
        payload.setdefault("effective_method", {})
        _merge_nested_dict(payload["effective_method"], effective_method)

    with open(cas_info_path, "w") as f:
        json.dump(payload, f, indent=2)
    return cas_info_path


def _compute_energy_decomposition(cas, mf):
    """Compute qualitative core/active/virtual energy decomposition."""
    occ_full = cas.occupations_full
    mo_coeff = cas.mo_coeff_full
    if occ_full is None or mo_coeff is None:
        return 0.0, 0.0, 0.0

    try:
        fock = np.asarray(mf.get_fock())
        if fock.ndim == 3:
            fock = 0.5 * (fock[0] + fock[1])
    except Exception:
        return 0.0, 0.0, 0.0

    n = min(len(occ_full), mo_coeff.shape[1])
    fock_uno = mo_coeff[:, :n].T @ fock @ mo_coeff[:, :n]
    orbital_energies = np.diag(fock_uno)

    occ = occ_full[:n]
    core_mask = occ > 1.98
    act_mask = (occ >= 0.02) & (occ <= 1.98)
    vir_mask = occ < 0.02

    e_core = float(np.sum(occ[core_mask] * orbital_energies[core_mask]))
    e_act = float(np.sum(occ[act_mask] * orbital_energies[act_mask]))
    e_vir = float(np.sum(occ[vir_mask] * orbital_energies[vir_mask]))
    return e_core, e_act, e_vir


def _build_cas_effective_method_payload(cas, settings, settings_payload: dict | None) -> dict:
    """Build the effective CAS route/method block shared across JSON and HDF5."""
    return {
        "cpt_cas_type": cas.cpt_cas_type,
        "source_method": cas.source_method,
        "localization": _build_effective_localization_payload(settings, settings_payload),
        "selection": _build_effective_selection_payload(cas, settings_payload),
    }


def _build_cas_settings_payload(
    cas,
    settings,
    settings_payload: dict | None,
    *,
    requested_config,
) -> dict:
    """Build the canonical requested/effective CAS settings payload."""
    payload = dict(settings_payload or {})
    payload["requested_config"] = requested_config
    payload["effective_method"] = _build_cas_effective_method_payload(cas, settings, settings_payload)
    payload["effective_parameters"] = _build_effective_parameter_payload(cas, settings, settings_payload)
    return payload


def _save_scf_summary(
    mf,
    output_dir: str,
    stem: str,
    settings=None,
    charge: int = 0,
    target_spin: float = 0.0,
) -> str:
    """Save SCF-only metadata for later buildcas restoration."""
    scf_dir = os.path.join(output_dir, "scf")
    os.makedirs(scf_dir, exist_ok=True)

    e_solvent = 0.0
    if hasattr(mf, "with_solvent"):
        e_solvent = float(mf.scf_summary.get("e_solvent", 0.0))

    payload = {
        "energy": float(mf.e_tot),
        "E_solvent": e_solvent,
        "E_gas_phase": float(mf.e_tot) - e_solvent,
        "converged": bool(mf.converged),
        "charge": charge,
        "target_spin": target_spin,
    }
    if settings is not None:
        scf_settings_payload = _normalize_settings_payload(_build_scf_settings_payload(settings))
        payload["settings"] = {"scf": scf_settings_payload}
        # Keep a flat top-level mirror of the most important SCF identity/control
        # fields so historical compare scripts and quick manual inspection do not
        # need to traverse the nested settings block.
        for key in (
            "basis_set_default",
            "basis_set_per_element",
            "scf_method",
            "xc_functional",
            "relativistic",
            "solvation_model",
            "conv_tol",
            "max_cycle",
            "scf_spin",
        ):
            if key in scf_settings_payload:
                payload[key] = scf_settings_payload[key]

    scf_info_path = os.path.join(scf_dir, f"{stem}_scf_info.json")
    with open(scf_info_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("SCF summary saved to %s", scf_info_path)
    return scf_info_path


def _load_scf_summary(output_dir: str, stem: str) -> dict:
    """Load SCF-only metadata previously written by `_save_scf_summary()`."""
    scf_info_path = os.path.join(output_dir, "scf", f"{stem}_scf_info.json")
    if not os.path.isfile(scf_info_path):
        return {}
    with open(scf_info_path) as f:
        return json.load(f)


def _save_fcidump_summary(
    output_dir: str,
    stem: str,
    *,
    fcidump_path: str,
    selection_file: str,
    n_electrons: int,
    n_orbitals: int,
    ms2: int,
    target_spin: float,
    zero_ecore: bool,
    frozen_core_indices=None,
    settings_payload: dict | None = None,
) -> str:
    """Save FCIDUMP-stage metadata beside the generated integrals."""
    fcidump_dir = os.path.join(output_dir, "fcidump")
    os.makedirs(fcidump_dir, exist_ok=True)

    payload = {
        "fcidump_path": os.path.abspath(fcidump_path),
        "selection_file": os.path.abspath(selection_file),
        "n_electrons": int(n_electrons),
        "n_orbitals": int(n_orbitals),
        "ms2": int(ms2),
        "target_spin": float(target_spin),
        "zero_ecore": bool(zero_ecore),
        "frozen_core_indices": list(frozen_core_indices or []),
    }
    if settings_payload:
        payload["settings"] = {"fcidump": _normalize_settings_payload(settings_payload)}

    info_path = os.path.join(fcidump_dir, f"{stem}_fcidump_info.json")
    with open(info_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("FCIDUMP summary saved to %s", info_path)
    return info_path


def _save_dmrg_summary(
    dmrg_dir: str,
    stem: str,
    *,
    fcidump_path: str,
    h5_path: str,
    noon_plot_path: str,
    bond_dim: int,
    symm_type: str,
    n_orb: int,
    n_elec: int,
    ms2: int,
    e_active: float,
    e_core: float,
    e_total: float,
    wall_time_s: float,
    spin_squared=None,
    settings_payload: dict | None = None,
) -> str:
    """Save DMRG-stage metadata beside the DMRG results."""
    os.makedirs(dmrg_dir, exist_ok=True)
    payload = {
        "fcidump_path": os.path.abspath(fcidump_path),
        "results_h5": os.path.abspath(h5_path),
        "noon_plot_path": os.path.abspath(noon_plot_path),
        "bond_dim": int(bond_dim),
        "symm_type": str(symm_type),
        "n_orb": int(n_orb),
        "n_elec": int(n_elec),
        "ms2": int(ms2),
        "e_active": float(e_active),
        "e_core": float(e_core),
        "e_total": float(e_total),
        "wall_time_s": float(wall_time_s),
    }
    if spin_squared is not None:
        payload["spin_squared"] = float(spin_squared)
    if settings_payload:
        payload["settings"] = {"dmrg": _normalize_settings_payload(settings_payload)}

    info_path = os.path.join(dmrg_dir, f"{stem}_dmrg_info.json")
    with open(info_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("DMRG summary saved to %s", info_path)
    return info_path


def _save_cas_state(
    cas,
    mol,
    mf,
    output_dir: str,
    stem: str,
    settings=None,
    charge: int = 0,
    target_spin: float = 0.0,
    settings_payload: dict | None = None,
) -> str:
    """Save CAS full state to disk (HDF5)."""
    scf_dir = os.path.join(output_dir, "scf")
    orbitals_dir = os.path.join(output_dir, "orbitals")
    os.makedirs(scf_dir, exist_ok=True)
    os.makedirs(orbitals_dir, exist_ok=True)

    cas_info_name = f"{stem}_cas_info.json"
    h5_name = f"{stem}_cas_data.h5"
    cas_info_path = os.path.join(scf_dir, cas_info_name)
    h5_path = os.path.join(orbitals_dir, h5_name)

    e_core, e_act, e_vir = _compute_energy_decomposition(cas, mf)

    e_solvent = 0.0
    if hasattr(mf, "with_solvent"):
        e_solvent = float(mf.scf_summary.get("e_solvent", 0.0))
        logger.info("Solvent energy extracted: %.10f Hartree", e_solvent)

    requested_cas_payload = _build_requested_cas_payload(settings_payload)
    cas_settings_payload = _build_cas_settings_payload(
        cas,
        settings,
        settings_payload,
        requested_config=requested_cas_payload,
    )
    cas_info = {
        "requested_config": requested_cas_payload,
        "results": {
            "energy": float(mf.e_tot),
            "E_solvent": e_solvent,
            "E_gas_phase": float(mf.e_tot) - e_solvent,
            "E_core": e_core,
            "E_act": e_act,
            "E_vir": e_vir,
            "E_tot": float(mf.e_tot),
            "converged": bool(mf.converged),
            "n_electrons": cas.n_electrons,
            "n_orbitals": cas.n_orbitals,
            "active_indices": cas.active_indices,
            "generated_files": {
                "cas_info_json": cas_info_path,
                "cas_data_h5": h5_path,
            },
        },
        "effective_method": cas_settings_payload["effective_method"],
        "effective_parameters": cas_settings_payload["effective_parameters"],
    }
    with open(cas_info_path, "w") as f:
        json.dump(cas_info, f, indent=2)

    h5_kwargs = dict(compression="gzip", compression_opts=9)
    with h5py.File(h5_path, "w") as f:
        if cas.mo_coeff_full is not None:
            f.create_dataset("mo_coeff_full", data=cas.mo_coeff_full, **h5_kwargs)
        if cas.occupations_full is not None:
            f.create_dataset("occupations_full", data=cas.occupations_full, **h5_kwargs)
        if cas.mo_coeff_alpha is not None:
            f.create_dataset("mo_coeff_alpha", data=cas.mo_coeff_alpha, **h5_kwargs)
        if cas.mo_coeff_beta is not None:
            f.create_dataset("mo_coeff_beta", data=cas.mo_coeff_beta, **h5_kwargs)
        if cas.occupations is not None:
            f.create_dataset("occupations", data=cas.occupations, **h5_kwargs)

        if cas.orbital_labels_full:
            f.create_dataset(
                "orbital_labels_full",
                data=[str(label) for label in cas.orbital_labels_full],
            )
        if cas.orbital_labels:
            f.create_dataset(
                "orbital_labels",
                data=[str(label) for label in cas.orbital_labels],
            )

        meta = f.create_group("metadata")
        meta.attrs["n_electrons"] = cas.n_electrons
        meta.attrs["n_orbitals"] = cas.n_orbitals
        meta.attrs["cpt_cas_type"] = cas.cpt_cas_type
        meta.attrs["source_method"] = cas.source_method
        meta.attrs["description"] = cas.description
        meta.attrs["selection_method"] = cas.selection_method

        if cas.active_indices is not None:
            meta.create_dataset("active_indices", data=np.array(cas.active_indices, dtype=int))
        if cas.projection_weights is not None:
            meta.create_dataset("projection_weights", data=cas.projection_weights, **h5_kwargs)
        if cas.projection_weights_metal is not None:
            meta.create_dataset("projection_weights_metal", data=cas.projection_weights_metal, **h5_kwargs)
        if cas.projection_weights_bridging is not None:
            meta.create_dataset(
                "projection_weights_bridging",
                data=cas.projection_weights_bridging,
                **h5_kwargs,
            )

        if settings is not None:
            meta.attrs["scf_method"] = getattr(settings, "scf_method", "uks")
            meta.attrs["xc_functional"] = getattr(settings, "xc_functional", "B3LYP")
            meta.attrs["relativistic"] = getattr(settings, "relativistic", "none")
            meta.attrs["solvation_model"] = getattr(settings, "solvation_model", "none")
            meta.attrs["solvation_epsilon"] = getattr(settings, "solvation_epsilon", 4.0)
            meta.attrs["e_solvent"] = e_solvent
            meta.attrs["frac_occ"] = getattr(settings, "frac_occ", False)
            meta.attrs["smearing_method"] = getattr(settings, "smearing_method", "none")
            meta.attrs["smearing_sigma"] = getattr(settings, "smearing_sigma", 0.01)

        meta.attrs["charge"] = charge
        meta.attrs["target_spin"] = target_spin
        if settings_payload:
            h5_settings_payload = _build_cas_settings_payload(
                cas,
                settings,
                settings_payload,
                requested_config=dict(settings_payload),
            )
            meta.attrs["settings_json"] = json.dumps(
                h5_settings_payload,
                ensure_ascii=False,
            )
        if settings is not None:
            for key in (
                "basis_set_default",
                "basis_set_per_element",
                "density_fit",
                "density_fit_auxbasis",
                "density_fit_only_dfj",
                "grids_level",
                "grids_small_rho_cutoff",
                "grids_prune",
                "conv_tol",
                "max_cycle",
                "scf_verbose",
                "init_guess",
                "scf_damp",
                "scf_level_shift",
                "diis_space",
            ):
                if not hasattr(settings, key):
                    continue
                value = getattr(settings, key)
                if key == "basis_set_per_element":
                    meta.attrs["basis_set_per_element_json"] = json.dumps(value, ensure_ascii=False)
                elif value is None:
                    continue
                else:
                    meta.attrs[key] = value

        mapping = f.create_group("active_space_mapping")
        if cas.active_indices is not None:
            mapping.create_dataset("active_indices", data=np.array(cas.active_indices, dtype=int))
        if cas.orbital_labels:
            _write_string_dataset(mapping, "orbital_labels", cas.orbital_labels)
        if cas.orbital_labels_full:
            _write_string_dataset(mapping, "orbital_labels_full", cas.orbital_labels_full)

        molecule = f.create_group("molecule")
        molecule.attrs["charge"] = charge
        molecule.attrs["target_spin"] = target_spin
        if settings is not None:
            for key in (
                "basis_set_default",
                "scf_method",
                "xc_functional",
                "relativistic",
                "solvation_model",
                "solvation_epsilon",
            ):
                if hasattr(settings, key):
                    molecule.attrs[key] = getattr(settings, key)
            if hasattr(settings, "basis_set_per_element"):
                molecule.attrs["basis_set_per_element_json"] = json.dumps(
                    getattr(settings, "basis_set_per_element"),
                    ensure_ascii=False,
                )
        molecule.attrs["active_norb"] = cas.n_orbitals
        molecule.attrs["active_nelec"] = cas.n_electrons
        molecule.attrs["serialized_solver_mol"] = _serialize_solver_mol(mol) or ""
        symbols, positions = _extract_mol_symbols_positions(mol)
        if symbols:
            _write_string_dataset(molecule, "atom_symbols", symbols)
        if positions is not None and np.size(positions):
            molecule.create_dataset("atom_positions", data=np.asarray(positions, dtype=float), **h5_kwargs)

    logger.info("CAS state saved to %s", h5_path)
    return h5_path


def _read_mf_settings_from_h5(h5_path: str):
    """Read ComputationSettings fields from HDF5 metadata for mf reconstruction."""
    try:
        with h5py.File(h5_path, "r") as f:
            meta = f.get("metadata")
            if meta is None or "scf_method" not in meta.attrs:
                return None
            return _build_mf_settings_namespace(dict(meta.attrs.items()))
    except Exception:
        return None


def _read_mf_settings_from_scf_summary(output_dir: str, stem: str):
    """Read SCF-stage settings from the JSON sidecar when HDF5 metadata is older."""
    payload = _load_scf_summary(output_dir, stem)
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    scf_settings = settings.get("scf") if isinstance(settings, dict) else None
    if not isinstance(scf_settings, dict) or "scf_method" not in scf_settings:
        return None
    return _build_mf_settings_namespace(scf_settings)


def load_cas_state(case_dir: str):
    """Load CAS full state from disk."""
    output_dir = os.path.join(case_dir, "outputs")
    scf_dir = os.path.join(output_dir, "scf")
    chkfile = _find_chkfile(scf_dir)

    stem = os.path.splitext(os.path.basename(chkfile))[0]
    orbitals_dir = os.path.join(output_dir, "orbitals")
    h5_path = os.path.join(orbitals_dir, f"{stem}_cas_data.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"CAS data file not found: {h5_path}")

    mol = lib.chkfile.load_mol(chkfile)
    scf_data = lib.chkfile.load(chkfile, "scf")
    is_uhf = isinstance(scf_data.get("mo_coeff"), (list, np.ndarray)) and np.ndim(
        scf_data["mo_coeff"]
    ) == 3

    mf_settings = _read_mf_settings_from_h5(h5_path)
    if mf_settings is None:
        mf_settings = _read_mf_settings_from_scf_summary(output_dir, stem)
    if mf_settings is not None:
        mf = _build_mf_object(mol, mf_settings)
    else:
        warnings.warn(
            "No SCF settings found in CAS data file.  Restoring mf without "
            "relativistic/solvation decorators.  Re-run 'apex-cas scf' and "
            "'apex-cas buildcas' to update the saved state.",
            stacklevel=2,
        )
        mf = _scf.UHF(mol) if is_uhf else _scf.RHF(mol)
    mf.__dict__.update(scf_data)
    mf.chkfile = chkfile

    with h5py.File(h5_path, "r") as f:
        cas = _CAS()
        if "mo_coeff_full" in f:
            cas.mo_coeff_full = f["mo_coeff_full"][:]
        if "occupations_full" in f:
            cas.occupations_full = f["occupations_full"][:]
        if "mo_coeff_alpha" in f:
            cas.mo_coeff_alpha = f["mo_coeff_alpha"][:]
        if "mo_coeff_beta" in f:
            cas.mo_coeff_beta = f["mo_coeff_beta"][:]
        if "occupations" in f:
            cas.occupations = f["occupations"][:]

        if "orbital_labels_full" in f:
            cas.orbital_labels_full = [
                _decode_h5_string(label) for label in f["orbital_labels_full"][:]
            ]
        if "orbital_labels" in f:
            cas.orbital_labels = [
                _decode_h5_string(label) for label in f["orbital_labels"][:]
            ]

        meta = f["metadata"]
        cas.n_electrons = int(meta.attrs["n_electrons"])
        cas.n_orbitals = int(meta.attrs["n_orbitals"])
        cas.cpt_cas_type = str(meta.attrs.get("cpt_cas_type", "uno"))
        cas.source_method = str(meta.attrs.get("source_method", ""))
        cas.description = str(meta.attrs.get("description", ""))
        cas.selection_method = str(meta.attrs.get("selection_method", ""))
        cas.n_qubits = 2 * cas.n_orbitals

        mapping = f.get("active_space_mapping")
        if mapping is not None and "active_indices" in mapping:
            cas.active_indices = list(mapping["active_indices"][:].astype(int))
        elif "active_indices" in meta:
            cas.active_indices = list(meta["active_indices"][:].astype(int))
        if "projection_weights" in meta:
            cas.projection_weights = meta["projection_weights"][:]
        if "projection_weights_metal" in meta:
            cas.projection_weights_metal = meta["projection_weights_metal"][:]
        if "projection_weights_bridging" in meta:
            cas.projection_weights_bridging = meta["projection_weights_bridging"][:]

        if mapping is not None and "orbital_labels" in mapping:
            cas.orbital_labels = [
                _decode_h5_string(label) for label in mapping["orbital_labels"][:]
            ]
        if mapping is not None and "orbital_labels_full" in mapping:
            cas.orbital_labels_full = [
                _decode_h5_string(label)
                for label in mapping["orbital_labels_full"][:]
            ]

        cas.charge = int(meta.attrs.get("charge", 0))
        cas.target_spin = float(meta.attrs.get("target_spin", 0.0))

    logger.info("CAS state loaded from %s", h5_path)
    return cas, mol, mf
