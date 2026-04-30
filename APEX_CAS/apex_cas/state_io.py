"""State persistence helpers for SCF/CAS workflows."""

import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _decode_h5_string(value):
    """Decode HDF5 string scalars/bytes into plain Python str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _build_scf_settings_payload(settings) -> dict:
    """Build grouped SCF-stage settings for JSON persistence."""
    payload = {
        "scf_spin": settings.scf_spin,
        "scf_method": settings.scf_method,
        "xc_functional": settings.xc_functional,
        "basis_set_default": settings.basis_set_default,
        "basis_set_per_element": dict(settings.basis_set_per_element),
        "basis_set_file": settings.basis_set_file,
        "relativistic": settings.relativistic,
        "solvation_model": settings.solvation_model,
        "density_fit": settings.density_fit,
        "density_fit_auxbasis": settings.density_fit_auxbasis,
        "density_fit_only_dfj": settings.density_fit_only_dfj,
        "grids_level": settings.grids_level,
        "grids_small_rho_cutoff": settings.grids_small_rho_cutoff,
        "grids_prune": settings.grids_prune,
        "conv_tol": settings.conv_tol,
        "max_cycle": settings.max_cycle,
        "scf_verbose": settings.scf_verbose,
        "init_guess": settings.init_guess,
        "scf_damp": settings.scf_damp,
        "scf_level_shift": settings.scf_level_shift,
        "diis_space": settings.diis_space,
        "frac_occ": settings.frac_occ,
        "smearing_method": settings.smearing_method,
        "smearing_sigma": settings.smearing_sigma,
        "scf_stage1_rough": settings.scf_stage1_rough,
        "scf_stage3_newton": settings.scf_stage3_newton,
        "newton_max_cycle": settings.newton_max_cycle,
        "newton_conv_tol": settings.newton_conv_tol,
        "scf_allow_unconverged": settings.scf_allow_unconverged,
    }
    if settings.solvation_model != "none":
        payload["solvation_epsilon"] = settings.solvation_epsilon
    return payload


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


def save_scf_summary(
    mf,
    output_dir: str,
    stem: str,
    settings=None,
    charge: int = 0,
    target_spin: float = 0.0,
) -> str:
    """Save SCF-only metadata for later buildcas restoration."""
    import json as _json

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
        payload["settings"] = {"scf": _build_scf_settings_payload(settings)}

    scf_info_path = os.path.join(scf_dir, f"{stem}_scf_info.json")
    with open(scf_info_path, "w") as f:
        _json.dump(payload, f, indent=2)
    logger.info("SCF summary saved to %s", scf_info_path)
    return scf_info_path


def load_scf_summary(output_dir: str, stem: str) -> dict:
    """Load SCF-only metadata previously written by save_scf_summary."""
    import json as _json

    scf_info_path = os.path.join(output_dir, "scf", f"{stem}_scf_info.json")
    if not os.path.isfile(scf_info_path):
        return {}
    with open(scf_info_path) as f:
        return _json.load(f)


def save_fcidump_summary(
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
    reference_fcidump: str | None = None,
    settings_payload: dict | None = None,
) -> str:
    """Save FCIDUMP-stage metadata beside the generated integrals."""
    import json as _json

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
    if reference_fcidump:
        payload["reference_fcidump"] = os.path.abspath(reference_fcidump)
    if settings_payload:
        payload["settings"] = {"fcidump": settings_payload}

    info_path = os.path.join(fcidump_dir, f"{stem}_fcidump_info.json")
    with open(info_path, "w") as f:
        _json.dump(payload, f, indent=2)
    logger.info("FCIDUMP summary saved to %s", info_path)
    return info_path


def save_dmrg_summary(
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
    import json as _json

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
        payload["settings"] = {"dmrg": settings_payload}

    info_path = os.path.join(dmrg_dir, f"{stem}_dmrg_info.json")
    with open(info_path, "w") as f:
        _json.dump(payload, f, indent=2)
    logger.info("DMRG summary saved to %s", info_path)
    return info_path


def save_cas_state(
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
    import json as _json
    import h5py

    scf_dir = os.path.join(output_dir, "scf")
    orbitals_dir = os.path.join(output_dir, "orbitals")
    os.makedirs(scf_dir, exist_ok=True)
    os.makedirs(orbitals_dir, exist_ok=True)

    cas_info_name = f"{stem}_cas_info.json"
    h5_name = f"{stem}_cas_data.h5"

    e_core, e_act, e_vir = _compute_energy_decomposition(cas, mf)

    e_solvent = 0.0
    if hasattr(mf, "with_solvent"):
        e_solvent = float(mf.scf_summary.get("e_solvent", 0.0))
        logger.info("Solvent energy extracted: %.10f Hartree", e_solvent)

    cas_info = {
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
        "cpt_cas_type": cas.cpt_cas_type,
        "source_method": cas.source_method,
        "selection_method": cas.selection_method,
        "active_indices": cas.active_indices,
    }
    if settings_payload:
        cas_info["settings"] = {"cas_build": settings_payload}
    cas_info_path = os.path.join(scf_dir, cas_info_name)
    with open(cas_info_path, "w") as f:
        _json.dump(cas_info, f, indent=2)

    h5_path = os.path.join(orbitals_dir, h5_name)
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
            f.create_dataset("orbital_labels_full", data=[str(l) for l in cas.orbital_labels_full])
        if cas.orbital_labels:
            f.create_dataset("orbital_labels", data=[str(l) for l in cas.orbital_labels])

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
            meta.attrs["scf_method"] = settings.scf_method
            meta.attrs["xc_functional"] = settings.xc_functional
            meta.attrs["relativistic"] = settings.relativistic
            meta.attrs["solvation_model"] = settings.solvation_model
            meta.attrs["solvation_epsilon"] = settings.solvation_epsilon
            meta.attrs["e_solvent"] = e_solvent
            meta.attrs["frac_occ"] = settings.frac_occ
            meta.attrs["smearing_method"] = settings.smearing_method
            meta.attrs["smearing_sigma"] = settings.smearing_sigma

        meta.attrs["charge"] = charge
        meta.attrs["target_spin"] = target_spin

    logger.info("CAS state saved to %s", h5_path)
    return h5_path


def find_chkfile(scf_dir: str) -> str:
    """Find a valid chkfile in the given directory."""
    import glob

    chk_candidates = [
        p for p in glob.glob(os.path.join(scf_dir, "*.chk")) if os.path.getsize(p) > 0
    ]
    if not chk_candidates:
        raise FileNotFoundError(f"No valid chkfile found in {scf_dir}/")

    chk_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    if len(chk_candidates) == 1:
        chkfile = chk_candidates[0]
    else:
        print(f"  Multiple chkfiles found in {scf_dir}/:")
        for i, p in enumerate(chk_candidates):
            size_kb = os.path.getsize(p) / 1024
            print(f"    [{i}] {os.path.basename(p)}  ({size_kb:.1f} KB)")
        if sys.stdin.isatty():
            resp = input(f"  Select [0-{len(chk_candidates)-1}] (default=0): ").strip()
            idx = 0 if resp == "" else int(resp)
        else:
            print("  Non-interactive mode: using first chkfile.")
            idx = 0
        chkfile = chk_candidates[idx]

    print(f"  Using chkfile: {os.path.basename(chkfile)}")
    return chkfile


def _read_mf_settings_from_h5(h5_path: str):
    """Read ComputationSettings fields from HDF5 metadata for mf reconstruction."""
    import h5py

    try:
        with h5py.File(h5_path, "r") as f:
            meta = f.get("metadata")
            if meta is None or "scf_method" not in meta.attrs:
                return None

            from . import ComputationSettings

            return ComputationSettings(
                scf_method=str(meta.attrs.get("scf_method", "uks")),
                xc_functional=str(meta.attrs.get("xc_functional", "B3LYP")),
                relativistic=str(meta.attrs.get("relativistic", "none")),
                solvation_model=str(meta.attrs.get("solvation_model", "none")),
                solvation_epsilon=float(meta.attrs.get("solvation_epsilon", 4.0)),
                frac_occ=bool(meta.attrs.get("frac_occ", False)),
                smearing_method=str(meta.attrs.get("smearing_method", "none")),
                smearing_sigma=float(meta.attrs.get("smearing_sigma", 0.01)),
            )
    except Exception:
        return None


def load_cas_state(case_dir: str):
    """Load CAS full state from disk."""
    import h5py
    from pyscf import gto, scf as _scf

    output_dir = os.path.join(case_dir, "outputs")
    scf_dir = os.path.join(output_dir, "scf")
    chkfile = find_chkfile(scf_dir)

    stem = os.path.splitext(os.path.basename(chkfile))[0]
    orbitals_dir = os.path.join(output_dir, "orbitals")
    h5_path = os.path.join(orbitals_dir, f"{stem}_cas_data.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"CAS data file not found: {h5_path}")

    from pyscf import lib

    mol = lib.chkfile.load_mol(chkfile)
    scf_data = lib.chkfile.load(chkfile, "scf")
    is_uhf = isinstance(scf_data.get("mo_coeff"), (list, np.ndarray)) and np.ndim(
        scf_data["mo_coeff"]
    ) == 3

    mf_settings = _read_mf_settings_from_h5(h5_path)
    if mf_settings is not None:
        from .CAS_builder import _build_mf_object

        mf = _build_mf_object(mol, mf_settings)
    else:
        import warnings

        warnings.warn(
            "No SCF settings found in CAS data file.  Restoring mf without "
            "relativistic/solvation decorators.  Re-run 'apex-cas compute' "
            "to update the saved state.",
            stacklevel=2,
        )
        mf = _scf.UHF(mol) if is_uhf else _scf.RHF(mol)
    mf.__dict__.update(scf_data)
    mf.chkfile = chkfile

    from . import CAS

    with h5py.File(h5_path, "r") as f:
        cas = CAS()
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
            cas.orbital_labels_full = [_decode_h5_string(l) for l in f["orbital_labels_full"][:]]
        if "orbital_labels" in f:
            cas.orbital_labels = [_decode_h5_string(l) for l in f["orbital_labels"][:]]

        meta = f["metadata"]
        cas.n_electrons = int(meta.attrs["n_electrons"])
        cas.n_orbitals = int(meta.attrs["n_orbitals"])
        cas.cpt_cas_type = str(meta.attrs.get("cpt_cas_type", "uno"))
        cas.source_method = str(meta.attrs.get("source_method", ""))
        cas.description = str(meta.attrs.get("description", ""))
        cas.selection_method = str(meta.attrs.get("selection_method", ""))
        cas.n_qubits = 2 * cas.n_orbitals

        if "active_indices" in meta:
            cas.active_indices = list(meta["active_indices"][:].astype(int))
        if "projection_weights" in meta:
            cas.projection_weights = meta["projection_weights"][:]
        if "projection_weights_metal" in meta:
            cas.projection_weights_metal = meta["projection_weights_metal"][:]
        if "projection_weights_bridging" in meta:
            cas.projection_weights_bridging = meta["projection_weights_bridging"][:]

        cas.charge = int(meta.attrs.get("charge", 0))
        cas.target_spin = float(meta.attrs.get("target_spin", 0.0))

    logger.info("CAS state loaded from %s", h5_path)
    return cas, mol, mf
