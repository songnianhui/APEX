"""Reference-state HAST-UCC on the active-space Hamiltonian defined by FCIDUMP."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging

import numpy as np

from .hdf5_state_io import save_hast_state_h5
from .post_scf_observables import analyze_active_space_spin_observables
from .reference_ucc import _load_reference_state_payload, load_reference_mf_from_npz


logger = logging.getLogger(__name__)


@dataclass
class ReferenceHASTResult:
    """Result of an active-space HAST-UCC calculation."""

    method: str
    energy: float
    correlation_energy: float
    converged: bool
    s_squared: float
    t1_norm: float
    uhf_energy: float
    nominal_backend: str
    two_s: float | None = None
    two_sz_fe1: float | None = None
    two_sz_fe2: float | None = None
    observables_complete: bool = False
    lambda_converged: bool | None = None
    observable_error: str | None = None
    post_scf_observables: dict | None = None
    reference_state_path: str | None = None
    reference_state_payload: dict | None = None
    tamps_vector: np.ndarray | None = None
    lamps_vector: np.ndarray | None = None
    dm1a_mo: np.ndarray | None = None
    dm1b_mo: np.ndarray | None = None
    dm1a_active: np.ndarray | None = None
    dm1b_active: np.ndarray | None = None


def _load_hast_restart_payload(h5_path: str) -> dict:
    import h5py

    payload: dict = {}
    with h5py.File(h5_path, "r") as f:
        meta = f.get("metadata")
        if meta is not None:
            for key in ("energy", "correlation_energy", "method", "nominal_backend"):
                if key in meta.attrs:
                    payload[key] = meta.attrs[key]
        amps = f.get("amplitudes")
        if amps is not None:
            if "tamps_vector" in amps:
                payload["tamps_vector"] = np.asarray(amps["tamps_vector"][()], dtype=float)
            if "lamps_vector" in amps:
                payload["lamps_vector"] = np.asarray(amps["lamps_vector"][()], dtype=float)
    return payload


def run_reference_hast_ucc(
    fcidump_data,
    uhf_npz_path: str,
    *,
    t_order: int = 3,
    max_cycle: int = 2000,
    conv_tol: float = 1e-8,
    residual_tol: float = 1e-6,
    diis: bool = True,
    diis_space: int = 6,
    diis_start_cycle: int = 0,
    iterative_damping: float = 1.0,
    level_shift: float = 0.0,
    newton_krylov: bool = False,
    frozen=None,
    mo_coeff=None,
    mo_occ=None,
    mo_energy=None,
    observable_inputs: dict | None = None,
    lambda_max_cycle: int | None = None,
    restart_h5_path: str | None = None,
    skip_energy_solve: bool = False,
):
    """Run HAST-UCC directly on the FCIDUMP Hamiltonian from a saved UHF state."""
    import pyhast

    if iterative_damping != 1.0:
        raise ValueError(
            "The current pyhast UCC implementation does not support non-unit "
            "iterative_damping for nested amplitude containers. Use "
            "iterative_damping = 1.0."
        )

    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    if mo_coeff is not None:
        mf.mo_coeff = mo_coeff
    if mo_occ is not None:
        mf.mo_occ = mo_occ
    if mo_energy is not None:
        mf.mo_energy = mo_energy
    ctor_kwargs = {
        "t_order": t_order,
        "verbose": 4,
        "diis": diis,
        "eval_t": "hastctr",
        "level_shift": level_shift,
        "newton_krylov": newton_krylov,
        "frozen": frozen,
    }
    if observable_inputs:
        ctor_kwargs.update(
            {
                "gen_lamb_eq": True,
                "gen_npdm_eq": True,
                "npdm_order": 2,
            }
        )
    try:
        mcc = pyhast.sr.ucc.UCC(mf, **ctor_kwargs)
    except TypeError:
        fallback_kwargs = {
            "t_order": t_order,
            "verbose": 4,
            "diis": diis,
            "eval_t": "hastctr",
        }
        if observable_inputs:
            fallback_kwargs.update(
                {
                    "gen_lamb_eq": True,
                    "gen_npdm_eq": True,
                    "npdm_order": 2,
                }
            )
        mcc = pyhast.sr.ucc.UCC(mf, **fallback_kwargs)
        if frozen is not None and hasattr(mcc, "frozen"):
            mcc.frozen = frozen
        if hasattr(mcc, "level_shift"):
            mcc.level_shift = level_shift
        if hasattr(mcc, "newton_krylov"):
            mcc.newton_krylov = newton_krylov
    if hasattr(mcc, "diis_space"):
        mcc.diis_space = diis_space
    if hasattr(mcc, "diis_start_cycle"):
        mcc.diis_start_cycle = diis_start_cycle
    if hasattr(mcc, "iterative_damping"):
        mcc.iterative_damping = iterative_damping

    restart_payload = _load_hast_restart_payload(restart_h5_path) if restart_h5_path else {}
    restart_tamps = restart_payload.get("tamps_vector")
    restart_lamps = restart_payload.get("lamps_vector")

    if restart_tamps is not None:
        mcc.tamps = mcc.vector_to_amplitudes(np.asarray(restart_tamps, dtype=float), mcc.order)

    if skip_energy_solve:
        if restart_tamps is None:
            raise ValueError("skip_energy_solve=True requires restart_h5_path with stored tamps_vector")
        if hasattr(mcc, "update_effective_ints"):
            mcc.update_effective_ints(mcc.tamps)
        corr = float(mcc.energy(mcc.tamps))
        energy = float(mf.e_tot) + corr
        mcc.e_corr = corr
        mcc.e_tot = energy
        kernel_result = energy
        converged = True
    else:
        kernel_result = mcc.kernel(max_cycle=max_cycle, tol=conv_tol, tolnormt=residual_tol)
        energy = getattr(mcc, "e_tot", None)
        if energy is None:
            if np.isscalar(kernel_result):
                energy = float(kernel_result)
            elif isinstance(kernel_result, (tuple, list)) and kernel_result:
                energy = float(kernel_result[0])
            else:
                raise RuntimeError("Could not determine total energy from HAST-UCC result")
        corr = getattr(mcc, "e_corr", float(energy) - float(mf.e_tot))
        converged = bool(getattr(mcc, "converged", True))

    t1_norm = float(getattr(mcc, "t1_norm", 0.0))

    method = "UCCSDTQ" if t_order >= 4 else "UCCSDT"
    try:
        s_squared = float(mf.spin_square()[0])
    except Exception:
        s_squared = 0.0

    two_s = None
    two_sz_fe1 = None
    two_sz_fe2 = None
    post_scf_observables = None
    state_payload = None
    dm1a_mo = None
    dm1b_mo = None
    dm1a_active = None
    dm1b_active = None
    lambda_converged = None
    observable_error = None
    observables_complete = False
    tamps_vector = np.asarray(mcc.amplitudes_to_vector(mcc.tamps), dtype=float)
    lamps_vector = None
    if observable_inputs:
        from pyscf.fci import spin_op

        try:
            state_payload = _load_reference_state_payload(uhf_npz_path)
            if "active_indices" not in observable_inputs:
                active_indices = state_payload.get("active_indices")
                if active_indices is None:
                    raise ValueError("Active-space mapping missing from step3 reference state; rerun step3 with HDF5 output.")
                observable_inputs = dict(observable_inputs)
                observable_inputs["active_indices"] = np.asarray(active_indices, dtype=int)
        except Exception as exc:
            observable_error = f"HAST-UCC observable stage load_reference_state failed: {exc}"
        else:
            try:
                lambda_kwargs = {"tol": conv_tol}
                if lambda_max_cycle is not None:
                    lambda_kwargs["max_cycle"] = int(lambda_max_cycle)
                if restart_lamps is not None:
                    lambda_kwargs["lamps"] = mcc.vector_to_amplitudes(
                        np.asarray(restart_lamps, dtype=float),
                        mcc.lamb_order,
                    )
                lambda_converged, lamps = mcc.solve_lambda(**lambda_kwargs)
                if getattr(mcc, "lamps", None) is not None:
                    lamps_vector = np.asarray(mcc.amplitudes_to_vector(mcc.lamps), dtype=float)
                if not lambda_converged:
                    observable_error = "HAST-UCC lambda equations did not converge"
                else:
                    try:
                        dm1, dm2 = mcc.make_npdms(order=2)
                        dm1a_mo, dm1b_mo = dm1
                        dm2aa, dm2ab, dm2bb = dm2
                    except Exception as exc:
                        observable_error = f"HAST-UCC observable stage make_npdms failed: {exc}"
                    else:
                        try:
                            s_squared = float(
                                spin_op.spin_square_general(
                                    dm1a_mo,
                                    dm1b_mo,
                                    dm2aa,
                                    dm2ab,
                                    dm2bb,
                                    mf.mo_coeff,
                                    mf.get_ovlp(),
                                )[0]
                            )
                        except Exception as exc:
                            observable_error = f"HAST-UCC observable stage spin_square_general failed: {exc}"
                        else:
                            try:
                                dm1a_active = (
                                    np.asarray(mf.mo_coeff[0], dtype=float)
                                    @ np.asarray(dm1a_mo, dtype=float)
                                    @ np.asarray(mf.mo_coeff[0], dtype=float).T
                                )
                                dm1b_active = (
                                    np.asarray(mf.mo_coeff[1], dtype=float)
                                    @ np.asarray(dm1b_mo, dtype=float)
                                    @ np.asarray(mf.mo_coeff[1], dtype=float).T
                                )
                                post_scf_observables = analyze_active_space_spin_observables(
                                    dm_a_active=dm1a_active,
                                    dm_b_active=dm1b_active,
                                    active_indices=np.asarray(observable_inputs["active_indices"], dtype=int),
                                    xyz_path=observable_inputs["xyz_path"],
                                    cluster_info_path=observable_inputs["cluster_info_path"],
                                    cas_settings_path=observable_inputs["cas_settings_path"],
                                    cas_data_h5_path=observable_inputs["cas_data_h5_path"],
                                    label=observable_inputs.get("label", ""),
                                    family=observable_inputs.get("family", ""),
                                    energy_hartree=float(energy),
                                    s2=s_squared,
                                    final_state_signature=observable_inputs.get("final_state_signature", ""),
                                    chan_benchmark_json=observable_inputs.get("chan_benchmark_json"),
                                    theory=observable_inputs.get("theory", method),
                                )
                                two_s = post_scf_observables.get("two_s")
                                primary = post_scf_observables.get("two_sz_by_metal_label", {})
                                two_sz_fe1 = primary.get("Fe1")
                                two_sz_fe2 = primary.get("Fe2")
                                observables_complete = True
                            except Exception as exc:
                                observable_error = f"HAST-UCC observable stage post_scf_analysis failed: {exc}"
            except Exception as exc:
                observable_error = f"HAST-UCC observable stage solve_lambda failed: {exc}"

        if post_scf_observables is None:
            post_scf_observables = {
                "status": "incomplete" if observable_error else "complete",
                "error": observable_error,
                "theory": observable_inputs.get("theory", method),
                "label": observable_inputs.get("label", ""),
                "family": observable_inputs.get("family", ""),
            }

    return ReferenceHASTResult(
        method=method,
        energy=float(energy),
        correlation_energy=float(corr),
        converged=converged,
        s_squared=s_squared,
        t1_norm=t1_norm,
        uhf_energy=float(mf.e_tot),
        nominal_backend=f"hast_ucc_t{t_order}",
        two_s=two_s,
        two_sz_fe1=two_sz_fe1,
        two_sz_fe2=two_sz_fe2,
        observables_complete=observables_complete,
        lambda_converged=lambda_converged,
        observable_error=observable_error,
        post_scf_observables=post_scf_observables,
        reference_state_path=uhf_npz_path,
        reference_state_payload=state_payload,
        tamps_vector=tamps_vector,
        lamps_vector=lamps_vector,
        dm1a_mo=np.asarray(dm1a_mo, dtype=float) if dm1a_mo is not None else None,
        dm1b_mo=np.asarray(dm1b_mo, dtype=float) if dm1b_mo is not None else None,
        dm1a_active=np.asarray(dm1a_active, dtype=float) if dm1a_active is not None else None,
        dm1b_active=np.asarray(dm1b_active, dtype=float) if dm1b_active is not None else None,
    )


def save_reference_hast_result(result: ReferenceHASTResult, npz_path: str):
    """Save active-space HAST-UCC results in parser-compatible schema."""
    payload = {
        "hast_method_level": result.nominal_backend,
        "hast_total": result.energy,
        "hast_corr": result.correlation_energy,
        "hast_converged": result.converged,
        "spin_sq": result.s_squared,
        "t1_norm": result.t1_norm,
        "uhf_energy": result.uhf_energy,
        "observables_complete": result.observables_complete,
        "lambda_converged": np.asarray(result.lambda_converged, dtype=object),
        "observable_error": np.asarray(result.observable_error, dtype=object),
    }
    if result.two_s is not None:
        payload["two_s"] = result.two_s
    if result.two_sz_fe1 is not None:
        payload["two_sz_fe1"] = result.two_sz_fe1
    if result.two_sz_fe2 is not None:
        payload["two_sz_fe2"] = result.two_sz_fe2
    if result.post_scf_observables is not None:
        payload["post_scf_observables_json"] = np.array(
            json.dumps(result.post_scf_observables, ensure_ascii=False),
            dtype=np.str_,
        )

    if result.method == "UCCSDTQ":
        payload["ccsdtq_total"] = result.energy
        payload["ccsdtq_corr"] = result.correlation_energy
        payload["ccsdtq_converged"] = result.converged
    else:
        payload["ccsdt_total"] = result.energy
        payload["ccsdt_corr"] = result.correlation_energy
        payload["ccsdt_converged"] = result.converged

    if result.dm1a_mo is not None:
        payload["dm1a_mo"] = result.dm1a_mo
    if result.dm1b_mo is not None:
        payload["dm1b_mo"] = result.dm1b_mo
    if result.dm1a_active is not None:
        payload["dm1a_active"] = result.dm1a_active
    if result.dm1b_active is not None:
        payload["dm1b_active"] = result.dm1b_active
    if result.tamps_vector is not None:
        payload["tamps_vector"] = result.tamps_vector
    if result.lamps_vector is not None:
        payload["lamps_vector"] = result.lamps_vector

    np.savez(npz_path, **payload)
    if npz_path.endswith(".npz"):
        try:
            save_hast_state_h5(
                npz_path[:-4] + ".h5",
                result=result,
                reference_state_path=result.reference_state_path,
                reference_state_payload=result.reference_state_payload,
            )
        except Exception as exc:
            logger.warning("Failed to save CCSDT HDF5 sidecar for %s: %s", npz_path, exc)
