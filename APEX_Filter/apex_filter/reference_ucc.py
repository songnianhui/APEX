"""Reference-state UCC on the active-space Hamiltonian defined by FCIDUMP.

This module extends the ``reference_uhf`` active-space route so that UCCSD and
UCCSD(T) operate on the same FCIDUMP Hamiltonian and the same reference orbitals
saved from step 3. This matches the Chan-2026 modeling level more closely than
the legacy full-molecule geometry+basis compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from .hdf5_state_io import load_uhf_state_h5
from .post_scf_observables import analyze_active_space_spin_observables
from .reference_uhf import build_fake_mol, build_reference_uhf_solver


@dataclass
class ReferenceUCCResult:
    """Result of an active-space UCCSD or UCCSD(T) calculation."""

    method: str
    energy: float
    correlation_energy: float
    converged: bool
    s_squared: float
    t1_norm: float
    uhf_energy: float
    ccsd_total: float
    ccsd_corr: float
    et_correction: float | None = None
    ccsd_t_total: float | None = None
    iterations: int | None = None
    t1: tuple | None = None
    t2: tuple | None = None
    two_s: float | None = None
    two_sz_fe1: float | None = None
    two_sz_fe2: float | None = None
    post_scf_observables: dict | None = None


def _load_reference_state_payload(uhf_npz_path: str) -> dict:
    if uhf_npz_path.endswith(".h5"):
        return load_uhf_state_h5(uhf_npz_path)
    npz = np.load(uhf_npz_path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_reference_mf_from_npz(fcidump_data, uhf_npz_path: str):
    """Rebuild a UHF object on the FCIDUMP Hamiltonian from saved step3 data."""
    data = _load_reference_state_payload(uhf_npz_path)

    mo_occ = (data["mo_occ_a"], data["mo_occ_b"])
    ms2 = int(round(float(np.sum(mo_occ[0]) - np.sum(mo_occ[1]))))
    mol = build_fake_mol(
        fcidump_data.norb,
        fcidump_data.nelec,
        ms2,
        ecore=fcidump_data.ecore,
    )
    mf = build_reference_uhf_solver(fcidump_data, mol)
    mf.mo_coeff = (data["mo_coeff_a"], data["mo_coeff_b"])
    mf.mo_occ = mo_occ
    mf.mo_energy = (data["mo_energy_a"], data["mo_energy_b"])
    mf.e_tot = float(data["energy"])
    mf.converged = bool(data["converged"])
    return mf


def run_reference_ucc(
    fcidump_data,
    uhf_npz_path: str,
    *,
    run_triples: bool = False,
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    diis_space: int = 12,
    observable_inputs: dict | None = None,
):
    """Run active-space UCCSD or UCCSD(T) from a saved reference-UHF state."""
    from pyscf import cc

    mf = load_reference_mf_from_npz(fcidump_data, uhf_npz_path)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle
    mycc.diis_space = diis_space
    mycc.direct = False
    mycc.kernel()

    nocc_a = int(np.sum(mf.mo_occ[0] > 0))
    nocc_b = int(np.sum(mf.mo_occ[1] > 0))
    t1_norm_a = np.linalg.norm(mycc.t1[0]) / np.sqrt(nocc_a) if nocc_a > 0 else 0.0
    t1_norm_b = np.linalg.norm(mycc.t1[1]) / np.sqrt(nocc_b) if nocc_b > 0 else 0.0
    t1_norm = max(t1_norm_a, t1_norm_b)
    s_squared = float(mycc.spin_square()[0])

    ccsd_total = float(mycc.e_tot)
    ccsd_corr = float(mycc.e_corr)
    method = "UCCSD"
    energy = ccsd_total
    corr_energy = ccsd_corr
    et_correction = None
    ccsd_t_total = None

    if run_triples:
        et_correction = float(mycc.ccsd_t())
        ccsd_t_total = ccsd_total + et_correction
        method = "UCCSD(T)"
        energy = ccsd_t_total
        corr_energy = energy - float(mf.e_tot)

    two_s = None
    two_sz_fe1 = None
    two_sz_fe2 = None
    post_scf_observables = None
    if observable_inputs:
        state_payload = _load_reference_state_payload(uhf_npz_path)
        active_indices = state_payload.get("active_indices")
        if active_indices is None:
            raise ValueError("Active-space mapping missing from step3 reference state; rerun step3 with HDF5 output.")

        if run_triples:
            from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm
            from pyscf.fci import spin_op

            _conv_l, l1, l2 = uccsd_t_lambda.kernel(mycc, tol=conv_tol)
            dm1a_mo, dm1b_mo = uccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, ao_repr=False)
            dm2aa, dm2ab, dm2bb = uccsd_t_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
            s_squared = float(
                spin_op.spin_square_general(
                    dm1a_mo,
                    dm1b_mo,
                    dm2aa,
                    dm2ab,
                    dm2bb,
                    mycc.mo_coeff,
                    mf.get_ovlp(),
                )[0]
            )
        else:
            dm1a_mo, dm1b_mo = mycc.make_rdm1(ao_repr=False)

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
            active_indices=np.asarray(active_indices, dtype=int),
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

    return ReferenceUCCResult(
        method=method,
        energy=float(energy),
        correlation_energy=float(corr_energy),
        converged=bool(mycc.converged),
        s_squared=s_squared,
        t1_norm=float(t1_norm),
        uhf_energy=float(mf.e_tot),
        ccsd_total=ccsd_total,
        ccsd_corr=ccsd_corr,
        et_correction=et_correction,
        ccsd_t_total=ccsd_t_total,
        iterations=int(mycc.cycles) if mycc.cycles is not None else None,
        t1=mycc.t1,
        t2=mycc.t2,
        two_s=two_s,
        two_sz_fe1=two_sz_fe1,
        two_sz_fe2=two_sz_fe2,
        post_scf_observables=post_scf_observables,
    )


def save_reference_ucc_result(result: ReferenceUCCResult, npz_path: str):
    """Save active-space UCC results in the same NPZ schema used by the parser."""
    payload = {
        "uhf_energy": result.uhf_energy,
        "ccsd_corr": result.ccsd_corr,
        "ccsd_total": result.ccsd_total,
        "ccsd_converged": result.converged,
        "ccsd_iterations": result.iterations if result.iterations is not None else 0,
        "spin_sq": result.s_squared,
        "t1_norm": result.t1_norm,
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
    if result.et_correction is not None:
        payload["et_correction"] = result.et_correction
        payload["ccsd_t_total"] = result.ccsd_t_total

    if result.t1 is not None:
        payload["t1_a"] = result.t1[0]
        payload["t1_b"] = result.t1[1]
    if result.t2 is not None:
        payload["t2_aaaa"] = result.t2[0]
        payload["t2_aabb"] = result.t2[1]
        payload["t2_bbbb"] = result.t2[2]

    np.savez(npz_path, **payload)
