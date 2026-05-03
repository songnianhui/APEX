"""Reference DMRG orchestration on the active-space Hamiltonian defined by FCIDUMP."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from .hdf5_state_io import _save_dmrg_state_h5
from shared.dmrg_controls import build_dmrg_sweep_schedule as _build_dmrg_sweep_schedule
from shared.settings_payloads import extend_settings_payload as _extend_settings_payload

@_dataclass
class _ReferenceDMRGResult:
    """Result of an active-space DMRG calculation for one bond dimension."""

    method: str
    energy: float
    correlation_energy: float
    converged: bool
    s_squared: float | None
    uhf_energy: float
    backend: str
    basis_mode: str
    bond_dim: int
    n_sweeps: int
    schedule_mode: str
    bond_dims: list[int]
    noises: list[float]
    thresholds: list[float]
    twosite_to_onesite: int | None = None
    dav_max_iter: int | None = None
    dav_def_max_size: int | None = None
    dav_rel_conv_thrd: float | None = None
    dav_type: str | None = None
    wall_time_s: float | None = None
    log_path: str | None = None
    fcidump_path: str | None = None
    reference_state_path: str | None = None
    basis_state_path: str | None = None
    scratch_dir: str | None = None


_DMRG_ENERGY_RE = re.compile(r"\bE\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)")


def _parse_last_dmrg_energy(text: str) -> float | None:
    matches = _DMRG_ENERGY_RE.findall(text or "")
    if not matches:
        return None
    return float(matches[-1])


def _read_uhf_energy(uhf_npz_path: str) -> float:
    data = np.load(uhf_npz_path, allow_pickle=True)
    try:
        return float(data["energy"])
    finally:
        data.close()


def _ensure_schedule_lists(
    *,
    bond_dim: int,
    n_sweeps: int,
    schedule_mode: str,
    bond_dims: list[int] | None,
    noises: list[float] | None,
    thresholds: list[float] | None,
) -> tuple[list[int], list[float], list[float]]:
    if bond_dims and noises and thresholds:
        return list(map(int, bond_dims)), list(map(float, noises)), list(map(float, thresholds))
    rebuilt_bd, rebuilt_noises, rebuilt_thr = _build_dmrg_sweep_schedule(
        mode=schedule_mode,
        bond_dim=int(bond_dim),
        convergence_tol=1e-8,
        n_sweeps=int(n_sweeps),
    )
    return rebuilt_bd, rebuilt_noises, rebuilt_thr


def run_reference_dmrg(
    fcidump_data,
    uhf_npz_path: str,
    dmrg_basis_npz_path: str,
    *,
    fcidump_path: str,
    backend: str = "pyblock2_sz",
    basis_mode: str = "step7_paired",
    bond_dim: int,
    n_sweeps: int = 8,
    convergence_tol: float = 1e-8,
    schedule_mode: str = "workflow",
    n_threads: int = 4,
    stack_mem: int = 2 * 1024**3,
    twosite_to_onesite: int | None = None,
    dav_max_iter: int | None = None,
    dav_def_max_size: int | None = None,
    dav_rel_conv_thrd: float | None = None,
    dav_type: str | None = None,
    scratch: str = "./scratch_dmrg",
    log_path: str | None = None,
):
    """Run BLOCK2 DMRG in a fresh worker process.

    A dedicated Python subprocess is used to avoid macOS/conda OpenMP runtime
    conflicts between PySCF and pyblock2/block2.
    """
    del fcidump_data  # kept in signature for call-shape symmetry with sibling reference drivers

    os.makedirs(scratch, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="apex_dmrg_", suffix=".json", dir=scratch, delete=False) as fh:
        result_json = fh.name

    cmd = [
        sys.executable,
        "-m",
        "apex_filter.reference_dmrg_worker",
        "--backend",
        str(backend),
        "--fcidump-path",
        os.path.abspath(fcidump_path),
        "--uhf-npz-path",
        os.path.abspath(uhf_npz_path),
        "--dmrg-basis-npz-path",
        os.path.abspath(dmrg_basis_npz_path),
        "--basis-mode",
        str(basis_mode),
        "--bond-dim",
        str(int(bond_dim)),
        "--n-sweeps",
        str(int(n_sweeps)),
        "--convergence-tol",
        str(float(convergence_tol)),
        "--schedule-mode",
        str(schedule_mode),
        "--n-threads",
        str(int(n_threads)),
        "--stack-mem",
        str(int(stack_mem)),
        "--twosite-to-onesite",
        "" if twosite_to_onesite is None else str(int(twosite_to_onesite)),
        "--dav-max-iter",
        "" if dav_max_iter is None else str(int(dav_max_iter)),
        "--dav-def-max-size",
        "" if dav_def_max_size is None else str(int(dav_def_max_size)),
        "--dav-rel-conv-thrd",
        "" if dav_rel_conv_thrd is None else str(float(dav_rel_conv_thrd)),
        "--dav-type",
        "" if dav_type is None else str(dav_type),
        "--scratch",
        os.path.abspath(scratch),
        "--result-json",
        result_json,
    ]
    # Remove empty optional arguments while preserving option/value pairs.
    compact_cmd: list[str] = []
    idx = 0
    while idx < len(cmd):
        token = cmd[idx]
        if token.startswith("--") and idx + 1 < len(cmd) and cmd[idx + 1] == "":
            idx += 2
            continue
        compact_cmd.append(token)
        idx += 1
    cmd = compact_cmd

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    output_text = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output_text)
            if output_text and not output_text.endswith("\n"):
                f.write("\n")
    parsed_energy = _parse_last_dmrg_energy(output_text)
    converged = "ATTENTION: DMRG is not converged" not in output_text

    payload = None
    if os.path.exists(result_json) and os.path.getsize(result_json) > 0:
        with open(result_json, "r") as f:
            payload = json.load(f)
    try:
        os.remove(result_json)
    except OSError:
        pass

    if payload is not None:
        payload.setdefault("backend", str(backend))
        payload.setdefault("basis_mode", str(basis_mode))
        payload.setdefault("twosite_to_onesite", twosite_to_onesite)
        payload.setdefault("dav_max_iter", dav_max_iter)
        payload.setdefault("dav_def_max_size", dav_def_max_size)
        payload.setdefault("dav_rel_conv_thrd", dav_rel_conv_thrd)
        payload.setdefault("dav_type", dav_type)
        payload["converged"] = bool(payload.get("converged", True) and converged)
        if payload.get("energy") is None and parsed_energy is not None:
            payload["energy"] = parsed_energy
        payload["log_path"] = os.path.abspath(log_path) if log_path else None
        payload["fcidump_path"] = os.path.abspath(fcidump_path)
        payload["reference_state_path"] = os.path.abspath(uhf_npz_path)
        payload["basis_state_path"] = os.path.abspath(dmrg_basis_npz_path)
        payload["scratch_dir"] = os.path.abspath(scratch)
        payload["bond_dims"], payload["noises"], payload["thresholds"] = _ensure_schedule_lists(
            bond_dim=payload["bond_dim"],
            n_sweeps=payload["n_sweeps"],
            schedule_mode=payload["schedule_mode"],
            bond_dims=payload.get("bond_dims"),
            noises=payload.get("noises"),
            thresholds=payload.get("thresholds"),
        )
        return _ReferenceDMRGResult(**payload)

    if parsed_energy is not None:
        uhf_energy = _read_uhf_energy(uhf_npz_path)
        bond_dims, noises, thresholds = _ensure_schedule_lists(
            bond_dim=int(bond_dim),
            n_sweeps=int(n_sweeps),
            schedule_mode=str(schedule_mode),
            bond_dims=[],
            noises=[],
            thresholds=[],
        )
        return _ReferenceDMRGResult(
            method="DMRG",
            energy=parsed_energy,
            correlation_energy=parsed_energy - uhf_energy,
            converged=False,
            s_squared=None,
            uhf_energy=uhf_energy,
            backend=str(backend),
            basis_mode=str(basis_mode),
            bond_dim=int(bond_dim),
            n_sweeps=int(n_sweeps),
            schedule_mode=str(schedule_mode),
            bond_dims=bond_dims,
            noises=noises,
            thresholds=thresholds,
            twosite_to_onesite=twosite_to_onesite,
            dav_max_iter=dav_max_iter,
            dav_def_max_size=dav_def_max_size,
            dav_rel_conv_thrd=dav_rel_conv_thrd,
            dav_type=dav_type,
            log_path=os.path.abspath(log_path) if log_path else None,
            fcidump_path=os.path.abspath(fcidump_path),
            reference_state_path=os.path.abspath(uhf_npz_path),
            basis_state_path=os.path.abspath(dmrg_basis_npz_path),
            scratch_dir=os.path.abspath(scratch),
        )

    if proc.returncode != 0:
        raise RuntimeError(output_text or f"DMRG worker failed with exit code {proc.returncode}")
    raise RuntimeError("DMRG worker exited without producing a result payload or parsable energy trace")


def _save_reference_dmrg_result(
    result: _ReferenceDMRGResult,
    npz_path: str,
    *,
    label: str | None = None,
    family: str | None = None,
    settings_payload: dict | None = None,
):
    """Save active-space DMRG results in the canonical NPZ sidecar schema."""
    settings_payload = _extend_settings_payload(
        settings_payload,
        n_sweeps=result.n_sweeps,
        twosite_to_onesite=result.twosite_to_onesite,
        dav_max_iter=result.dav_max_iter,
        dav_def_max_size=result.dav_def_max_size,
        dav_rel_conv_thrd=result.dav_rel_conv_thrd,
        dav_type=result.dav_type,
    )
    payload = {
        "uhf_energy": result.uhf_energy,
        "dmrg_total": result.energy,
        "dmrg_corr": result.correlation_energy,
        "dmrg_converged": result.converged,
        "backend": np.asarray(result.backend, dtype=object),
        "basis_mode": np.asarray(result.basis_mode, dtype=object),
        "bond_dim": result.bond_dim,
        "n_sweeps": result.n_sweeps,
        "schedule_mode": np.asarray(result.schedule_mode, dtype=object),
        "bond_dims": np.asarray(result.bond_dims, dtype=int),
        "dmrg_noises": np.asarray(result.noises, dtype=float),
        "dmrg_thresholds": np.asarray(result.thresholds, dtype=float),
        "twosite_to_onesite": np.asarray(result.twosite_to_onesite, dtype=object),
        "dav_max_iter": np.asarray(result.dav_max_iter, dtype=object),
        "dav_def_max_size": np.asarray(result.dav_def_max_size, dtype=object),
        "dav_rel_conv_thrd": np.asarray(result.dav_rel_conv_thrd, dtype=object),
        "dav_type": np.asarray(result.dav_type, dtype=object),
    }
    if result.s_squared is not None:
        payload["spin_sq"] = result.s_squared
    if npz_path.endswith(".npz"):
        _save_dmrg_state_h5(
            npz_path[:-4] + ".h5",
            result=result,
            label=label,
            family=family,
            reference_state_path=result.reference_state_path,
            basis_state_path=result.basis_state_path,
            scratch_dir=result.scratch_dir,
            settings_payload=settings_payload,
        )
    np.savez(npz_path, **payload)
