"""Shared DMRG control helpers for APEX_CAS and APEX_Filter.

This module defines project-level DMRG schedule semantics so that different
backends (PySCF DMRGCI, pyblock2 DMRGDriver) can share the same high-level
control modes.
"""

from __future__ import annotations

from typing import Iterable as _Iterable


def build_dmrg_sweep_schedule(
    *,
    mode: str,
    bond_dim: int,
    convergence_tol: float,
    n_sweeps: int | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Return per-sweep bond dimensions, noises, and thresholds.

    Parameters
    ----------
    mode
        Either ``"workflow"`` or ``"benchmark"``.
    bond_dim
        Target maximum bond dimension.
    convergence_tol
        Final DMRG convergence tolerance used by the caller.
    n_sweeps
        Optional override for total number of sweeps.
    """
    mode_norm = (mode or "workflow").strip().lower()
    if mode_norm not in {"workflow", "benchmark"}:
        raise ValueError(f"Unsupported DMRG mode: {mode}")

    if mode_norm == "benchmark":
        total_sweeps = int(n_sweeps) if n_sweeps is not None else 30
        # Mirrors the effective default PySCF DMRGCI SZ schedule closely:
        # scheduleSweeps [0,4,8,10,12,14,16,18]
        # scheduleMaxMs [200,400,500,500,500,500,500,500]
        # scheduleTols  [1e-4,1e-4,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
        # scheduleNoises[1e-4,1e-4,1e-4,1e-5,1e-6,1e-7,1e-8,0.0]
        stages = [
            {"start": 0, "bond_dim": min(200, bond_dim), "noise": 1e-4, "threshold": 1e-4},
            {"start": 4, "bond_dim": min(400, bond_dim), "noise": 1e-4, "threshold": 1e-4},
            {"start": 8, "bond_dim": bond_dim, "noise": 1e-4, "threshold": 1e-4},
            {"start": 10, "bond_dim": bond_dim, "noise": 1e-5, "threshold": 1e-5},
            {"start": 12, "bond_dim": bond_dim, "noise": 1e-6, "threshold": 1e-6},
            {"start": 14, "bond_dim": bond_dim, "noise": 1e-7, "threshold": 1e-7},
            {"start": 16, "bond_dim": bond_dim, "noise": 1e-8, "threshold": 1e-8},
            {"start": 18, "bond_dim": bond_dim, "noise": 0.0, "threshold": 1e-9},
        ]
        return _expand_stage_schedule(stages, total_sweeps)

    total_sweeps = int(n_sweeps) if n_sweeps is not None else 8
    stages = [
        {"repeats": 2, "bond_dim": min(50, bond_dim), "noise": 1e-4, "threshold": max(convergence_tol * 100, 1e-6)},
        {"repeats": 2, "bond_dim": min(100, bond_dim), "noise": 5e-5, "threshold": max(convergence_tol * 10, 1e-7)},
        {"repeats": 2, "bond_dim": min(200, bond_dim), "noise": 1e-5, "threshold": max(convergence_tol, 1e-8)},
        {"repeats": 2, "bond_dim": min(500, bond_dim), "noise": 1e-6, "threshold": convergence_tol},
    ]
    return _expand_repeat_schedule(stages, total_sweeps, bond_dim, convergence_tol)


def compress_dmrg_schedule_for_dmrgci(
    bond_dims: _Iterable[int],
    noises: _Iterable[float],
    thresholds: _Iterable[float],
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Compress per-sweep arrays into PySCF DMRGCI stage arrays."""
    sweep_list = list(bond_dims)
    noise_list = list(noises)
    thr_list = list(thresholds)
    if not (len(sweep_list) == len(noise_list) == len(thr_list)):
        raise ValueError("DMRG schedule arrays must have equal length")
    if not sweep_list:
        raise ValueError("DMRG schedule arrays must not be empty")

    starts = [0]
    max_ms = [int(sweep_list[0])]
    tol_list = [float(thr_list[0])]
    noise_out = [float(noise_list[0])]

    for idx in range(1, len(sweep_list)):
        triple = (int(sweep_list[idx]), float(noise_list[idx]), float(thr_list[idx]))
        prev = (int(sweep_list[idx - 1]), float(noise_list[idx - 1]), float(thr_list[idx - 1]))
        if triple != prev:
            starts.append(idx)
            max_ms.append(triple[0])
            noise_out.append(triple[1])
            tol_list.append(triple[2])

    return starts, max_ms, tol_list, noise_out


def infer_pyblock2_benchmark_controls(
    *,
    bond_dim: int,
    convergence_tol: float,
) -> dict[str, int | float | str | None]:
    """Return pyblock2 controls aligned to the closest exposed DMRGCI semantics.

    The most important matched control is the 2-site -> 1-site switch point.
    For ``maxM=500, tol=1e-8`` PySCF DMRGCI defaults to ``twodot_to_onedot=22``.
    Davidson-specific knobs do not have direct DMRGCI-level counterparts, so the
    benchmark defaults keep pyblock2's own documented defaults unless the caller
    overrides them explicitly.
    """
    del bond_dim, convergence_tol
    return {
        "twosite_to_onesite": 22,
        "dav_max_iter": 4000,
        "dav_def_max_size": 50,
        "dav_rel_conv_thrd": 0.0,
        "dav_type": None,
    }


def _expand_stage_schedule(
    stages: list[dict[str, float | int]],
    total_sweeps: int,
) -> tuple[list[int], list[float], list[float]]:
    bond_dims: list[int] = []
    noises: list[float] = []
    thresholds: list[float] = []
    stages_sorted = sorted(stages, key=lambda s: int(s["start"]))

    for idx, stage in enumerate(stages_sorted):
        start = int(stage["start"])
        end = total_sweeps
        if idx + 1 < len(stages_sorted):
            end = min(total_sweeps, int(stages_sorted[idx + 1]["start"]))
        if start >= total_sweeps:
            break
        repeats = max(0, end - start)
        for _ in range(repeats):
            bond_dims.append(int(stage["bond_dim"]))
            noises.append(float(stage["noise"]))
            thresholds.append(float(stage["threshold"]))

    if not bond_dims:
        raise ValueError("Expanded DMRG schedule is empty")
    if len(bond_dims) < total_sweeps:
        last_bd = bond_dims[-1]
        last_noise = noises[-1]
        last_thr = thresholds[-1]
        for _ in range(total_sweeps - len(bond_dims)):
            bond_dims.append(last_bd)
            noises.append(last_noise)
            thresholds.append(last_thr)
    return bond_dims[:total_sweeps], noises[:total_sweeps], thresholds[:total_sweeps]


def _expand_repeat_schedule(
    stages: list[dict[str, float | int]],
    total_sweeps: int,
    bond_dim: int,
    convergence_tol: float,
) -> tuple[list[int], list[float], list[float]]:
    bond_dims: list[int] = []
    noises: list[float] = []
    thresholds: list[float] = []
    for stage in stages:
        for _ in range(int(stage["repeats"])):
            bond_dims.append(int(stage["bond_dim"]))
            noises.append(float(stage["noise"]))
            thresholds.append(float(stage["threshold"]))
    while len(bond_dims) < total_sweeps:
        bond_dims.append(int(bond_dim))
        noises.append(1e-7)
        thresholds.append(float(convergence_tol))
    return bond_dims[:total_sweeps], noises[:total_sweeps], thresholds[:total_sweeps]
