"""Step 8-9 DMRG solve/extrapolation entrypoints for the staged APEX_Filter workflow."""

import os

from ._dmrg_summary import (
    _DMRG_SOURCE_MODE_CONVERGED_ONLY,
    _DMRG_SOURCE_MODE_UNCONVERGED_FALLBACK,
)
from .energy_extrapolation import dmrg_d_extrapolation as _dmrg_d_extrapolation
from ._case_artifacts import _preferred_step3_state_path, _preferred_step7_basis_path
from .pick import _apply_pick, _parse_pick_arg
from .reference_dmrg import _save_reference_dmrg_result, run_reference_dmrg as _run_reference_dmrg
from .selection_guidance import _attach_display_labels, _build_display_label_map
from .session import SessionManager as _SessionManager
from ._step_selection_artifacts import _cleanup_step_selection_artifacts
from shared.formatting import shell_safe_artifact_token as _shell_safe_artifact_token

_PRE_DMRG_PICK_MODES = {"all", "labels", "file"}
_DMRG_DEFAULTS = {
    "backend": "pyblock2_sz",
    "basis_mode": "step7_paired",
    "schedule_mode": "workflow",
    "bond_dims": [500, 1000],
    "n_sweeps": 8,
    "convergence_tol": 1e-8,
    "n_threads": 4,
    "stack_mem": 2 * 1024**3,
    "twosite_to_onesite": None,
    "dav_max_iter": None,
    "dav_def_max_size": None,
    "dav_rel_conv_thrd": None,
    "dav_type": None,
}

def step_dmrg(
    session_dir: str,
    *,
    pick: str = "all",
    backend: str = "pyblock2_sz",
    basis_mode: str = "step7_paired",
    schedule_mode: str = "workflow",
    bond_dims: list[int] | None = None,
    n_sweeps: int = 8,
    convergence_tol: float = 1e-8,
    n_threads: int = 4,
    stack_mem: int = 2 * 1024**3,
    twosite_to_onesite: int | None = None,
    dav_max_iter: int | None = None,
    dav_def_max_size: int | None = None,
    dav_rel_conv_thrd: float | None = None,
    dav_type: str | None = None,
):
    """Run active-space DMRG for selected configs and bond dimensions."""
    sm = _SessionManager(session_dir)
    sm.require_previous("step8_dmrg", "step7_dmrg_basis")
    controls = sm.resolve_method_controls(
        "dmrg",
        _DMRG_DEFAULTS,
        {
            "backend": backend,
            "basis_mode": basis_mode,
            "schedule_mode": schedule_mode,
            "bond_dims": bond_dims,
            "n_sweeps": n_sweeps,
            "convergence_tol": convergence_tol,
            "n_threads": n_threads,
            "stack_mem": stack_mem,
            "twosite_to_onesite": twosite_to_onesite,
            "dav_max_iter": dav_max_iter,
            "dav_def_max_size": dav_def_max_size,
            "dav_rel_conv_thrd": dav_rel_conv_thrd,
            "dav_type": dav_type,
        },
    )
    backend = controls["backend"]
    basis_mode = controls["basis_mode"]
    schedule_mode = controls["schedule_mode"]
    bond_dims = controls["bond_dims"]
    n_sweeps = controls["n_sweeps"]
    convergence_tol = controls["convergence_tol"]
    n_threads = controls["n_threads"]
    stack_mem = controls["stack_mem"]
    twosite_to_onesite = controls["twosite_to_onesite"]
    dav_max_iter = controls["dav_max_iter"]
    dav_def_max_size = controls["dav_def_max_size"]
    dav_rel_conv_thrd = controls["dav_rel_conv_thrd"]
    dav_type = controls["dav_type"]

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    basis_summary = sm.load_step_summary("step7_dmrg_basis", "dmrg_basis_summary.json")
    display_label_map = _build_display_label_map(basis_summary)

    pick_spec = _parse_pick_arg(pick)
    if pick_spec["mode"] not in _PRE_DMRG_PICK_MODES:
        raise ValueError(
            "Step 8 runs before DMRG energies are available, so only pick modes "
            "'all', 'labels', and 'file' are supported."
        )

    selected_labels = _apply_pick(pick_spec, basis_summary)
    config_map = {cfg.label: cfg for cfg in enum_data["configs"]}
    selected_configs = [config_map[label] for label in selected_labels if label in config_map]

    print("=" * 60)
    print(f"Step 8: DMRG solve ({len(selected_configs)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Backend      : {backend}")
    print(f"  Basis mode   : {basis_mode}")
    print(f"  Schedule mode: {schedule_mode}")
    print(f"  Bond dims    : {bond_dims}")

    sm.save_step_picked("step8_dmrg", selected_labels)
    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_step_summary("step8_dmrg", "dmrg_summary.json", [])
        return

    results_dir = sm.step_artifact_dir("step8_dmrg", "results")
    basis_dir = sm.step_artifact_dir("step7_dmrg_basis", "results")
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary = []
    for cfg in selected_configs:
        safe_label = cfg.label.replace("|", "_").replace(" ", "_")
        uhf_npz = _preferred_step3_state_path(uhf_dir, safe_label)
        basis_npz = _preferred_step7_basis_path(basis_dir, safe_label)

        for bond_dim in bond_dims:
            artifact_token = _shell_safe_artifact_token(cfg.label)
            out_npz = os.path.join(results_dir, f"{artifact_token}_M{bond_dim}_dmrg.npz")
            log_path = os.path.join(results_dir, f"{artifact_token}_M{bond_dim}_dmrg.log")
            scratch = os.path.join(results_dir, f"{artifact_token}_M{bond_dim}_scratch")
            print(f"  {cfg.label} @ M={bond_dim} ... ", end="", flush=True)
            try:
                result = _run_reference_dmrg(
                    state["fcidump_data"],
                    uhf_npz,
                    basis_npz,
                    fcidump_path=state["fcidump_path"],
                    backend=backend,
                    basis_mode=basis_mode,
                    bond_dim=bond_dim,
                    n_sweeps=n_sweeps,
                    convergence_tol=convergence_tol,
                    schedule_mode=schedule_mode,
                    n_threads=n_threads,
                    stack_mem=stack_mem,
                    twosite_to_onesite=twosite_to_onesite,
                    dav_max_iter=dav_max_iter,
                    dav_def_max_size=dav_def_max_size,
                    dav_rel_conv_thrd=dav_rel_conv_thrd,
                    dav_type=dav_type,
                    scratch=scratch,
                    log_path=log_path,
                )
                _save_reference_dmrg_result(
                    result,
                    out_npz,
                    label=cfg.label,
                    family=cfg.spin_isomer.family if cfg.spin_isomer else "",
                    settings_payload=sm._build_step_settings_payload(
                        None,
                        theory="DMRG",
                        backend=backend,
                        basis_mode=basis_mode,
                        schedule_mode=schedule_mode,
                        bond_dim=bond_dim,
                        n_sweeps=n_sweeps,
                        convergence_tol=convergence_tol,
                        n_threads=n_threads,
                        stack_mem=stack_mem,
                        twosite_to_onesite=twosite_to_onesite,
                        dav_max_iter=dav_max_iter,
                        dav_def_max_size=dav_def_max_size,
                        dav_rel_conv_thrd=dav_rel_conv_thrd,
                        dav_type=dav_type,
                    ),
                )
                wall_time = getattr(result, "wall_time_s", None)
                wall_time_str = f"{wall_time:.2f}s" if wall_time is not None else "N/A"
                print(
                    f"converged={result.converged}  "
                    f"E={result.energy:.10f}  "
                    f"Wall_time={wall_time_str}"
                )
                summary.append(
                    {
                        "label": cfg.label,
                        "display_label": display_label_map.get(cfg.label, cfg.label),
                        "bond_dim": bond_dim,
                        "backend": backend,
                        "basis_mode": basis_mode,
                        "schedule_mode": schedule_mode,
                        "energy": result.energy,
                        "converged": result.converged,
                        "s_squared": result.s_squared,
                        "wall_time_s": wall_time,
                        "result_path": os.path.abspath(out_npz),
                        "log_path": getattr(result, "log_path", None),
                        "twosite_to_onesite": getattr(result, "twosite_to_onesite", None),
                        "dav_max_iter": getattr(result, "dav_max_iter", None),
                        "dav_def_max_size": getattr(result, "dav_def_max_size", None),
                        "dav_rel_conv_thrd": getattr(result, "dav_rel_conv_thrd", None),
                        "dav_type": getattr(result, "dav_type", None),
                        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    }
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                summary.append(
                    {
                        "label": cfg.label,
                        "display_label": display_label_map.get(cfg.label, cfg.label),
                        "bond_dim": bond_dim,
                        "backend": backend,
                        "basis_mode": basis_mode,
                        "schedule_mode": schedule_mode,
                        "energy": None,
                        "converged": False,
                        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    }
                )

    _attach_display_labels(summary, basis_summary)
    step_dir = os.path.join(sm.session_dir, "step8_dmrg")
    sm.save_step_summary("step8_dmrg", "dmrg_summary.json", summary)
    _cleanup_step_selection_artifacts(step_dir)
    n_total = len(summary)
    n_conv = sum(1 for row in summary if row.get("converged"))
    print(
        "Step 8 complete. "
        f"Saved {n_total} DMRG results "
        f"({n_conv} converged, {n_total - n_conv} unconverged)."
    )


def step_extrapolate_dmrg(session_dir: str):
    """Extrapolate DMRG energies to infinite bond dimension per config."""
    sm = _SessionManager(session_dir)
    sm.require_previous("step9_extrapolate", "step8_dmrg")

    dmrg_summary = sm.load_step_summary("step8_dmrg", "dmrg_summary.json")
    by_label = {}
    for row in dmrg_summary:
        if row.get("energy") is None:
            continue
        by_label.setdefault(row["label"], []).append(row)

    print("=" * 60)
    print("Step 9: DMRG extrapolation")
    print("=" * 60)

    summary = []
    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda r: r["bond_dim"])
        converged_rows = [row for row in rows if row.get("converged")]
        if converged_rows:
            used_rows = converged_rows
            source_mode = _DMRG_SOURCE_MODE_CONVERGED_ONLY
        elif len(rows) >= 2:
            used_rows = rows
            source_mode = _DMRG_SOURCE_MODE_UNCONVERGED_FALLBACK
        else:
            continue
        bond_dims = [int(r["bond_dim"]) for r in used_rows]
        energies = [float(r["energy"]) for r in used_rows]
        family = next((row.get("family", "") for row in used_rows if row.get("family")), "")
        extrap = _dmrg_d_extrapolation(bond_dims, energies)
        print(
            f"  {label}: E_inf={extrap.energy:.10f} "
            f"from M={bond_dims} "
            f"[{source_mode}]"
        )
        summary.append(
            {
                "label": label,
                "display_label": used_rows[0].get("display_label", label),
                "method": extrap.method,
                "energy": extrap.energy,
                "uncertainty": extrap.uncertainty,
                "bond_dims": bond_dims,
                "energies": energies,
                "family": family,
                "description": extrap.description,
                "source_mode": source_mode,
                "n_points_used": len(used_rows),
                "n_converged_points": len(converged_rows),
                "n_total_points": len(rows),
            }
        )

    _attach_display_labels(summary, dmrg_summary)
    step_dir = os.path.join(sm.session_dir, "step9_extrapolate")
    sm.save_step_summary("step9_extrapolate", "dmrg_extrapolation_summary.json", summary)
    _cleanup_step_selection_artifacts(step_dir)
    print(f"Step 9 complete. {len(summary)} extrapolated energies saved.")
