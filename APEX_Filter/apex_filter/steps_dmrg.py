"""DMRG solve and extrapolation steps for the interactive APEX_Filter pipeline."""

import hashlib
import os
import re
import unicodedata

from .energy_extrapolation import dmrg_d_extrapolation
from .pick import apply_pick, parse_pick_arg
from .reference_dmrg import run_reference_dmrg, save_reference_dmrg_result
from .selection_guidance import attach_display_labels, build_display_label_map
from .session import SessionManager

_PRE_DMKG_PICK_MODES = {"all", "labels", "file"}
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


def _preferred_step3_state_path(uhf_dir: str, safe_label: str) -> str:
    h5_path = os.path.join(uhf_dir, f"{safe_label}_uhf.h5")
    if os.path.isfile(h5_path):
        return h5_path
    return os.path.join(uhf_dir, f"{safe_label}_uhf.npz")


def _preferred_step7_basis_path(basis_dir: str, safe_label: str) -> str:
    h5_path = os.path.join(basis_dir, f"{safe_label}_dmrg_basis.h5")
    if os.path.isfile(h5_path):
        return h5_path
    return os.path.join(basis_dir, f"{safe_label}_dmrg_basis.npz")


def _shell_safe_artifact_token(label: str) -> str:
    """Return an ASCII/shell-safe token for scratch/log/result artifact names."""
    normalized = unicodedata.normalize("NFKD", str(label))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._-")
    prefix = safe or "dmrg_state"
    suffix = hashlib.sha1(str(label).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{suffix}"


def _remove_selection_artifacts(step_dir: str) -> None:
    """Remove stale selection/worklist artifacts for non-pick-driven steps."""
    for stale in (
        "selection_candidates.csv",
        "selection_worklist.csv",
        "selection_guide.md",
        "selection_candidates.json",
        "pick_labels_all.json",
        "pick_labels_template.json",
    ):
        stale_path = os.path.join(step_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)


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
    sm = SessionManager(session_dir)
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
    basis_summary = sm.load_dmrg_basis_summary()
    display_label_map = build_display_label_map(basis_summary)

    pick_spec = parse_pick_arg(pick)
    if pick_spec["mode"] not in _PRE_DMKG_PICK_MODES:
        raise ValueError(
            "Step 8 runs before DMRG energies are available, so only pick modes "
            "'all', 'labels', and 'file' are supported."
        )

    selected_labels = apply_pick(pick_spec, basis_summary)
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

    sm.save_dmrg_picked(selected_labels)
    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_dmrg_summary([])
        return

    results_dir = sm.dmrg_results_dir
    basis_dir = sm.dmrg_basis_results_dir
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
                result = run_reference_dmrg(
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
                save_reference_dmrg_result(result, out_npz)
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

    attach_display_labels(summary, basis_summary)
    step_dir = os.path.join(sm.session_dir, "step8_dmrg")
    sm.save_dmrg_summary(summary)
    _remove_selection_artifacts(step_dir)
    n_total = len(summary)
    n_conv = sum(1 for row in summary if row.get("converged"))
    print(
        "Step 8 complete. "
        f"Saved {n_total} DMRG results "
        f"({n_conv} converged, {n_total - n_conv} unconverged)."
    )


def step_extrapolate_dmrg(session_dir: str):
    """Extrapolate DMRG energies to infinite bond dimension per config."""
    sm = SessionManager(session_dir)
    sm.require_previous("step9_extrapolate", "step8_dmrg")

    dmrg_summary = sm.load_dmrg_summary()
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
            source_mode = "converged_only"
        elif len(rows) >= 2:
            used_rows = rows
            source_mode = "unconverged_fallback"
        else:
            continue
        bond_dims = [int(r["bond_dim"]) for r in used_rows]
        energies = [float(r["energy"]) for r in used_rows]
        family = next((row.get("family", "") for row in used_rows if row.get("family")), "")
        extrap = dmrg_d_extrapolation(bond_dims, energies)
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

    attach_display_labels(summary, dmrg_summary)
    step_dir = os.path.join(sm.session_dir, "step9_extrapolate")
    sm.save_dmrg_extrapolation_summary(summary)
    _remove_selection_artifacts(step_dir)
    print(f"Step 9 complete. {len(summary)} extrapolated energies saved.")
