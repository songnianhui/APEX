"""UCC screening steps for the interactive APEX_Filter pipeline."""

import json
import logging
import os
from pathlib import Path

import yaml

from shared.apex_cas_provenance import load_apex_cas_provenance
from shared.artifact_paths import resolve_cluster_info_path, resolve_structure_path
from .pick import apply_pick, parse_pick_arg
from .reference_hast_ucc import run_reference_hast_ucc, save_reference_hast_result
from .reference_ucc import run_reference_ucc, save_reference_ucc_result
from .selection_guidance import attach_display_labels, build_display_label_map, write_selection_artifacts
from .session import SessionManager

logger = logging.getLogger(__name__)

_CCSD_DEFAULTS = {
    "code": "pyscf",
    "basis_set": "cc-pVDZ",
}

_CCSD_T_DEFAULTS = {
    "code": "pyscf",
    "basis_set": "cc-pVDZ",
    "n_final": 5,
}

_CCSDT_DEFAULTS = {
    "code": "hast_ucc",
    "basis_set": "cc-pVDZ",
    "n_final": 5,
    "conv_tol": 1e-8,
    "residual_tol": 1e-6,
    "max_cycle": 2000,
    "lambda_max_cycle": 500,
    "diis_space": 6,
    "diis_start_cycle": 0,
    "iterative_damping": 1.0,
    "level_shift": 0.0,
    "newton_krylov": False,
}


def _preferred_step3_state_path(uhf_dir: str, safe_label: str) -> str:
    h5_path = os.path.join(uhf_dir, f"{safe_label}_uhf.h5")
    if os.path.isfile(h5_path):
        return h5_path
    return os.path.join(uhf_dir, f"{safe_label}_uhf.npz")


def _resolve_case_dir_from_fcidump_path(fcidump_path: str) -> str:
    return str(Path(fcidump_path).resolve().parents[2])


def _resolve_cas_settings_path(case_dir: str) -> str | None:
    inputs_dir = os.path.join(case_dir, "inputs")
    if not os.path.isdir(inputs_dir):
        return None
    matches = [
        os.path.join(inputs_dir, name)
        for name in sorted(os.listdir(inputs_dir))
        if name.endswith("_cas_settings.yaml") or name == "cas_settings.yaml"
    ]
    if len(matches) == 1:
        return os.path.abspath(matches[0])
    return None


def _resolve_cas_data_h5_path(case_dir: str) -> str | None:
    provenance = load_apex_cas_provenance(case_dir)
    stem = provenance.get("stem", "")
    if not stem:
        return None
    path = os.path.join(case_dir, "outputs", "orbitals", f"{stem}_cas_data.h5")
    return os.path.abspath(path) if os.path.isfile(path) else None


def _build_ucc_observable_inputs(state: dict, cfg) -> dict | None:
    config_path = state.get("config_path")
    if not config_path or not os.path.isfile(config_path):
        return None
    config_raw = yaml.safe_load(Path(config_path).read_text()) or {}
    case_dir = _resolve_case_dir_from_fcidump_path(state["fcidump_path"])
    config_dir = os.path.dirname(os.path.abspath(config_path))
    xyz_path = resolve_structure_path(config_raw, case_dir)
    cluster_info_path = resolve_cluster_info_path(config_raw, case_dir, config_dir)
    cas_settings_path = _resolve_cas_settings_path(case_dir)
    cas_data_h5_path = _resolve_cas_data_h5_path(case_dir)
    if not all([xyz_path, cluster_info_path, cas_settings_path, cas_data_h5_path]):
        return None
    return {
        "xyz_path": xyz_path,
        "cluster_info_path": cluster_info_path,
        "cas_settings_path": cas_settings_path,
        "cas_data_h5_path": cas_data_h5_path,
        "chan_benchmark_json": config_raw.get("chan_benchmark_json"),
        "label": cfg.label,
        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
    }


def _run_active_space_ucc_batch(
    selected_configs,
    *,
    fcidump_data,
    state,
    uhf_dir,
    level_dir,
    display_label_map=None,
    run_triples=False,
):
    """Run active-space UCC directly on the FCIDUMP Hamiltonian."""
    os.makedirs(level_dir, exist_ok=True)
    results = []
    suffix = "_ccsd_t_results.npz" if run_triples else "_ccsd_results.npz"
    method_label = "CCSD(T)" if run_triples else "CCSD"

    for idx, cfg in enumerate(selected_configs, 1):
        label = cfg.label.replace("|", "_").replace(" ", "_")
        uhf_npz = _preferred_step3_state_path(uhf_dir, label)
        result_npz = os.path.join(level_dir, f"{label}{suffix}")
        result_json = os.path.join(level_dir, f"{label}_post_scf_observables.json")
        observable_inputs = _build_ucc_observable_inputs(state, cfg)
        if observable_inputs is not None:
            observable_inputs["theory"] = "UCCSD(T)" if run_triples else "UCCSD"

        print(f"  [{idx}/{len(selected_configs)}] {cfg.label} ... ", end="", flush=True)
        try:
            result = run_reference_ucc(
                fcidump_data,
                uhf_npz,
                run_triples=run_triples,
                observable_inputs=observable_inputs,
            )
            save_reference_ucc_result(result, result_npz)
            if result.post_scf_observables is not None:
                Path(result_json).write_text(
                    json.dumps(result.post_scf_observables, indent=2, ensure_ascii=False) + "\n"
                )
            status = "OK" if result.converged else "NOT CONVERGED"
            print(f"{status}  E={result.energy:.10f}  <S^2>={result.s_squared:.4f}")
            results.append(
                {
                    "label": cfg.label,
                    "display_label": (display_label_map or {}).get(cfg.label, cfg.label),
                    "method": result.method,
                    "energy": result.energy,
                    "correlation_energy": result.correlation_energy,
                    "converged": result.converged,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "s_squared": result.s_squared,
                    "two_s": result.two_s,
                    "two_sz_fe1": result.two_sz_fe1,
                    "two_sz_fe2": result.two_sz_fe2,
                }
            )
        except Exception as exc:
            logger.warning("%s failed for %s: %s", method_label, cfg.label, exc)
            print(f"FAILED: {exc}")
            results.append(
                {
                    "label": cfg.label,
                    "display_label": (display_label_map or {}).get(cfg.label, cfg.label),
                    "method": f"U{method_label}",
                    "energy": None,
                    "correlation_energy": None,
                    "converged": False,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "s_squared": None,
                    "two_s": None,
                    "two_sz_fe1": None,
                    "two_sz_fe2": None,
                }
            )

    results.sort(key=lambda r: r.get("energy") or float("inf"))
    return results


def _select_configs_for_step(configs, selected_labels):
    """Resolve picked labels to configuration objects."""
    config_map = {c.label: c for c in configs}
    selected_configs = []
    for lbl in selected_labels:
        if lbl in config_map:
            selected_configs.append(config_map[lbl])
        else:
            logger.warning("Config '%s' not found in enumeration, skipping", lbl)
    return selected_configs


def step_ccsd(session_dir: str, *, pick: str = "all", code: str = "pyscf", basis_set: str = "cc-pVDZ"):
    """Run CCSD for selected configurations."""
    sm = SessionManager(session_dir)
    sm.require_previous("step4_ccsd", "step3_uhf")
    controls = sm.resolve_method_controls(
        "ccsd",
        _CCSD_DEFAULTS,
        {"code": code, "basis_set": basis_set},
    )
    code = controls["code"]
    basis_set = controls["basis_set"]
    if code.lower() != "pyscf":
        raise ValueError("step_ccsd supports only code='pyscf' on the active-space FCIDUMP route")

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    uhf_summary = sm.load_uhf_summary()
    display_label_map = build_display_label_map(uhf_summary)

    fcid = state["fcidump_data"]
    configs = enum_data["configs"]

    pick_spec = parse_pick_arg(pick)
    selected_labels = apply_pick(pick_spec, uhf_summary)

    print("=" * 60)
    print(f"Step 4: CCSD ({len(selected_labels)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Code: {code}")
    if basis_set != "cc-pVDZ":
        print(f"  NOTE: ignoring --basis-set={basis_set} on the active-space FCIDUMP route")
    print("  Route: active-space Hamiltonian via FCIDUMP + saved step3 reference UHF")

    selected_configs = _select_configs_for_step(configs, selected_labels)

    sm.save_ccsd_picked(selected_labels)

    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_ccsd_summary([])
        return

    level_dir = sm.ccsd_scripts_dir
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    results = _run_active_space_ucc_batch(
        selected_configs,
        fcidump_data=fcid,
        state=state,
        uhf_dir=uhf_dir,
        level_dir=level_dir,
        display_label_map=display_label_map,
        run_triples=False,
    )
    attach_display_labels(results, uhf_summary)
    n_converged = sum(1 for r in results if r.get("converged"))
    print(f"  Parsed {len(results)} results ({n_converged} converged)")
    results = sm.rebuild_ccsd_summary(configs, uhf_summary, current_results=results)
    attach_display_labels(results, uhf_summary)
    sm.save_ccsd_summary(results)
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step4_ccsd"),
        step_name="Step 4 CCSD",
        next_step_name="ccsd-t",
        summary=results,
        keep_default="1",
    )
    print(f"Step 4 complete. {n_converged} CCSD results saved.")


def step_ccsd_t(
    session_dir: str,
    *,
    pick: str = "all",
    code: str = "pyscf",
    basis_set: str = "cc-pVDZ",
    n_final: int = 5,
):
    """Run CCSD(T) for selected configurations and produce final ranking."""
    sm = SessionManager(session_dir)
    sm.require_previous("step5_ccsd_t", "step4_ccsd")
    controls = sm.resolve_method_controls(
        "ccsd_t",
        _CCSD_T_DEFAULTS,
        {"code": code, "basis_set": basis_set, "n_final": n_final},
    )
    code = controls["code"]
    basis_set = controls["basis_set"]
    n_final = controls["n_final"]
    if code.lower() != "pyscf":
        raise ValueError("step_ccsd_t supports only code='pyscf' on the active-space FCIDUMP route")

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    ccsd_summary = sm.load_ccsd_summary()
    display_label_map = build_display_label_map(ccsd_summary)

    fcid = state["fcidump_data"]
    configs = enum_data["configs"]

    pick_spec = parse_pick_arg(pick)
    selected_labels = apply_pick(pick_spec, ccsd_summary)

    print("=" * 60)
    print(f"Step 5: CCSD(T) ({len(selected_labels)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Code: {code}")
    print(f"  n_final: {n_final}")
    if basis_set != "cc-pVDZ":
        print(f"  NOTE: ignoring --basis-set={basis_set} on the active-space FCIDUMP route")
    print("  Route: active-space Hamiltonian via FCIDUMP + saved step3 reference UHF")

    selected_configs = _select_configs_for_step(configs, selected_labels)

    sm.save_ccsd_t_picked(selected_labels)

    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_ccsd_t_summary([])
        return

    level_dir = sm.ccsd_t_scripts_dir
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    results = _run_active_space_ucc_batch(
        selected_configs,
        fcidump_data=fcid,
        state=state,
        uhf_dir=uhf_dir,
        level_dir=level_dir,
        display_label_map=display_label_map,
        run_triples=True,
    )
    attach_display_labels(results, ccsd_summary)
    n_converged = sum(1 for r in results if r.get("converged"))

    final = results[:n_final]

    print(f"\n  Parsed {len(results)} results ({n_converged} converged)")
    print("\n" + "=" * 70)
    print("Final ranking (CCSD(T))")
    print("=" * 70)
    for rank, r in enumerate(final, 1):
        conv_str = "OK" if r.get("converged") else "NOT CONV"
        e_str = f"{r['energy']:.12f}" if r.get("energy") is not None else "N/A"
        print(f"  #{rank:3d}  E = {e_str}  [{conv_str}]  {r.get('display_label', r['label'])}")
    print("=" * 70)

    sm.save_ccsd_t_summary(results)
    write_selection_artifacts(
        sm._step_dir("step5_ccsd_t"),
        step_name="Step 5 CCSD(T)",
        next_step_name="ccsdt",
        summary=results,
        keep_default="1",
    )
    print(f"\nStep 5 complete. Final ranking ({len(final)} configs) saved.")


def step_ccsdt(
    session_dir: str,
    *,
    pick: str = "all",
    code: str = "hast_ucc",
    basis_set: str = "cc-pVDZ",
    n_final: int = 5,
    conv_tol: float = 1e-8,
    residual_tol: float = 1e-6,
    max_cycle: int = 2000,
    lambda_max_cycle: int = 500,
    diis_space: int = 6,
    diis_start_cycle: int = 0,
    iterative_damping: float = 1.0,
    level_shift: float = 0.0,
    newton_krylov: bool = False,
):
    """Run HAST-UCC CCSDT for selected configurations on the FCIDUMP Hamiltonian."""
    sm = SessionManager(session_dir)
    sm.require_previous("step6_ccsdt", "step5_ccsd_t")
    controls = sm.resolve_method_controls(
        "ccsdt",
        _CCSDT_DEFAULTS,
        {
            "code": code,
            "basis_set": basis_set,
            "n_final": n_final,
            "conv_tol": conv_tol,
            "residual_tol": residual_tol,
            "max_cycle": max_cycle,
            "lambda_max_cycle": lambda_max_cycle,
            "diis_space": diis_space,
            "diis_start_cycle": diis_start_cycle,
            "iterative_damping": iterative_damping,
            "level_shift": level_shift,
            "newton_krylov": newton_krylov,
        },
    )
    code = controls["code"]
    basis_set = controls["basis_set"]
    n_final = controls["n_final"]
    conv_tol = controls["conv_tol"]
    residual_tol = controls["residual_tol"]
    max_cycle = controls["max_cycle"]
    lambda_max_cycle = controls["lambda_max_cycle"]
    diis_space = controls["diis_space"]
    diis_start_cycle = controls["diis_start_cycle"]
    iterative_damping = controls["iterative_damping"]
    level_shift = controls["level_shift"]
    newton_krylov = controls["newton_krylov"]
    if code.lower() != "hast_ucc":
        raise ValueError("step_ccsdt currently supports only code='hast_ucc'")

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    ccsd_t_summary = sm.load_ccsd_t_summary()
    display_label_map = build_display_label_map(ccsd_t_summary)

    fcid = state["fcidump_data"]
    configs = enum_data["configs"]

    pick_spec = parse_pick_arg(pick)
    selected_labels = apply_pick(pick_spec, ccsd_t_summary)

    print("=" * 60)
    print(f"Step 6: CCSDT ({len(selected_labels)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Code: {code}")
    print(f"  conv_tol = {conv_tol:g}")
    print(f"  residual_tol = {residual_tol:g}")
    print(f"  max_cycle = {max_cycle}")
    print(f"  lambda_max_cycle = {lambda_max_cycle}")
    print(f"  diis_space = {diis_space}")
    print(f"  diis_start_cycle = {diis_start_cycle}")
    print(f"  iterative_damping = {iterative_damping:g}")
    print(f"  level_shift = {level_shift:g}")
    print(f"  newton_krylov = {newton_krylov}")
    if basis_set != "cc-pVDZ":
        print(f"  NOTE: ignoring --basis-set={basis_set} on the active-space FCIDUMP route")
    print("  Route: active-space Hamiltonian via FCIDUMP + saved step3 reference UHF")

    selected_configs = _select_configs_for_step(configs, selected_labels)
    sm.save_ccsdt_picked(selected_labels)

    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_ccsdt_summary([])
        return

    level_dir = sm.ccsdt_scripts_dir
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    os.makedirs(level_dir, exist_ok=True)
    results = []
    for idx, cfg in enumerate(selected_configs, 1):
        label = cfg.label.replace("|", "_").replace(" ", "_")
        uhf_npz = _preferred_step3_state_path(uhf_dir, label)
        result_npz = os.path.join(level_dir, f"{label}_ccsdt_results.npz")
        result_json = os.path.join(level_dir, f"{label}_post_scf_observables.json")
        observable_inputs = _build_ucc_observable_inputs(state, cfg)
        if observable_inputs is not None:
            observable_inputs["theory"] = "UCCSDT"
        print(f"  [{idx}/{len(selected_configs)}] {cfg.label} ... ", end="", flush=True)
        try:
            result = run_reference_hast_ucc(
                fcid,
                uhf_npz,
                t_order=3,
                conv_tol=conv_tol,
                residual_tol=residual_tol,
                max_cycle=max_cycle,
                lambda_max_cycle=lambda_max_cycle,
                diis_space=diis_space,
                diis_start_cycle=diis_start_cycle,
                iterative_damping=iterative_damping,
                level_shift=level_shift,
                newton_krylov=newton_krylov,
                observable_inputs=observable_inputs,
            )
            save_reference_hast_result(result, result_npz)
            if result.post_scf_observables is not None:
                Path(result_json).write_text(
                    json.dumps(result.post_scf_observables, indent=2, ensure_ascii=False) + "\n"
                )
            status = "OK" if result.converged else "NOT CONVERGED"
            if result.observables_complete:
                print(f"{status}  E={result.energy:.10f}  <S^2>={result.s_squared:.4f}")
            elif result.observable_error:
                print(f"{status}  E={result.energy:.10f}  [OBS INCOMPLETE: {result.observable_error}]")
            else:
                print(f"{status}  E={result.energy:.10f}")
            results.append(
                {
                    "label": cfg.label,
                    "display_label": display_label_map.get(cfg.label, cfg.label),
                    "method": result.method,
                    "energy": result.energy,
                    "correlation_energy": result.correlation_energy,
                    "converged": result.converged,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "s_squared": result.s_squared,
                    "two_s": result.two_s,
                    "two_sz_fe1": result.two_sz_fe1,
                    "two_sz_fe2": result.two_sz_fe2,
                    "observables_complete": result.observables_complete,
                    "lambda_converged": result.lambda_converged,
                    "observable_error": result.observable_error,
                }
            )
        except Exception as exc:
            logger.warning("CCSDT failed for %s: %s", cfg.label, exc)
            print(f"FAILED: {exc}")
            results.append(
                {
                    "label": cfg.label,
                    "display_label": display_label_map.get(cfg.label, cfg.label),
                    "method": "UCCSDT",
                    "energy": None,
                    "correlation_energy": None,
                    "converged": False,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "s_squared": None,
                    "two_s": None,
                    "two_sz_fe1": None,
                    "two_sz_fe2": None,
                }
            )

    results.sort(key=lambda r: r.get("energy") or float("inf"))
    attach_display_labels(results, ccsd_t_summary)
    n_converged = sum(1 for r in results if r.get("converged"))
    final = results[:n_final]

    print(f"\n  Parsed {len(results)} results ({n_converged} converged)")
    print("\n" + "=" * 70)
    print("Final ranking (CCSDT)")
    print("=" * 70)
    for rank, r in enumerate(final, 1):
        conv_str = "OK" if r.get("converged") else "NOT CONV"
        e_str = f"{r['energy']:.12f}" if r.get("energy") is not None else "N/A"
        print(f"  #{rank:3d}  E = {e_str}  [{conv_str}]  {r.get('display_label', r['label'])}")
    print("=" * 70)

    sm.save_ccsdt_summary(results)
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step6_ccsdt"),
        step_name="Step 6 CCSDT",
        next_step_name="dmrg-basis",
        summary=results,
        keep_default="1",
    )
    print(f"\nStep 6 complete. Final ranking ({len(final)} configs) saved.")
