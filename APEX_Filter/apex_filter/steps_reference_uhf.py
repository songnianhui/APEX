"""Step 3 reference-UHF entrypoints for the staged APEX_Filter workflow."""

import json
import logging
import os
from pathlib import Path as _Path

from ._case_artifacts import (
    _build_case_observable_inputs,
)
from .pick import _apply_pick, _parse_pick_arg
from .post_scf_observables import analyze_step3_uhf_observables as _analyze_step3_uhf_observables
from .reference_uhf import converge_reference_uhf as _converge_reference_uhf
from .selection_guidance import _attach_display_labels, _write_selection_artifacts
from .session import SessionManager as _SessionManager

logger = logging.getLogger(__name__)

_STEP3_SUPPORTED_PICK_MODES = {"all", "labels", "file"}
_UHF_DEFAULTS = {
    "conv_tol": 1e-8,
    "max_cycle": 2000,
    "stabilize_cycles": 20,
    "level_shift": 0.3,
    "damp": 0.2,
    "newton_refine": False,
    "newton_max_cycle": 8,
}
def step_uhf(
    session_dir: str,
    *,
    pick: str = "all",
    conv_tol: float = 1e-8,
    max_cycle: int = 2000,
    stabilize_cycles: int = 20,
    level_shift: float = 0.3,
    damp: float = 0.2,
    newton_refine: bool = False,
    newton_max_cycle: int = 8,
):
    """Run reference-state active-space UHF for selected configurations."""
    sm = _SessionManager(session_dir)
    sm.require_previous("step3_uhf", "step2_enumerate")
    controls = sm.resolve_method_controls(
        "uhf",
        _UHF_DEFAULTS,
        {
            "conv_tol": conv_tol,
            "max_cycle": max_cycle,
            "stabilize_cycles": stabilize_cycles,
            "level_shift": level_shift,
            "damp": damp,
            "newton_refine": newton_refine,
            "newton_max_cycle": newton_max_cycle,
        },
    )
    conv_tol = controls["conv_tol"]
    max_cycle = controls["max_cycle"]
    stabilize_cycles = controls["stabilize_cycles"]
    level_shift = controls["level_shift"]
    damp = controls["damp"]
    newton_refine = controls["newton_refine"]
    newton_max_cycle = controls["newton_max_cycle"]

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()

    cas = state["cas"]
    ci = state["cluster_info"]
    fcid = state["fcidump_data"]
    configs = enum_data["configs"]

    summary_for_pick = [
        {
            "label": cfg.label,
            "energy": 0.0,
            "converged": True,
            "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
        }
        for cfg in configs
    ]

    pick_spec = _parse_pick_arg(pick)
    if pick_spec["mode"] not in _STEP3_SUPPORTED_PICK_MODES:
        raise ValueError(
            "Step 3 runs before any energies exist, so only pick modes "
            "'all', 'labels', and 'file' are supported."
        )
    selected_labels = _apply_pick(pick_spec, summary_for_pick)
    label_set = set(selected_labels)
    selected_configs = [c for c in configs if c.label in label_set]

    print("=" * 60)
    print(f"Step 3: Reference UHF ({len(selected_configs)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Selected {len(selected_configs)} of {len(configs)} configs")
    print(
        "  BS-UHF stabilization: "
        f"cycles={stabilize_cycles}, level_shift={level_shift}, damp={damp}"
    )
    if newton_refine:
        print(f"  Newton refinement: enabled (max_cycle={newton_max_cycle})")

    sm.save_step_picked("step3_uhf", selected_labels)
    uhf_settings_payload = sm._build_step_settings_payload(
        state.get("settings"),
        theory="UHF",
        conv_tol=conv_tol,
        max_cycle=max_cycle,
        stabilize_cycles=stabilize_cycles,
        level_shift=level_shift,
        damp=damp,
        newton_refine=newton_refine,
        newton_max_cycle=newton_max_cycle,
    )

    results = []
    for i, cfg in enumerate(selected_configs):
        print(f"  [{i+1}/{len(selected_configs)}] {cfg.label} ... ", end="", flush=True)
        try:
            scf_result = _converge_reference_uhf(
                cas,
                cfg,
                fcid,
                ci,
                conv_tol=conv_tol,
                max_cycle=max_cycle,
                stabilize_cycles=stabilize_cycles,
                level_shift=level_shift,
                damp=damp,
                newton_refine=newton_refine,
                newton_max_cycle=newton_max_cycle,
            )
            sm.save_uhf_result(
                cfg.label,
                scf_result,
                family=cfg.spin_isomer.family if cfg.spin_isomer else "",
                state=state,
                settings_payload=uhf_settings_payload,
            )
            safe_label = cfg.label.replace("|", "_").replace(" ", "_")
            step3_results_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
            observable_path = os.path.join(step3_results_dir, f"{safe_label}_post_scf_observables.json")
            h5_path = os.path.join(step3_results_dir, f"{safe_label}_uhf.h5")
            observable_inputs = _build_case_observable_inputs(state, cfg)
            observables = None
            if observable_inputs is not None:
                try:
                    observables = _analyze_step3_uhf_observables(
                        step3_h5_path=h5_path,
                        cas_data_h5_path=observable_inputs["cas_data_h5_path"],
                        xyz_path=observable_inputs["xyz_path"],
                        cluster_info_path=observable_inputs["cluster_info_path"],
                        cas_settings_path=observable_inputs["cas_settings_path"],
                    )
                    _Path(observable_path).write_text(
                        json.dumps(observables, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.warning("Post-SCF observable analysis failed for %s: %s", cfg.label, exc)

            status = "OK" if scf_result.converged else "NOT CONVERGED"
            energy_str = (
                f"{scf_result.energy:.10f}"
                if scf_result.energy is not None else "N/A"
            )
            print(f"{status}  E={energy_str}  <S^2>={scf_result.s_squared:.4f}")
            primary_two_sz = (observables or {}).get("two_sz_by_metal_label", {}) if observables else {}

            results.append(
                {
                    "label": cfg.label,
                    "display_label": (scf_result.diagnostics or {}).get("final_state_signature") or cfg.label,
                    "energy": scf_result.energy,
                    "converged": scf_result.converged,
                    "s_squared": scf_result.s_squared,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "last_delta_e": (scf_result.diagnostics or {}).get("final_delta_e"),
                    "energy_tail": (scf_result.diagnostics or {}).get("energy_tail", []),
                    "two_s": (observables or {}).get("two_s"),
                    "two_sz_fe1": primary_two_sz.get("Fe1"),
                    "two_sz_fe2": primary_two_sz.get("Fe2"),
                    "final_d_basin": (scf_result.diagnostics or {}).get("final_d_basin", {}),
                    "final_site_spin_proxy": (scf_result.diagnostics or {}).get("final_site_spin_proxy", {}),
                    "final_state_signature": (scf_result.diagnostics or {}).get("final_state_signature"),
                }
            )
        except Exception as exc:
            logger.warning("UHF failed for %s: %s", cfg.label, exc)
            print(f"FAILED: {exc}")
            results.append(
                {
                    "label": cfg.label,
                    "display_label": cfg.label,
                    "energy": None,
                    "converged": False,
                    "s_squared": None,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "last_delta_e": None,
                    "energy_tail": [],
                    "final_d_basin": {},
                    "final_site_spin_proxy": {},
                    "final_state_signature": None,
                }
            )

    results.sort(key=lambda r: r.get("energy") or float("inf"))

    n_converged = sum(1 for r in results if r.get("converged"))
    energies = [r["energy"] for r in results if r.get("converged") and r["energy"] is not None]
    print(f"\n  Converged: {n_converged}/{len(results)}")
    if energies:
        print(f"  Energy range: [{min(energies):.10f}, {max(energies):.10f}]")

    results = sm._rebuild_uhf_summary(configs, current_results=results)
    _attach_display_labels(results, None)
    sm.save_step_summary("step3_uhf", "uhf_summary.json", results)
    _write_selection_artifacts(
        os.path.join(sm.session_dir, "step3_uhf"),
        step_name="Step 3 reference UHF",
        next_step_name="ccsd",
        summary=results,
        keep_default="1",
    )
    print(f"Step 3 complete. {n_converged} converged in this run; {len(results)} total step3 results available.")
