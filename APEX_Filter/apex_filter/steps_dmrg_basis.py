"""DMRG orbital-basis preparation step."""

import csv
import os
from shared.orbital_methods.metadata import DMRG_BASIS_SOURCE_METHOD

from .CAS_loader import load_filter_inputs
from .dmrg_orbital_basis import build_dmrg_orbital_basis, save_dmrg_orbital_basis
from .pick import apply_pick, parse_pick_arg
from .selection_guidance import attach_display_labels, build_display_label_map, write_selection_artifacts
from .session import SessionManager

_DMRG_BASIS_DEFAULTS = {
    "localization_method": "pm",
    "cc_conv_tol": 1e-8,
    "cc_max_cycle": 2000,
    "cc_diis_space": 12,
    "cc_direct": False,
    "pm_pop_method": "mulliken",
    "pm_conv_tol": 1e-6,
    "pm_conv_tol_grad": None,
    "pm_max_cycle": 100,
    "boys_conv_tol": 1e-6,
    "boys_conv_tol_grad": None,
    "boys_max_cycle": 100,
    "ordering_matrix_mode": "exchange_proxy",
    "exchange_proxy_max_orbitals": 64,
    "ga_generations": 100,
    "ga_population": 50,
    "ga_mutation_rate": 0.1,
    "ga_seed": 17,
}


def step_dmrg_basis(
    session_dir: str,
    *,
    pick: str = "all",
    localization_method: str = "pm",
    cc_conv_tol: float = 1e-8,
    cc_max_cycle: int = 2000,
    cc_diis_space: int = 12,
    cc_direct: bool = False,
    pm_pop_method: str = "mulliken",
    pm_conv_tol: float = 1e-6,
    pm_conv_tol_grad: float | None = None,
    pm_max_cycle: int = 100,
    boys_conv_tol: float = 1e-6,
    boys_conv_tol_grad: float | None = None,
    boys_max_cycle: int = 100,
    ordering_matrix_mode: str = "exchange_proxy",
    exchange_proxy_max_orbitals: int = 64,
    ga_generations: int = 100,
    ga_population: int = 50,
    ga_mutation_rate: float = 0.1,
    ga_seed: int = 17,
):
    """Prepare unrestricted orbital bases for later UDMRG calculations."""
    sm = SessionManager(session_dir)
    sm.require_previous("step7_dmrg_basis", "step6_ccsdt")
    controls = sm.resolve_method_controls(
        "dmrg_basis",
        _DMRG_BASIS_DEFAULTS,
        {
            "localization_method": localization_method,
            "cc_conv_tol": cc_conv_tol,
            "cc_max_cycle": cc_max_cycle,
            "cc_diis_space": cc_diis_space,
            "cc_direct": cc_direct,
            "pm_pop_method": pm_pop_method,
            "pm_conv_tol": pm_conv_tol,
            "pm_conv_tol_grad": pm_conv_tol_grad,
            "pm_max_cycle": pm_max_cycle,
            "boys_conv_tol": boys_conv_tol,
            "boys_conv_tol_grad": boys_conv_tol_grad,
            "boys_max_cycle": boys_max_cycle,
            "ordering_matrix_mode": ordering_matrix_mode,
            "exchange_proxy_max_orbitals": exchange_proxy_max_orbitals,
            "ga_generations": ga_generations,
            "ga_population": ga_population,
            "ga_mutation_rate": ga_mutation_rate,
            "ga_seed": ga_seed,
        },
    )
    localization_method = controls["localization_method"]
    cc_conv_tol = controls["cc_conv_tol"]
    cc_max_cycle = controls["cc_max_cycle"]
    cc_diis_space = controls["cc_diis_space"]
    cc_direct = controls["cc_direct"]
    pm_pop_method = controls["pm_pop_method"]
    pm_conv_tol = controls["pm_conv_tol"]
    pm_conv_tol_grad = controls.get("pm_conv_tol_grad")
    pm_max_cycle = controls["pm_max_cycle"]
    boys_conv_tol = controls["boys_conv_tol"]
    boys_conv_tol_grad = controls.get("boys_conv_tol_grad")
    boys_max_cycle = controls["boys_max_cycle"]
    ordering_matrix_mode = controls["ordering_matrix_mode"]
    exchange_proxy_max_orbitals = controls["exchange_proxy_max_orbitals"]
    ga_generations = controls["ga_generations"]
    ga_population = controls["ga_population"]
    ga_mutation_rate = controls["ga_mutation_rate"]
    ga_seed = controls["ga_seed"]

    session_meta = sm.load()
    config_path = session_meta.get("config_path")
    if not config_path:
        raise RuntimeError("Session is missing config_path; cannot rebuild the original mol object")

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    ccsdt_summary = sm.load_ccsdt_summary()
    ccsdt_by_label = {row.get("label"): row for row in ccsdt_summary if row.get("label")}
    display_label_map = build_display_label_map(ccsdt_summary)
    inputs = load_filter_inputs(config_path)

    cas = state["cas"]
    fcid = state["fcidump_data"]
    mol = inputs.mol
    configs = enum_data["configs"]

    pick_spec = parse_pick_arg(pick)
    selected_labels = apply_pick(pick_spec, ccsdt_summary)
    config_map = {cfg.label: cfg for cfg in configs}
    selected_configs = [config_map[label] for label in selected_labels if label in config_map]

    print("=" * 60)
    print(f"Step 7: DMRG orbital basis ({len(selected_configs)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  Localization: {localization_method}")
    print("  Route: UCCSD NO -> split localization -> alpha/beta pairing -> GA ordering")

    sm.save_dmrg_basis_picked(selected_labels)
    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_dmrg_basis_summary([])
        return

    results_dir = sm.dmrg_basis_results_dir
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary = []
    for idx, cfg in enumerate(selected_configs, 1):
        label = cfg.label.replace("|", "_").replace(" ", "_")
        uhf_npz = os.path.join(uhf_dir, f"{label}_uhf.npz")
        out_npz = os.path.join(results_dir, f"{label}_dmrg_basis.npz")

        print(f"  [{idx}/{len(selected_configs)}] {cfg.label} ... ", end="", flush=True)
        try:
            basis = build_dmrg_orbital_basis(
                mol,
                cas,
                fcid,
                uhf_npz,
                localization_method=localization_method,
                cc_conv_tol=cc_conv_tol,
                cc_max_cycle=cc_max_cycle,
                cc_diis_space=cc_diis_space,
                cc_direct=cc_direct,
                pm_pop_method=pm_pop_method,
                pm_conv_tol=pm_conv_tol,
                pm_conv_tol_grad=pm_conv_tol_grad,
                pm_max_cycle=pm_max_cycle,
                boys_conv_tol=boys_conv_tol,
                boys_conv_tol_grad=boys_conv_tol_grad,
                boys_max_cycle=boys_max_cycle,
                ordering_matrix_mode=ordering_matrix_mode,
                exchange_proxy_max_orbitals=exchange_proxy_max_orbitals,
                ga_generations=ga_generations,
                ga_population=ga_population,
                ga_mutation_rate=ga_mutation_rate,
                ga_seed=ga_seed,
            )
            save_dmrg_orbital_basis(basis, out_npz)
            print("OK")
            upstream = ccsdt_by_label.get(cfg.label, {})
            summary.append(
                {
                    "label": cfg.label,
                    "display_label": display_label_map.get(cfg.label, cfg.label),
                    "energy": upstream.get("energy"),
                    "converged": True,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "localization_method": basis.localization_method,
                    "source_method": basis.source_method,
                    "nocc_alpha": basis.nocc_alpha,
                    "nocc_beta": basis.nocc_beta,
                    "orth_err_alpha": basis.orth_err_alpha,
                    "orth_err_beta": basis.orth_err_beta,
                    "pair_diag_overlap_min": basis.pair_diag_overlap_min,
                    "pair_diag_overlap_mean": basis.pair_diag_overlap_mean,
                    "diag_dominant_fraction": basis.diag_dominant_fraction,
                    "ordering_is_permutation": basis.ordering_is_permutation,
                    "ga_cost": basis.ga_cost,
                    "fiedler_cost": basis.fiedler_cost,
                }
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            upstream = ccsdt_by_label.get(cfg.label, {})
            summary.append(
                {
                    "label": cfg.label,
                    "display_label": display_label_map.get(cfg.label, cfg.label),
                    "energy": upstream.get("energy"),
                    "converged": False,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "localization_method": localization_method,
                    "source_method": DMRG_BASIS_SOURCE_METHOD,
                }
            )

    attach_display_labels(summary, ccsdt_summary)
    sm.save_dmrg_basis_summary(summary)
    _write_dmrg_basis_qc_artifacts(os.path.join(sm.session_dir, "step7_dmrg_basis"), summary)
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step7_dmrg_basis"),
        step_name="Step 7 DMRG orbital basis",
        next_step_name="dmrg",
        summary=summary,
        keep_default="1",
    )
    n_ok = sum(1 for row in summary if row.get("converged"))
    print(f"Step 7 complete. {n_ok} orbital-basis preparations saved.")


def _write_dmrg_basis_qc_artifacts(step_dir: str, summary: list[dict]):
    """Write a compact QC table for the DMRG-basis preparation step."""
    rows = [
        {
            "label": row.get("label", ""),
            "display_label": row.get("display_label", row.get("label", "")),
            "converged": row.get("converged", False),
            "orth_err_alpha": row.get("orth_err_alpha"),
            "orth_err_beta": row.get("orth_err_beta"),
            "pair_diag_overlap_min": row.get("pair_diag_overlap_min"),
            "pair_diag_overlap_mean": row.get("pair_diag_overlap_mean"),
            "diag_dominant_fraction": row.get("diag_dominant_fraction"),
            "ordering_is_permutation": row.get("ordering_is_permutation"),
            "ga_cost": row.get("ga_cost"),
            "fiedler_cost": row.get("fiedler_cost"),
        }
        for row in summary
    ]
    if not rows:
        return

    with open(os.path.join(step_dir, "dmrg_basis_qc.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys())
    with open(os.path.join(step_dir, "dmrg_basis_qc.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
