"""FNO-style high-order CC and composite-energy steps.

Current implementation note:
- This step uses spin-resolved occupied natural orbitals from active-space UCCSD.
- It freezes the leading occupied NOs in each spin channel.
- All virtual orbitals are retained.

So the current semantics are closer to an ``occupied-NO freeze`` workflow than a
full virtual-space threshold FNO implementation.
"""

from __future__ import annotations

import os

import numpy as np

from .energy_extrapolation import cc_composite_energy
from .fno_truncation import build_fno_subspace_from_uccsd
from .pick import apply_pick, parse_pick_arg
from .reference_hast_ucc import run_reference_hast_ucc
from .selection_guidance import attach_display_labels, build_display_label_map, write_selection_artifacts
from .session import SessionManager

_FNO_DEFAULTS = {
    "freeze_occ": [2, 4, 6],
}


def _sanitize_label(label: str) -> str:
    return label.replace("|", "_").replace(" ", "_")


def _save_fno_pair_result(npz_path: str, payload: dict):
    np.savez(npz_path, **payload)


def step_fno_uccsdtq(
    session_dir: str,
    *,
    pick: str = "all",
    freeze_occ: list[int] | None = None,
):
    """Run occupied-NO-freeze CCSDT / CCSDTQ on selected configs."""
    sm = SessionManager(session_dir)
    sm.require_previous("step11_fno_uccsdtq", "step6_ccsdt")
    controls = sm.resolve_method_controls(
        "fno_uccsdtq",
        _FNO_DEFAULTS,
        {"freeze_occ": freeze_occ},
    )
    freeze_occ = controls["freeze_occ"]

    state = sm.load_load_state()
    enum_data = sm.load_enumeration()
    ccsdt_summary = sm.load_ccsdt_summary()
    display_label_map = build_display_label_map(ccsdt_summary)

    pick_spec = parse_pick_arg(pick)
    selected_labels = apply_pick(pick_spec, ccsdt_summary)
    config_map = {cfg.label: cfg for cfg in enum_data["configs"]}
    selected_configs = [config_map[label] for label in selected_labels if label in config_map]

    print("=" * 60)
    print(f"Step 11: FNO-UCCSDTQ ({len(selected_configs)} configs)")
    print("=" * 60)
    print(f"  Pick strategy: {pick}")
    print(f"  freeze_occ   : {freeze_occ}")
    print("  Scheme       : spin-resolved occupied-NO freeze (all virtual orbitals retained)")

    sm.save_fno_picked(selected_labels)
    if not selected_configs:
        print("  No configs selected. Aborting.")
        sm.save_fno_summary([])
        return

    fcid = state["fcidump_data"]
    uhf_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    results_dir = sm.fno_results_dir
    os.makedirs(results_dir, exist_ok=True)

    summary = []
    for cfg in selected_configs:
        safe_label = _sanitize_label(cfg.label)
        uhf_npz = os.path.join(uhf_dir, f"{safe_label}_uhf.npz")

        for freeze in freeze_occ:
            out_npz = os.path.join(results_dir, f"{safe_label}_freeze{freeze}_fno_uccsdtq.npz")
            print(f"  {cfg.label} @ freeze_occ={freeze} ... ", end="", flush=True)
            try:
                subspace = build_fno_subspace_from_uccsd(fcid, uhf_npz, freeze_occ=freeze)
                ccsdt = run_reference_hast_ucc(
                    fcid,
                    uhf_npz,
                    t_order=3,
                    frozen=subspace.frozen,
                    mo_coeff=subspace.mo_coeff,
                    mo_occ=subspace.mo_occ,
                    mo_energy=subspace.mo_energy,
                )
                ccsdtq = run_reference_hast_ucc(
                    fcid,
                    uhf_npz,
                    t_order=4,
                    frozen=subspace.frozen,
                    mo_coeff=subspace.mo_coeff,
                    mo_occ=subspace.mo_occ,
                    mo_energy=subspace.mo_energy,
                )

                payload = {
                    "fno_scheme": "occupied_no_freeze",
                    "freeze_occ_alpha": subspace.frozen_occ_alpha,
                    "freeze_occ_beta": subspace.frozen_occ_beta,
                    "kept_occ_alpha": subspace.kept_occ_alpha,
                    "kept_occ_beta": subspace.kept_occ_beta,
                    "uccsd_total": subspace.uccsd_energy,
                    "uccsd_corr": subspace.uccsd_corr,
                    "uccsd_converged": subspace.converged,
                    "ccsdt_total": ccsdt.energy,
                    "ccsdt_corr": ccsdt.correlation_energy,
                    "ccsdt_converged": ccsdt.converged,
                    "ccsdtq_total": ccsdtq.energy,
                    "ccsdtq_corr": ccsdtq.correlation_energy,
                    "ccsdtq_converged": ccsdtq.converged,
                    "occ_noons_alpha": subspace.occupied_noons_alpha,
                    "occ_noons_beta": subspace.occupied_noons_beta,
                }
                _save_fno_pair_result(out_npz, payload)
                print(f"OK  E(T)={ccsdt.energy:.10f}  E(TQ)={ccsdtq.energy:.10f}")
                summary.append(
                    {
                        "label": cfg.label,
                        "display_label": display_label_map.get(cfg.label, cfg.label),
                        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                        "fno_scheme": "occupied_no_freeze",
                        "freeze_occ": freeze,
                        "kept_occ_alpha": subspace.kept_occ_alpha,
                        "kept_occ_beta": subspace.kept_occ_beta,
                        "ccsdt_fno_energy": ccsdt.energy if ccsdt.converged else None,
                        "ccsdtq_fno_energy": ccsdtq.energy if ccsdtq.converged else None,
                        "converged": bool(ccsdt.converged and ccsdtq.converged),
                    }
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                summary.append(
                    {
                        "label": cfg.label,
                        "display_label": display_label_map.get(cfg.label, cfg.label),
                        "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                        "fno_scheme": "occupied_no_freeze",
                        "freeze_occ": freeze,
                        "kept_occ_alpha": None,
                        "kept_occ_beta": None,
                        "ccsdt_fno_energy": None,
                        "ccsdtq_fno_energy": None,
                        "converged": False,
                    }
                )

    summary.sort(
        key=lambda row: (
            row["label"],
            row["freeze_occ"],
            row["ccsdtq_fno_energy"] if row["ccsdtq_fno_energy"] is not None else float("inf"),
        )
    )
    attach_display_labels(summary, ccsdt_summary)
    sm.save_fno_summary(summary)
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step11_fno_uccsdtq"),
        step_name="Step 11 FNO-UCCSDTQ",
        next_step_name="cc-composite",
        summary=summary,
        keep_default="1",
    )
    n_ok = sum(1 for row in summary if row.get("converged"))
    print(f"Step 11 complete. {n_ok} FNO CC pairs saved.")


def step_cc_composite(session_dir: str):
    """Combine full CCSDT with FNO CCSDT/CCSDTQ corrections."""
    sm = SessionManager(session_dir)
    sm.require_previous("step12_cc_composite", "step11_fno_uccsdtq")

    ccsdt_summary = sm.load_ccsdt_summary()
    fno_summary = sm.load_fno_summary()
    ccsdt_by_label = {
        row["label"]: row for row in ccsdt_summary if row.get("converged") and row.get("energy") is not None
    }

    print("=" * 60)
    print("Step 12: CC composite energy")
    print("=" * 60)

    summary = []
    for row in fno_summary:
        label = row["label"]
        full = ccsdt_by_label.get(label)
        if full is None:
            continue
        e_fno_t = row.get("ccsdt_fno_energy")
        e_fno_tq = row.get("ccsdtq_fno_energy")
        if e_fno_t is None or e_fno_tq is None:
            continue

        composite = cc_composite_energy(full["energy"], e_fno_tq, e_fno_t)
        summary.append(
            {
                "label": label,
                "display_label": row.get("display_label", label),
                "family": row.get("family", ""),
                "fno_scheme": row.get("fno_scheme", "occupied_no_freeze"),
                "freeze_occ": row.get("freeze_occ"),
                "ccsdt_full_energy": full["energy"],
                "ccsdt_fno_energy": e_fno_t,
                "ccsdtq_fno_energy": e_fno_tq,
                "energy": composite.energy,
                "uncertainty": composite.uncertainty,
                "description": composite.description,
                "converged": True,
            }
        )

    summary.sort(key=lambda row: row["energy"] if row["energy"] is not None else float("inf"))
    attach_display_labels(summary, fno_summary)
    sm.save_cc_composite_summary(summary)
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step12_cc_composite"),
        step_name="Step 12 CC composite",
        next_step_name=None,
        summary=summary,
        keep_default="1",
    )
    for row in summary[:10]:
        print(
            f"  {row.get('display_label', row['label'])} freeze={row['freeze_occ']} "
            f"E={row['energy']:.10f} ± {row['uncertainty']:.2e}"
        )
    print(f"Step 12 complete. {len(summary)} composite energies saved.")
