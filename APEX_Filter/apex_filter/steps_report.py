"""Final summary/reporting step for the interactive APEX_Filter pipeline."""

from __future__ import annotations

from datetime import datetime
import os

from .session import SessionManager


def _index_by_label(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        label = row.get("label")
        if label:
            indexed[label] = row
    return indexed


def _best_family(*rows: dict | None) -> str:
    for row in rows:
        if row and row.get("family"):
            return row["family"]
    return ""


def _best_display_label(label: str, *rows: dict | None) -> str:
    for row in rows:
        if row and row.get("display_label"):
            return row["display_label"]
    return label


def _pick_best_row(rows: list[dict], key: str) -> dict | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row[key])


def _build_label_rows(
    uhf_summary: list[dict],
    ccsd_summary: list[dict],
    ccsd_t_summary: list[dict],
    ccsdt_summary: list[dict],
    dmrg_extrapolation_summary: list[dict],
    cc_composite_summary: list[dict] | None = None,
) -> list[dict]:
    uhf_by_label = _index_by_label(uhf_summary)
    ccsd_by_label = _index_by_label(ccsd_summary)
    ccsd_t_by_label = _index_by_label(ccsd_t_summary)
    ccsdt_by_label = _index_by_label(ccsdt_summary)
    dmrg_by_label = _index_by_label(dmrg_extrapolation_summary)
    composite_by_label = {}
    for row in cc_composite_summary or []:
        label = row.get("label")
        if not label or row.get("energy") is None:
            continue
        prev = composite_by_label.get(label)
        if prev is None or row["energy"] < prev["energy"]:
            composite_by_label[label] = row

    labels = sorted(
        set(uhf_by_label)
        | set(ccsd_by_label)
        | set(ccsd_t_by_label)
        | set(ccsdt_by_label)
        | set(dmrg_by_label)
        | set(composite_by_label)
    )
    rows = []
    for label in labels:
        uhf = uhf_by_label.get(label)
        ccsd = ccsd_by_label.get(label)
        ccsd_t = ccsd_t_by_label.get(label)
        ccsdt = ccsdt_by_label.get(label)
        dmrg = dmrg_by_label.get(label)
        composite = composite_by_label.get(label)
        family = _best_family(composite, ccsd_t, ccsdt, ccsd, uhf, dmrg)
        display_label = _best_display_label(label, composite, ccsd_t, ccsdt, ccsd, uhf, dmrg)

        uhf_energy = uhf.get("energy") if uhf else None
        uhf_converged = uhf.get("converged") if uhf else None
        uhf_s_squared = uhf.get("s_squared") if uhf else None
        ccsd_energy = ccsd.get("energy") if ccsd and ccsd.get("converged") else None
        ccsd_corr_energy = ccsd.get("correlation_energy") if ccsd and ccsd.get("converged") else None
        ccsd_t_energy = ccsd_t.get("energy") if ccsd_t and ccsd_t.get("converged") else None
        ccsdt_energy = ccsdt.get("energy") if ccsdt and ccsdt.get("converged") else None
        dmrg_energy = dmrg.get("energy") if dmrg else None
        dmrg_uncertainty = dmrg.get("uncertainty") if dmrg else None
        dmrg_source_mode = dmrg.get("source_mode") if dmrg else None
        composite_energy = composite.get("energy") if composite else None
        composite_uncertainty = composite.get("uncertainty") if composite else None
        composite_freeze_occ = composite.get("freeze_occ") if composite else None

        dmrg_is_ranking_eligible = dmrg_energy is not None and dmrg_source_mode != "unconverged_fallback"

        consensus_energy = None
        consensus_uncertainty = None
        if ccsdt_energy is not None and dmrg_is_ranking_eligible:
            consensus_energy = 0.5 * (ccsdt_energy + dmrg_energy)
            half_diff = 0.5 * abs(ccsdt_energy - dmrg_energy)
            consensus_uncertainty = max(
                half_diff,
                float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
            )

        composite_dmrg_consensus_energy = None
        composite_dmrg_consensus_uncertainty = None
        if composite_energy is not None and dmrg_is_ranking_eligible:
            composite_dmrg_consensus_energy = 0.5 * (composite_energy + dmrg_energy)
            composite_dmrg_consensus_uncertainty = max(
                0.5 * abs(composite_energy - dmrg_energy),
                float(composite_uncertainty) if composite_uncertainty is not None else 0.0,
                float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
            )

        if composite_dmrg_consensus_energy is not None:
            ranking_method = "CC_composite+DMRG_consensus"
            ranking_energy = composite_dmrg_consensus_energy
            ranking_uncertainty = composite_dmrg_consensus_uncertainty
        elif composite_energy is not None:
            ranking_method = "CC_composite"
            ranking_energy = composite_energy
            ranking_uncertainty = composite_uncertainty
        elif consensus_energy is not None:
            ranking_method = "CCSDT+DMRG_consensus"
            ranking_energy = consensus_energy
            ranking_uncertainty = consensus_uncertainty
        elif dmrg_is_ranking_eligible:
            ranking_method = "DMRG_extrapolated"
            ranking_energy = dmrg_energy
            ranking_uncertainty = dmrg_uncertainty
        elif ccsdt_energy is not None:
            ranking_method = "CCSDT"
            ranking_energy = ccsdt_energy
            ranking_uncertainty = None
        else:
            ranking_method = "CCSD(T)"
            ranking_energy = ccsd_t_energy
            ranking_uncertainty = None

        rows.append(
            {
                "label": label,
                "display_label": display_label,
                "family": family,
                "uhf_energy": uhf_energy,
                "uhf_converged": uhf_converged,
                "uhf_s_squared": uhf_s_squared,
                "ccsd_energy": ccsd_energy,
                "ccsd_correlation_energy": ccsd_corr_energy,
                "ccsd_t_energy": ccsd_t_energy,
                "ccsdt_energy": ccsdt_energy,
                "cc_composite_energy": composite_energy,
                "cc_composite_uncertainty": composite_uncertainty,
                "cc_composite_freeze_occ": composite_freeze_occ,
                "dmrg_extrapolated_energy": dmrg_energy,
                "dmrg_uncertainty": dmrg_uncertainty,
                "dmrg_source_mode": dmrg_source_mode,
                "dmrg_ranking_eligible": dmrg_is_ranking_eligible,
                "dmrg_bond_dims": dmrg.get("bond_dims", []) if dmrg else [],
                "consensus_energy": consensus_energy,
                "consensus_uncertainty": consensus_uncertainty,
                "composite_dmrg_consensus_energy": composite_dmrg_consensus_energy,
                "composite_dmrg_consensus_uncertainty": composite_dmrg_consensus_uncertainty,
                "ranking_method": ranking_method,
                "ranking_energy": ranking_energy,
                "ranking_uncertainty": ranking_uncertainty,
            }
        )
    return rows


def _build_final_rows(
    uhf_summary: list[dict],
    ccsd_summary: list[dict],
    ccsd_t_summary: list[dict],
    ccsdt_summary: list[dict],
    dmrg_extrapolation_summary: list[dict],
    cc_composite_summary: list[dict] | None = None,
) -> list[dict]:
    label_rows = _build_label_rows(
        uhf_summary,
        ccsd_summary,
        ccsd_t_summary,
        ccsdt_summary,
        dmrg_extrapolation_summary,
        cc_composite_summary,
    )
    by_state: dict[str, list[dict]] = {}
    for row in label_rows:
        state = row.get("display_label") or row["label"]
        by_state.setdefault(state, []).append(row)

    rows = []
    for state, state_rows in sorted(by_state.items()):
        best_uhf = _pick_best_row(state_rows, "uhf_energy")
        best_ccsd = _pick_best_row(state_rows, "ccsd_energy")
        best_ccsd_t = _pick_best_row(state_rows, "ccsd_t_energy")
        best_ccsdt = _pick_best_row(state_rows, "ccsdt_energy")
        best_dmrg = _pick_best_row(state_rows, "dmrg_extrapolated_energy")
        best_composite = _pick_best_row(state_rows, "cc_composite_energy")
        representative = _pick_best_row(state_rows, "ranking_energy") or state_rows[0]

        uhf_energy = best_uhf.get("uhf_energy") if best_uhf else None
        uhf_converged = best_uhf.get("uhf_converged") if best_uhf else None
        uhf_s_squared = best_uhf.get("uhf_s_squared") if best_uhf else None
        ccsd_energy = best_ccsd.get("ccsd_energy") if best_ccsd else None
        ccsd_corr_energy = best_ccsd.get("ccsd_correlation_energy") if best_ccsd else None
        ccsd_t_energy = best_ccsd_t.get("ccsd_t_energy") if best_ccsd_t else None
        ccsdt_energy = best_ccsdt.get("ccsdt_energy") if best_ccsdt else None
        dmrg_energy = best_dmrg.get("dmrg_extrapolated_energy") if best_dmrg else None
        dmrg_uncertainty = best_dmrg.get("dmrg_uncertainty") if best_dmrg else None
        dmrg_source_mode = best_dmrg.get("dmrg_source_mode") if best_dmrg else None
        dmrg_bond_dims = best_dmrg.get("dmrg_bond_dims") if best_dmrg else []
        composite_energy = best_composite.get("cc_composite_energy") if best_composite else None
        composite_uncertainty = best_composite.get("cc_composite_uncertainty") if best_composite else None
        composite_freeze_occ = best_composite.get("cc_composite_freeze_occ") if best_composite else None
        family = _best_family(*state_rows)
        source_initial_labels = sorted({row["label"] for row in state_rows if row.get("label")})

        dmrg_is_ranking_eligible = dmrg_energy is not None and dmrg_source_mode != "unconverged_fallback"

        consensus_energy = None
        consensus_uncertainty = None
        if ccsdt_energy is not None and dmrg_is_ranking_eligible:
            consensus_energy = 0.5 * (ccsdt_energy + dmrg_energy)
            half_diff = 0.5 * abs(ccsdt_energy - dmrg_energy)
            consensus_uncertainty = max(
                half_diff,
                float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
            )

        composite_dmrg_consensus_energy = None
        composite_dmrg_consensus_uncertainty = None
        if composite_energy is not None and dmrg_is_ranking_eligible:
            composite_dmrg_consensus_energy = 0.5 * (composite_energy + dmrg_energy)
            composite_dmrg_consensus_uncertainty = max(
                0.5 * abs(composite_energy - dmrg_energy),
                float(composite_uncertainty) if composite_uncertainty is not None else 0.0,
                float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
            )

        if composite_dmrg_consensus_energy is not None:
            ranking_method = "CC_composite+DMRG_consensus"
            ranking_energy = composite_dmrg_consensus_energy
            ranking_uncertainty = composite_dmrg_consensus_uncertainty
        elif composite_energy is not None:
            ranking_method = "CC_composite"
            ranking_energy = composite_energy
            ranking_uncertainty = composite_uncertainty
        elif consensus_energy is not None:
            ranking_method = "CCSDT+DMRG_consensus"
            ranking_energy = consensus_energy
            ranking_uncertainty = consensus_uncertainty
        elif dmrg_is_ranking_eligible:
            ranking_method = "DMRG_extrapolated"
            ranking_energy = dmrg_energy
            ranking_uncertainty = dmrg_uncertainty
        elif ccsdt_energy is not None:
            ranking_method = "CCSDT"
            ranking_energy = ccsdt_energy
            ranking_uncertainty = None
        else:
            ranking_method = "CCSD(T)"
            ranking_energy = ccsd_t_energy
            ranking_uncertainty = None

        rows.append(
            {
                "label": representative["label"],
                "display_label": state,
                "family": family,
                "representative_label": representative["label"],
                "source_initial_labels": source_initial_labels,
                "source_initial_count": len(source_initial_labels),
                "uhf_energy": uhf_energy,
                "uhf_converged": uhf_converged,
                "uhf_s_squared": uhf_s_squared,
                "ccsd_energy": ccsd_energy,
                "ccsd_correlation_energy": ccsd_corr_energy,
                "ccsd_t_energy": ccsd_t_energy,
                "ccsdt_energy": ccsdt_energy,
                "cc_composite_energy": composite_energy,
                "cc_composite_uncertainty": composite_uncertainty,
                "cc_composite_freeze_occ": composite_freeze_occ,
                "dmrg_extrapolated_energy": dmrg_energy,
                "dmrg_uncertainty": dmrg_uncertainty,
                "dmrg_source_mode": dmrg_source_mode,
                "dmrg_ranking_eligible": dmrg_is_ranking_eligible,
                "dmrg_bond_dims": dmrg_bond_dims or [],
                "consensus_energy": consensus_energy,
                "consensus_uncertainty": consensus_uncertainty,
                "composite_dmrg_consensus_energy": composite_dmrg_consensus_energy,
                "composite_dmrg_consensus_uncertainty": composite_dmrg_consensus_uncertainty,
                "ranking_method": ranking_method,
                "ranking_energy": ranking_energy,
                "ranking_uncertainty": ranking_uncertainty,
            }
        )

    rows.sort(key=lambda row: row["ranking_energy"] if row["ranking_energy"] is not None else float("inf"))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _fmt_energy(value) -> str:
    return f"{value:.10f}" if value is not None else "N/A"


def _fmt_uncertainty(value) -> str:
    return f"{value:.2e}" if value is not None else "N/A"


def _render_markdown(rows: list[dict], *, report_scope: str) -> str:
    lines = [
        "# APEX Final Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Report scope: `{report_scope}`",
        "",
        "Ranking priority: `CC composite + DMRG consensus` > `CC composite` > `CCSDT+DMRG consensus` > `DMRG extrapolated` > `CCSDT` > `CCSD(T)`",
        "",
        "Consensus definition: mean of `CCSDT` and extrapolated `DMRG`; uncertainty is `max(DMRG uncertainty, |CCSDT-DMRG|/2)`.",
        "",
        "Only `DMRG` entries with `source_mode != unconverged_fallback` are allowed to influence ranking directly. Fallback extrapolations are retained for reference/provenance but do not outrank converged CC results.",
        "",
        "| Rank | State | Representative Label | Source Guess Count | Family | Ranking Method | Ranking Energy (Ha) | Ranking Uncertainty | CCSD(T) | CCSDT | CC Composite | DMRG Extrap. | Consensus |",
        "|------|-------|----------------------|--------------------|--------|----------------|---------------------|---------------------|---------|-------|--------------|--------------|-----------|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {display_label} | {representative_label} | {source_initial_count} | {family} | {ranking_method} | {ranking_energy} | {ranking_uncertainty} | {ccsd_t} | {ccsdt} | {cc_composite} | {dmrg} | {consensus} |".format(
                rank=row["rank"],
                display_label=row.get("display_label", row["label"]).replace("|", "\\|"),
                representative_label=row.get("representative_label", row["label"]).replace("|", "\\|"),
                source_initial_count=row.get("source_initial_count", 1),
                family=row.get("family", ""),
                ranking_method=row["ranking_method"],
                ranking_energy=_fmt_energy(row["ranking_energy"]),
                ranking_uncertainty=_fmt_uncertainty(row.get("ranking_uncertainty")),
                ccsd_t=_fmt_energy(row.get("ccsd_t_energy")),
                ccsdt=_fmt_energy(row.get("ccsdt_energy")),
                cc_composite=_fmt_energy(row.get("cc_composite_energy")),
                dmrg=_fmt_energy(row.get("dmrg_extrapolated_energy")),
                consensus=_fmt_energy(row.get("consensus_energy")),
            )
        )
    lines.extend(["", "## Provenance and Method Ladder", ""])
    for row in rows:
        lines.append(f"### {row.get('display_label', row['label'])}")
        lines.append("")
        lines.append(f"- Representative label: `{row.get('representative_label', row['label'])}`")
        lines.append(f"- Source guess count: `{row.get('source_initial_count', 1)}`")
        lines.append(f"- Source initial labels: `{', '.join(row.get('source_initial_labels', []))}`")
        if row.get("dmrg_source_mode"):
            lines.append(f"- DMRG source mode: `{row['dmrg_source_mode']}`")
        lines.append("- Method ladder:")
        lines.append(f"  - UHF: `{_fmt_energy(row.get('uhf_energy'))}`  (<S^2> = `{_fmt_energy(row.get('uhf_s_squared'))}`)")
        lines.append(f"  - CCSD: `{_fmt_energy(row.get('ccsd_energy'))}`  (corr = `{_fmt_energy(row.get('ccsd_correlation_energy'))}`)")
        lines.append(f"  - CCSD(T): `{_fmt_energy(row.get('ccsd_t_energy'))}`")
        lines.append(f"  - CCSDT: `{_fmt_energy(row.get('ccsdt_energy'))}`")
        lines.append(f"  - DMRG extrapolated: `{_fmt_energy(row.get('dmrg_extrapolated_energy'))}`")
        lines.append(f"  - CC composite: `{_fmt_energy(row.get('cc_composite_energy'))}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def step_report(session_dir: str):
    """Build a final multi-method summary from CC and DMRG stages."""
    sm = SessionManager(session_dir)
    sm.require_previous("step10_report", "step9_extrapolate")

    uhf_summary = sm.load_uhf_summary()
    ccsd_summary = sm.load_ccsd_summary()
    ccsd_t_summary = sm.load_ccsd_t_summary()
    ccsdt_summary = sm.load_ccsdt_summary()
    dmrg_extrapolation_summary = sm.load_dmrg_extrapolation_summary()
    cc_composite_summary = []
    composite_path = os.path.join(sm.session_dir, "step12_cc_composite", "cc_composite_summary.json")
    if os.path.exists(composite_path):
        cc_composite_summary = sm.load_cc_composite_summary()

    print("=" * 60)
    print("Step 10: Final summary/report")
    print("=" * 60)

    rows = _build_final_rows(
        uhf_summary,
        ccsd_summary,
        ccsd_t_summary,
        ccsdt_summary,
        dmrg_extrapolation_summary,
        cc_composite_summary,
    )
    report_scope = "post-step12 composite-capable" if cc_composite_summary else "post-step9 interim summary"
    markdown = _render_markdown(rows, report_scope=report_scope)

    for row in rows[:10]:
        print(
            f"  #{row['rank']:3d}  {row.get('display_label', row['label']):<40}  "
            f"{row['ranking_method']:<22}  E={_fmt_energy(row['ranking_energy'])}"
        )

    sm.save_final_summary(rows, markdown=markdown)
    for stale in (
        "selection_candidates.csv",
        "selection_worklist.csv",
        "selection_guide.md",
        "selection_candidates.json",
        "pick_labels_all.json",
        "pick_labels_template.json",
    ):
        stale_path = os.path.join(sm.session_dir, "step10_report", stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    print(f"Step 10 complete. Final summary saved for {len(rows)} configurations.")
