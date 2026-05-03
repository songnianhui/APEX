"""Step 10 final reporting entrypoint for the staged APEX_Filter workflow."""

from __future__ import annotations

import csv
from datetime import datetime as _datetime
import io
import json
import os

from ._dmrg_summary import _dmrg_source_mode_allows_ranking
from .session import SessionManager as _SessionManager
from ._step_selection_artifacts import _cleanup_step_selection_artifacts
from shared.formatting import format_energy as _fmt_energy


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


def _observable_fields(row: dict | None, prefix: str = "") -> dict[str, float | None]:
    if not row:
        return {
            "s_squared": None,
            "two_s": None,
            "two_sz_fe1": None,
            "two_sz_fe2": None,
        }
    def key(name: str) -> str:
        return f"{prefix}{name}" if prefix else name
    return {
        "s_squared": row.get(key("s_squared")),
        "two_s": row.get(key("two_s")),
        "two_sz_fe1": row.get(key("two_sz_fe1")),
        "two_sz_fe2": row.get(key("two_sz_fe2")),
    }


def _load_fcidump_ecore(session_dir: str) -> float | None:
    step1_dir = os.path.join(session_dir, "step1_load")
    ref_path = os.path.join(step1_dir, "fcidump_ref.json")
    if not os.path.exists(ref_path):
        return None
    with open(ref_path, encoding="utf-8") as f:
        payload = json.load(f)
    fcidump_path = payload.get("fcidump_path")
    if not fcidump_path:
        return None
    ecore_path = fcidump_path + ".ecore"
    if not os.path.exists(ecore_path):
        return None
    with open(ecore_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _build_ccsdt_dmrg_consensus(
    ccsdt_energy: float | None,
    dmrg_energy: float | None,
    dmrg_uncertainty: float | None,
    *,
    dmrg_is_ranking_eligible: bool,
) -> tuple[float | None, float | None]:
    if ccsdt_energy is None or dmrg_energy is None or not dmrg_is_ranking_eligible:
        return None, None

    consensus_energy = 0.5 * (ccsdt_energy + dmrg_energy)
    half_diff = 0.5 * abs(ccsdt_energy - dmrg_energy)
    consensus_uncertainty = max(
        half_diff,
        float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
    )
    return consensus_energy, consensus_uncertainty


def _build_composite_dmrg_consensus(
    composite_energy: float | None,
    composite_uncertainty: float | None,
    dmrg_energy: float | None,
    dmrg_uncertainty: float | None,
    *,
    dmrg_is_ranking_eligible: bool,
) -> tuple[float | None, float | None]:
    if composite_energy is None or dmrg_energy is None or not dmrg_is_ranking_eligible:
        return None, None

    consensus_energy = 0.5 * (composite_energy + dmrg_energy)
    consensus_uncertainty = max(
        0.5 * abs(composite_energy - dmrg_energy),
        float(composite_uncertainty) if composite_uncertainty is not None else 0.0,
        float(dmrg_uncertainty) if dmrg_uncertainty is not None else 0.0,
    )
    return consensus_energy, consensus_uncertainty


def _resolve_ranking_result(
    *,
    ccsd_t_energy: float | None,
    ccsdt_energy: float | None,
    dmrg_energy: float | None,
    dmrg_uncertainty: float | None,
    composite_energy: float | None,
    composite_uncertainty: float | None,
    dmrg_is_ranking_eligible: bool,
) -> tuple[str, float | None, float | None, float | None, float | None]:
    ccsdt_dmrg_consensus_energy, ccsdt_dmrg_consensus_uncertainty = _build_ccsdt_dmrg_consensus(
        ccsdt_energy,
        dmrg_energy,
        dmrg_uncertainty,
        dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
    )
    composite_dmrg_consensus_energy, composite_dmrg_consensus_uncertainty = _build_composite_dmrg_consensus(
        composite_energy,
        composite_uncertainty,
        dmrg_energy,
        dmrg_uncertainty,
        dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
    )

    if composite_dmrg_consensus_energy is not None:
        return (
            "CC composite + DMRG consensus",
            composite_dmrg_consensus_energy,
            composite_dmrg_consensus_uncertainty,
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
        )
    if composite_energy is not None:
        return (
            "CC composite",
            composite_energy,
            composite_uncertainty,
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
        )
    if ccsdt_dmrg_consensus_energy is not None:
        return (
            "CCSDT + DMRG consensus",
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
        )
    if dmrg_energy is not None and dmrg_is_ranking_eligible:
        return (
            "DMRG extrapolated",
            dmrg_energy,
            dmrg_uncertainty,
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
        )
    if ccsdt_energy is not None:
        return (
            "CCSDT",
            ccsdt_energy,
            None,
            ccsdt_dmrg_consensus_energy,
            ccsdt_dmrg_consensus_uncertainty,
        )
    return (
        "CCSD(T)",
        ccsd_t_energy,
        None,
        ccsdt_dmrg_consensus_energy,
        ccsdt_dmrg_consensus_uncertainty,
    )


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
        uhf_two_s = uhf.get("two_s") if uhf else None
        uhf_two_sz_fe1 = uhf.get("two_sz_fe1") if uhf else None
        uhf_two_sz_fe2 = uhf.get("two_sz_fe2") if uhf else None
        ccsd_energy = ccsd.get("energy") if ccsd and ccsd.get("converged") else None
        ccsd_corr_energy = ccsd.get("correlation_energy") if ccsd and ccsd.get("converged") else None
        ccsd_s_squared = ccsd.get("s_squared") if ccsd and ccsd.get("converged") else None
        ccsd_two_s = ccsd.get("two_s") if ccsd and ccsd.get("converged") else None
        ccsd_two_sz_fe1 = ccsd.get("two_sz_fe1") if ccsd and ccsd.get("converged") else None
        ccsd_two_sz_fe2 = ccsd.get("two_sz_fe2") if ccsd and ccsd.get("converged") else None
        ccsd_t_energy = ccsd_t.get("energy") if ccsd_t and ccsd_t.get("converged") else None
        ccsd_t_s_squared = ccsd_t.get("s_squared") if ccsd_t and ccsd_t.get("converged") else None
        ccsd_t_two_s = ccsd_t.get("two_s") if ccsd_t and ccsd_t.get("converged") else None
        ccsd_t_two_sz_fe1 = ccsd_t.get("two_sz_fe1") if ccsd_t and ccsd_t.get("converged") else None
        ccsd_t_two_sz_fe2 = ccsd_t.get("two_sz_fe2") if ccsd_t and ccsd_t.get("converged") else None
        ccsdt_energy = ccsdt.get("energy") if ccsdt and ccsdt.get("converged") else None
        ccsdt_s_squared = ccsdt.get("s_squared") if ccsdt and ccsdt.get("converged") else None
        ccsdt_two_s = ccsdt.get("two_s") if ccsdt and ccsdt.get("converged") else None
        ccsdt_two_sz_fe1 = ccsdt.get("two_sz_fe1") if ccsdt and ccsdt.get("converged") else None
        ccsdt_two_sz_fe2 = ccsdt.get("two_sz_fe2") if ccsdt and ccsdt.get("converged") else None
        dmrg_energy = dmrg.get("energy") if dmrg else None
        dmrg_uncertainty = dmrg.get("uncertainty") if dmrg else None
        dmrg_source_mode = dmrg.get("source_mode") if dmrg else None
        composite_energy = composite.get("energy") if composite else None
        composite_uncertainty = composite.get("uncertainty") if composite else None
        composite_freeze_occ = composite.get("freeze_occ") if composite else None

        dmrg_is_ranking_eligible = dmrg_energy is not None and _dmrg_source_mode_allows_ranking(dmrg_source_mode)

        (
            ranking_method,
            ranking_energy,
            ranking_uncertainty,
            consensus_energy,
            consensus_uncertainty,
        ) = _resolve_ranking_result(
            ccsd_t_energy=ccsd_t_energy,
            ccsdt_energy=ccsdt_energy,
            dmrg_energy=dmrg_energy,
            dmrg_uncertainty=dmrg_uncertainty,
            composite_energy=composite_energy,
            composite_uncertainty=composite_uncertainty,
            dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
        )
        composite_dmrg_consensus_energy, composite_dmrg_consensus_uncertainty = _build_composite_dmrg_consensus(
            composite_energy,
            composite_uncertainty,
            dmrg_energy,
            dmrg_uncertainty,
            dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
        )

        rows.append(
            {
                "label": label,
                "display_label": display_label,
                "family": family,
                "uhf_energy": uhf_energy,
                "uhf_converged": uhf_converged,
                "uhf_s_squared": uhf_s_squared,
                "uhf_two_s": uhf_two_s,
                "uhf_two_sz_fe1": uhf_two_sz_fe1,
                "uhf_two_sz_fe2": uhf_two_sz_fe2,
                "ccsd_energy": ccsd_energy,
                "ccsd_correlation_energy": ccsd_corr_energy,
                "ccsd_s_squared": ccsd_s_squared,
                "ccsd_two_s": ccsd_two_s,
                "ccsd_two_sz_fe1": ccsd_two_sz_fe1,
                "ccsd_two_sz_fe2": ccsd_two_sz_fe2,
                "ccsd_t_energy": ccsd_t_energy,
                "ccsd_t_s_squared": ccsd_t_s_squared,
                "ccsd_t_two_s": ccsd_t_two_s,
                "ccsd_t_two_sz_fe1": ccsd_t_two_sz_fe1,
                "ccsd_t_two_sz_fe2": ccsd_t_two_sz_fe2,
                "ccsdt_energy": ccsdt_energy,
                "ccsdt_s_squared": ccsdt_s_squared,
                "ccsdt_two_s": ccsdt_two_s,
                "ccsdt_two_sz_fe1": ccsdt_two_sz_fe1,
                "ccsdt_two_sz_fe2": ccsdt_two_sz_fe2,
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
    *,
    fcidump_ecore: float | None = None,
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
        uhf_observables = _observable_fields(best_uhf, "uhf_")
        ccsd_energy = best_ccsd.get("ccsd_energy") if best_ccsd else None
        ccsd_corr_energy = best_ccsd.get("ccsd_correlation_energy") if best_ccsd else None
        ccsd_observables = _observable_fields(best_ccsd, "ccsd_")
        ccsd_t_energy = best_ccsd_t.get("ccsd_t_energy") if best_ccsd_t else None
        ccsd_t_observables = _observable_fields(best_ccsd_t, "ccsd_t_")
        ccsdt_energy = best_ccsdt.get("ccsdt_energy") if best_ccsdt else None
        ccsdt_observables = _observable_fields(best_ccsdt, "ccsdt_")
        dmrg_energy = best_dmrg.get("dmrg_extrapolated_energy") if best_dmrg else None
        dmrg_uncertainty = best_dmrg.get("dmrg_uncertainty") if best_dmrg else None
        dmrg_source_mode = best_dmrg.get("dmrg_source_mode") if best_dmrg else None
        dmrg_bond_dims = best_dmrg.get("dmrg_bond_dims") if best_dmrg else []
        composite_energy = best_composite.get("cc_composite_energy") if best_composite else None
        composite_uncertainty = best_composite.get("cc_composite_uncertainty") if best_composite else None
        composite_freeze_occ = best_composite.get("cc_composite_freeze_occ") if best_composite else None
        family = _best_family(*state_rows)
        source_initial_labels = sorted({row["label"] for row in state_rows if row.get("label")})

        dmrg_is_ranking_eligible = dmrg_energy is not None and _dmrg_source_mode_allows_ranking(dmrg_source_mode)

        (
            ranking_method,
            ranking_energy,
            ranking_uncertainty,
            consensus_energy,
            consensus_uncertainty,
        ) = _resolve_ranking_result(
            ccsd_t_energy=ccsd_t_energy,
            ccsdt_energy=ccsdt_energy,
            dmrg_energy=dmrg_energy,
            dmrg_uncertainty=dmrg_uncertainty,
            composite_energy=composite_energy,
            composite_uncertainty=composite_uncertainty,
            dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
        )
        composite_dmrg_consensus_energy, composite_dmrg_consensus_uncertainty = _build_composite_dmrg_consensus(
            composite_energy,
            composite_uncertainty,
            dmrg_energy,
            dmrg_uncertainty,
            dmrg_is_ranking_eligible=dmrg_is_ranking_eligible,
        )

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
                "uhf_two_s": uhf_observables["two_s"],
                "uhf_two_sz_fe1": uhf_observables["two_sz_fe1"],
                "uhf_two_sz_fe2": uhf_observables["two_sz_fe2"],
                "ccsd_energy": ccsd_energy,
                "ccsd_correlation_energy": ccsd_corr_energy,
                "ccsd_s_squared": ccsd_observables["s_squared"],
                "ccsd_two_s": ccsd_observables["two_s"],
                "ccsd_two_sz_fe1": ccsd_observables["two_sz_fe1"],
                "ccsd_two_sz_fe2": ccsd_observables["two_sz_fe2"],
                "ccsd_t_energy": ccsd_t_energy,
                "ccsd_t_s_squared": ccsd_t_observables["s_squared"],
                "ccsd_t_two_s": ccsd_t_observables["two_s"],
                "ccsd_t_two_sz_fe1": ccsd_t_observables["two_sz_fe1"],
                "ccsd_t_two_sz_fe2": ccsd_t_observables["two_sz_fe2"],
                "ccsdt_energy": ccsdt_energy,
                "ccsdt_s_squared": ccsdt_observables["s_squared"],
                "ccsdt_two_s": ccsdt_observables["two_s"],
                "ccsdt_two_sz_fe1": ccsdt_observables["two_sz_fe1"],
                "ccsdt_two_sz_fe2": ccsdt_observables["two_sz_fe2"],
                "cc_composite_energy": composite_energy,
                "cc_composite_uncertainty": composite_uncertainty,
                "cc_composite_freeze_occ": composite_freeze_occ,
                "dmrg_extrapolated_energy": dmrg_energy,
                "dmrg_uncertainty": dmrg_uncertainty,
                "dmrg_source_mode": dmrg_source_mode,
                "dmrg_ranking_eligible": dmrg_is_ranking_eligible,
                "dmrg_bond_dims": dmrg_bond_dims or [],
                "e_core": fcidump_ecore,
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


def _fmt_uncertainty(value) -> str:
    return f"{value:.2e}" if value is not None else "N/A"


def _csv_display_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ";".join(_csv_display_value(item) for item in value)
    return str(value)


def _state_column_name(row: dict) -> str:
    rank = row.get("rank", "?")
    label = row.get("display_label", row.get("label", f"state_{rank}"))
    return f"rank{rank}:{label}"


def _render_csv(rows: list[dict]) -> str:
    fieldnames = ["metric", "description"] + [_state_column_name(row) for row in rows]
    metrics = _csv_metric_definitions()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for metric, description in metrics:
        record = {"metric": metric, "description": description}
        for row in rows:
            record[_state_column_name(row)] = _csv_display_value(row.get(metric))
        writer.writerow(record)
    return buffer.getvalue()


def _csv_metric_definitions() -> list[tuple[str, str]]:
    return [
        ("rank", "Final ranking position"),
        ("family", "Spin-isomer family"),
        ("representative_label", "Representative internal label used for ranking"),
        ("source_initial_count", "Number of source initial guesses merged into this final state"),
        ("source_initial_labels", "Semicolon-separated source initial labels"),
        ("ranking_method", "Highest-level method used for final ranking"),
        ("ranking_energy", "Final ranking energy (Ha)"),
        ("ranking_uncertainty", "Final ranking uncertainty (Ha)"),
        ("e_core", "FCIDUMP core energy used by downstream active-space solvers (Ha)"),
        ("uhf_energy", "UHF total energy (Ha)"),
        ("uhf_converged", "Whether UHF converged"),
        ("uhf_s_squared", "UHF <S^2>"),
        ("uhf_two_s", "UHF derived 2S"),
        ("uhf_two_sz_fe1", "UHF 2Sz on Fe1"),
        ("uhf_two_sz_fe2", "UHF 2Sz on Fe2"),
        ("ccsd_energy", "CCSD total energy (Ha)"),
        ("ccsd_correlation_energy", "CCSD correlation energy (Ha)"),
        ("ccsd_s_squared", "CCSD <S^2>"),
        ("ccsd_two_s", "CCSD derived 2S"),
        ("ccsd_two_sz_fe1", "CCSD 2Sz on Fe1"),
        ("ccsd_two_sz_fe2", "CCSD 2Sz on Fe2"),
        ("ccsd_t_energy", "CCSD(T) total energy (Ha)"),
        ("ccsd_t_s_squared", "CCSD(T) <S^2>"),
        ("ccsd_t_two_s", "CCSD(T) derived 2S"),
        ("ccsd_t_two_sz_fe1", "CCSD(T) 2Sz on Fe1"),
        ("ccsd_t_two_sz_fe2", "CCSD(T) 2Sz on Fe2"),
        ("ccsdt_energy", "CCSDT total energy (Ha)"),
        ("ccsdt_s_squared", "CCSDT <S^2>"),
        ("ccsdt_two_s", "CCSDT derived 2S"),
        ("ccsdt_two_sz_fe1", "CCSDT 2Sz on Fe1"),
        ("ccsdt_two_sz_fe2", "CCSDT 2Sz on Fe2"),
        ("cc_composite_energy", "CC composite total energy (Ha)"),
        ("cc_composite_uncertainty", "CC composite uncertainty (Ha)"),
        ("cc_composite_freeze_occ", "CC composite freeze_occ setting"),
        ("dmrg_extrapolated_energy", "Extrapolated DMRG energy (Ha)"),
        ("dmrg_uncertainty", "Extrapolated DMRG uncertainty (Ha)"),
        ("dmrg_source_mode", "DMRG source-mode used in ranking"),
        ("dmrg_ranking_eligible", "Whether DMRG result is ranking-eligible"),
        ("dmrg_bond_dims", "Semicolon-separated DMRG bond dimensions"),
        ("consensus_energy", "CCSDT+DMRG consensus energy (Ha)"),
        ("consensus_uncertainty", "CCSDT+DMRG consensus uncertainty (Ha)"),
        ("composite_dmrg_consensus_energy", "CC composite + DMRG consensus energy (Ha)"),
        ("composite_dmrg_consensus_uncertainty", "CC composite + DMRG consensus uncertainty (Ha)"),
    ]


def _render_csv_subset(
    rows: list[dict],
    metrics: list[tuple[str, str]],
) -> str:
    fieldnames = ["metric", "description"] + [_state_column_name(row) for row in rows]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for metric, description in metrics:
        record = {"metric": metric, "description": description}
        for row in rows:
            record[_state_column_name(row)] = _csv_display_value(row.get(metric))
        writer.writerow(record)
    return buffer.getvalue()


def _render_csv_sections(rows: list[dict]) -> dict[str, str]:
    metric_map = dict(_csv_metric_definitions())

    def pick(keys: list[str]) -> list[tuple[str, str]]:
        return [(key, metric_map[key]) for key in keys]

    return {
        "final_report_energies.csv": _render_csv_subset(
            rows,
            pick(
                [
                    "rank",
                    "family",
                    "representative_label",
                    "source_initial_count",
                    "source_initial_labels",
                    "ranking_method",
                    "ranking_energy",
                    "ranking_uncertainty",
                    "e_core",
                    "uhf_energy",
                    "ccsd_energy",
                    "ccsd_correlation_energy",
                    "ccsd_t_energy",
                    "ccsdt_energy",
                    "cc_composite_energy",
                    "cc_composite_uncertainty",
                    "cc_composite_freeze_occ",
                    "dmrg_extrapolated_energy",
                    "dmrg_uncertainty",
                    "ranking_energy",
                    "ranking_uncertainty",
                ]
            ),
        ),
        "final_report_observables.csv": _render_csv_subset(
            rows,
            pick(
                [
                    "rank",
                    "family",
                    "representative_label",
                    "uhf_converged",
                    "uhf_s_squared",
                    "uhf_two_s",
                    "uhf_two_sz_fe1",
                    "uhf_two_sz_fe2",
                    "ccsd_s_squared",
                    "ccsd_two_s",
                    "ccsd_two_sz_fe1",
                    "ccsd_two_sz_fe2",
                    "ccsd_t_s_squared",
                    "ccsd_t_two_s",
                    "ccsd_t_two_sz_fe1",
                    "ccsd_t_two_sz_fe2",
                    "ccsdt_s_squared",
                    "ccsdt_two_s",
                    "ccsdt_two_sz_fe1",
                    "ccsdt_two_sz_fe2",
                ]
            ),
        ),
    }


def _render_markdown(rows: list[dict], *, report_scope: str, benchmark_note: str | None = None) -> str:
    lines = [
        "# APEX Final Summary",
        "",
        f"Generated: {_datetime.now().isoformat()}",
        "",
        f"Report scope: `{report_scope}`",
        "",
        "Ranking priority: `CC composite + DMRG consensus` > `CC composite` > `CCSDT + DMRG consensus` > `DMRG extrapolated` > `CCSDT` > `CCSD(T)`",
        "",
        "Consensus definition: the `CCSDT + DMRG consensus` energy is the mean of `CCSDT` and extrapolated `DMRG`; uncertainty is `max(DMRG uncertainty, |CCSDT-DMRG|/2)`.",
        "",
        "Only `DMRG` entries whose source mode remains ranking-eligible are allowed to influence ranking directly. Fallback extrapolations are retained for reference/provenance but do not outrank converged CC results.",
        "",
        "Benchmark note: when a Chan-reference benchmark bundle is available, cross-check this report against the `chan_ref/` tables for the same case.",
        "",
        "| Rank | State | Representative Label | Source Guess Count | Family | Ranking Method | Ranking Energy (Ha) | Ranking Uncertainty | CCSD(T) | CCSDT | CC Composite | DMRG Extrap. | Consensus |",
        "|------|-------|----------------------|--------------------|--------|----------------|---------------------|---------------------|---------|-------|--------------|--------------|-----------|",
    ]
    if benchmark_note:
        lines.extend([benchmark_note, ""])
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
        lines.append(
            f"  - UHF: `{_fmt_energy(row.get('uhf_energy'))}`  "
            f"(<S^2> = `{_fmt_energy(row.get('uhf_s_squared'))}`, 2S = `{_fmt_energy(row.get('uhf_two_s'))}`, "
            f"2Sz[Fe1] = `{_fmt_energy(row.get('uhf_two_sz_fe1'))}`, 2Sz[Fe2] = `{_fmt_energy(row.get('uhf_two_sz_fe2'))}`)"
        )
        lines.append(
            f"  - CCSD: `{_fmt_energy(row.get('ccsd_energy'))}`  "
            f"(corr = `{_fmt_energy(row.get('ccsd_correlation_energy'))}`, <S^2> = `{_fmt_energy(row.get('ccsd_s_squared'))}`, "
            f"2S = `{_fmt_energy(row.get('ccsd_two_s'))}`, 2Sz[Fe1] = `{_fmt_energy(row.get('ccsd_two_sz_fe1'))}`, "
            f"2Sz[Fe2] = `{_fmt_energy(row.get('ccsd_two_sz_fe2'))}`)"
        )
        lines.append(
            f"  - CCSD(T): `{_fmt_energy(row.get('ccsd_t_energy'))}`  "
            f"(<S^2> = `{_fmt_energy(row.get('ccsd_t_s_squared'))}`, 2S = `{_fmt_energy(row.get('ccsd_t_two_s'))}`, "
            f"2Sz[Fe1] = `{_fmt_energy(row.get('ccsd_t_two_sz_fe1'))}`, 2Sz[Fe2] = `{_fmt_energy(row.get('ccsd_t_two_sz_fe2'))}`)"
        )
        lines.append(
            f"  - CCSDT: `{_fmt_energy(row.get('ccsdt_energy'))}`  "
            f"(<S^2> = `{_fmt_energy(row.get('ccsdt_s_squared'))}`, 2S = `{_fmt_energy(row.get('ccsdt_two_s'))}`, "
            f"2Sz[Fe1] = `{_fmt_energy(row.get('ccsdt_two_sz_fe1'))}`, 2Sz[Fe2] = `{_fmt_energy(row.get('ccsdt_two_sz_fe2'))}`)"
        )
        lines.append(f"  - E_core: `{_fmt_energy(row.get('e_core'))}`")
        lines.append(f"  - DMRG extrapolated: `{_fmt_energy(row.get('dmrg_extrapolated_energy'))}`")
        lines.append(f"  - CC composite: `{_fmt_energy(row.get('cc_composite_energy'))}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def step_report(session_dir: str):
    """Build a final multi-method summary from CC and DMRG stages."""
    sm = _SessionManager(session_dir)
    sm.require_previous("step10_report", "step9_extrapolate")

    uhf_summary = sm.load_step_summary("step3_uhf", "uhf_summary.json")
    ccsd_summary = sm.load_step_summary("step4_ccsd", "ccsd_summary.json")
    ccsd_t_summary = sm.load_step_summary("step5_ccsd_t", "ccsd_t_summary.json")
    ccsdt_summary = sm.load_step_summary("step6_ccsdt", "ccsdt_summary.json")
    dmrg_extrapolation_summary = sm.load_step_summary("step9_extrapolate", "dmrg_extrapolation_summary.json")
    cc_composite_summary = []
    composite_path = os.path.join(sm.session_dir, "step12_cc_composite", "cc_composite_summary.json")
    if os.path.exists(composite_path):
        cc_composite_summary = sm.load_cc_composite_summary()

    print("=" * 60)
    print("Step 10: Final summary/report")
    print("=" * 60)
    fcidump_ecore = _load_fcidump_ecore(sm.session_dir)

    rows = _build_final_rows(
        uhf_summary,
        ccsd_summary,
        ccsd_t_summary,
        ccsdt_summary,
        dmrg_extrapolation_summary,
        fcidump_ecore=fcidump_ecore,
        cc_composite_summary=cc_composite_summary,
    )
    csv_sections = _render_csv_sections(rows)

    for row in rows[:10]:
        print(
            f"  #{row['rank']:3d}  {row.get('display_label', row['label']):<40}  "
            f"{row['ranking_method']:<22}  E={_fmt_energy(row['ranking_energy'])}"
        )

    sm._save_final_summary(rows, extra_text_files=csv_sections)
    for legacy_name in [
        "final_report.md",
        "final_report.csv",
        "final_report_ranking.csv",
        "final_report_dmrg.csv",
    ]:
        legacy_path = os.path.join(sm.session_dir, "step10_report", legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
    _cleanup_step_selection_artifacts(os.path.join(sm.session_dir, "step10_report"))
    print(f"Step 10 complete. Final summary saved for {len(rows)} configurations.")
