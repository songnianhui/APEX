"""Helpers for per-step selection guides and pick-file artifacts.

This module provides shared formatting/writing utilities used by the canonical
Step 3-10 orchestrators when they emit reviewable summaries and reusable pick
artifacts. All helpers here are workflow-internal shared utilities.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def _fmt_value(key: str, value) -> str:
    if value is None:
        return "N/A"
    if key in {
        "energy",
        "ranking_energy",
        "ccsd_t_energy",
        "ccsdt_energy",
        "cc_composite_energy",
        "dmrg_extrapolated_energy",
        "consensus_energy",
        "composite_dmrg_consensus_energy",
    }:
        return f"{float(value):.10f}"
    if key in {
        "uncertainty",
        "dmrg_uncertainty",
        "consensus_uncertainty",
        "ranking_uncertainty",
        "cc_composite_uncertainty",
        "composite_dmrg_consensus_uncertainty",
    }:
        return f"{float(value):.2e}"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _fmt_markdown_value(key: str, value) -> str:
    """Render values safely inside markdown tables."""
    rendered = _fmt_value(key, value)
    return rendered.replace("|", "\\|")


def _display_name(key: str) -> str:
    aliases = {
        "display_label": "State",
        "label": "Label",
        "family": "Family",
        "energy": "Energy",
        "converged": "Converged",
        "s_squared": "<S^2>",
        "oxidation": "Oxidation",
        "config_id": "Config ID",
        "fno_scheme": "FNO Scheme",
        "freeze_occ": "Freeze Occ",
        "kept_occ_alpha": "Kept Occ α",
        "kept_occ_beta": "Kept Occ β",
        "localization_method": "Localization",
        "source_method": "Source Method",
        "nocc_alpha": "Nocc α",
        "nocc_beta": "Nocc β",
        "bond_dim": "Bond Dim",
        "bond_dims": "Bond Dims",
        "uncertainty": "Uncertainty",
        "ranking_method": "Ranking Method",
        "ranking_energy": "Ranking Energy",
        "ranking_uncertainty": "Ranking Uncertainty",
        "ccsd_t_energy": "CCSD(T)",
        "ccsdt_energy": "CCSDT",
        "cc_composite_energy": "CC Composite",
        "cc_composite_uncertainty": "CC Composite Unc.",
        "cc_composite_freeze_occ": "CC Composite Freeze",
        "dmrg_extrapolated_energy": "DMRG Extrap.",
        "dmrg_uncertainty": "DMRG Unc.",
        "consensus_energy": "Consensus",
        "consensus_uncertainty": "Consensus Unc.",
        "composite_dmrg_consensus_energy": "CC+DMRG Consensus",
        "composite_dmrg_consensus_uncertainty": "CC+DMRG Consensus Unc.",
    }
    return aliases.get(key, key.replace("_", " ").title())


def _has_display_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _choose_preview_columns(summary: list[dict]) -> list[str]:
    base = []
    if any(_has_display_value(row.get("display_label")) for row in summary):
        base.append("display_label")
    base.extend(["label", "family"])
    candidates = [
        "energy",
        "ranking_method",
        "ranking_energy",
        "ranking_uncertainty",
        "uncertainty",
        "converged",
        "s_squared",
        "oxidation",
        "fno_scheme",
        "freeze_occ",
        "kept_occ_alpha",
        "kept_occ_beta",
        "bond_dim",
        "bond_dims",
        "localization_method",
        "source_method",
        "ccsd_t_energy",
        "ccsdt_energy",
        "cc_composite_energy",
        "cc_composite_uncertainty",
        "cc_composite_freeze_occ",
        "dmrg_extrapolated_energy",
        "dmrg_uncertainty",
        "consensus_energy",
        "consensus_uncertainty",
        "composite_dmrg_consensus_energy",
        "composite_dmrg_consensus_uncertainty",
        "config_id",
    ]
    present = []
    for key in candidates:
        if any(_has_display_value(row.get(key)) for row in summary):
            present.append(key)
    return base + present


def _build_display_label_map(summary: list[dict]) -> dict[str, str]:
    """Map machine labels to the best available user-facing display label."""
    mapping: dict[str, str] = {}
    for row in summary:
        label = row.get("label")
        if not label:
            continue
        display_label = (
            row.get("display_label")
            or row.get("final_state_signature")
            or label
        )
        mapping[label] = display_label
    return mapping


def _attach_display_labels(summary: list[dict], upstream_summary: list[dict] | None) -> list[dict]:
    """Propagate display labels from an upstream step while preserving machine labels."""
    display_map = _build_display_label_map(upstream_summary or [])
    for row in summary:
        label = row.get("label")
        if not label:
            continue
        current = row.get("display_label")
        final_sig = row.get("final_state_signature")
        if final_sig and (not current or current == label):
            row["display_label"] = final_sig
        else:
            row["display_label"] = current or display_map.get(label, label)
    return summary


def _write_selection_artifacts(
    step_dir: str,
    *,
    step_name: str,
    next_step_name: str | None,
    summary: list[dict],
    stats: dict | None = None,
    max_preview: int = 20,
    keep_default: str = "",
    write_json_artifacts: bool = False,
):
    """Write standardized selection outputs for the current step.

    Files written:
    - ``selection_candidates.csv``: spreadsheet-friendly candidate table
    - ``selection_worklist.csv``: editable CSV with a `keep` column for manual picks
    - ``selection_guide.md``: human-readable preview + import instructions
    """
    os.makedirs(step_dir, exist_ok=True)

    ordered = list(summary)
    labels = [row["label"] for row in ordered if row.get("label")]
    preview_columns = _choose_preview_columns(ordered) if ordered else ["label", "family"]
    csv_role = f"{step_name} candidate table"
    csv_comment = (
        "Spreadsheet-friendly view of the current step results. "
        "Comment lines beginning with '#' are informational and can be ignored by parsers."
    )
    worklist_role = f"{step_name} editable worklist"
    worklist_comment = (
        "Editable selection worklist. Change only the 'keep' column to choose which rows enter the next step."
    )

    if write_json_artifacts:
        with open(os.path.join(step_dir, "selection_candidates.json"), "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
        with open(os.path.join(step_dir, "pick_labels_all.json"), "w", encoding="utf-8") as f:
            json.dump({"labels": labels}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(step_dir, "pick_labels_template.json"), "w", encoding="utf-8") as f:
            json.dump({"labels": labels[: min(5, len(labels))]}, f, indent=2, ensure_ascii=False)
    else:
        for stale in (
            "selection_candidates.json",
            "pick_labels_all.json",
            "pick_labels_template.json",
        ):
            path = os.path.join(step_dir, stale)
            if os.path.exists(path):
                os.remove(path)

    with open(os.path.join(step_dir, "selection_candidates.csv"), "w", newline="") as f:
        f.write(f"# File role: {csv_role}\n")
        f.write(f"# {csv_comment}\n")
        writer = csv.DictWriter(f, fieldnames=preview_columns)
        writer.writeheader()
        for row in ordered:
            writer.writerow({col: _fmt_value(col, row.get(col)) for col in preview_columns})

    worklist_columns = ["keep"] + preview_columns
    with open(os.path.join(step_dir, "selection_worklist.csv"), "w", newline="") as f:
        f.write(f"# File role: {worklist_role}\n")
        f.write(f"# {worklist_comment}\n")
        writer = csv.DictWriter(f, fieldnames=worklist_columns)
        writer.writeheader()
        for row in ordered:
            payload = {"keep": keep_default}
            payload.update({col: _fmt_value(col, row.get(col)) for col in preview_columns})
            writer.writerow(payload)

    lines = [
        f"# {step_name} Selection Guide",
        "",
        "> File role: human-readable guide for this step's candidate table and next-step import workflow.",
        "",
        f"Total candidates: {len(ordered)}",
        "",
        "The full candidate table is also available in:",
        "",
        "`selection_candidates.csv`",
        "",
        "The editable spreadsheet worklist is:",
        "",
        "`selection_worklist.csv`",
    ]
    if stats:
        stat_order = [
            ("family_scheme", "Family scheme"),
            ("benchmark_profile", "Benchmark profile"),
            ("config_reduction_mode", "Config reduction mode"),
            ("raw_spin_patterns", "Raw spin patterns"),
            ("spin_families", "Spin families"),
            ("spin_x_oxidation", "Spin x oxidation"),
            ("spin_x_oxidation_x_d_before_reduction", "Spin x oxidation x d"),
            ("total_configs_after_reduction", "Total configs (saved)"),
        ]
        lines.extend(["", "## Enumeration Layers", ""])
        for key, title in stat_order:
            if key in stats:
                lines.append(f"- {title}: {stats[key]}")
    if next_step_name:
        lines.extend(
            [
                f"Next step: {next_step_name}",
                "",
                "To import a user-selected subset into the next step, mark rows in `selection_worklist.csv` by setting `keep` to one of:",
                "",
                "`1`, `true`, `yes`, `keep`, `select`, `include`, `run`",
                "",
                "Then run the next step with:",
                "",
                f'```bash\napex-filter {next_step_name} --session <session_dir> --pick "file /path/to/selection_worklist.csv"\n```',
            ]
        )

    if ordered:
        header = " | ".join(["#"] + [_display_name(col) for col in preview_columns])
        divider = " | ".join(["---"] * (len(preview_columns) + 1))
        lines.extend(
            [
                "",
                "## Preview",
                "",
                f"| {header} |",
                f"| {divider} |",
            ]
        )
        for idx, row in enumerate(ordered[:max_preview], 1):
            rendered = [_fmt_markdown_value(col, row.get(col)) for col in preview_columns]
            lines.append(f"| {idx} | " + " | ".join(rendered) + " |")

    with open(os.path.join(step_dir, "selection_guide.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
