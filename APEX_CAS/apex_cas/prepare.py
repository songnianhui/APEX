"""Preparation helpers for explicit cluster annotations.

This module builds a user-facing draft of ``cluster_info`` before SCF/CAS
steps.  The goal is to make the structure interpretation explicit and editable:

* generate contiguous user labels (e.g. bridge S1/S2, terminal S3-S6)
* emit a full per-atom CSV draft
* render a labeled structure PNG for visual checking
* validate an edited draft CSV and finalize it into ``cluster_info.yaml``
"""

from __future__ import annotations

import csv
import os
from collections import Counter as _Counter
from dataclasses import dataclass as _dataclass
from typing import Iterable as _Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from shared.structure_parser import _build_connectivity, parse_structure as _parse_structure


ROLE_PRIORITY = {
    "metal": 0,
    "bridging": 1,
    "terminal": 2,
    "interstitial": 3,
    "spectator": 4,
}

ROLE_COLORS = {
    "metal": "#c0392b",
    "bridging": "#2980b9",
    "terminal": "#27ae60",
    "interstitial": "#8e44ad",
    "spectator": "#7f8c8d",
}

DEFAULT_PROJECTION_ROLE = {
    "metal": "metal_df",
    "bridging": "bridging_p",
    "terminal": "exclude",
    "interstitial": "exclude",
    "spectator": "exclude",
}

VALID_ROLES = {"metal", "bridging", "terminal", "interstitial", "spectator"}
VALID_PROJECTION_ROLES = {"metal_df", "bridging_p", "exclude"}


@_dataclass
class _PreparedAtom:
    atom_index: int
    xyz_serial: int
    element: str
    user_label: str
    role: str
    charge: int
    projection_role: str
    bound_to: str = ""
    bridging_to: str = ""
    ligand_type: str = ""
    display_contacts: str = ""
    neighbor_elements: str = ""
    auto_reason: str = ""
    note: str = ""


def _prepare_cluster_inputs(
    structure_path: str,
    *,
    case_dir: str,
    charge: int,
    target_spin: float,
    symmetry_group_override: str | None,
    reduction_symmetry_override: str | None,
    symmetry_detection_mode: str,
    family_scheme: str,
    benchmark_profile: str,
    config_reduction_mode: str,
    force: bool = False,
) -> dict:
    """Generate draft cluster-info artifacts for a structure."""
    cluster_info = _parse_structure(
        structure_path,
        charge=charge,
        target_spin=target_spin,
        cluster_info_path=None,
        symmetry_group_override=symmetry_group_override,
        reduction_symmetry_override=reduction_symmetry_override,
        symmetry_detection_mode=symmetry_detection_mode,
        family_scheme=family_scheme,
        benchmark_profile=benchmark_profile,
        config_reduction_mode=config_reduction_mode,
    )

    inputs_dir = os.path.join(case_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    prepared_atoms = _build_prepared_atoms(cluster_info)
    stem = os.path.splitext(os.path.basename(structure_path))[0]

    draft_csv = os.path.join(inputs_dir, f"{stem}_cluster_info_draft.csv")
    final_yaml = os.path.join(inputs_dir, f"{stem}_cluster_info.yaml")
    labeled_png = os.path.join(inputs_dir, f"{stem}_structure_labeled.png")

    _write_cluster_info_draft_csv(draft_csv, prepared_atoms)
    _write_labeled_structure_png(
        labeled_png,
        cluster_info.all_elements,
        np.asarray(cluster_info.all_positions, dtype=float),
        prepared_atoms,
    )
    return {
        "cluster_info": cluster_info,
        "prepared_atoms": prepared_atoms,
        "draft_csv": draft_csv,
        "final_yaml": final_yaml,
        "labeled_png": labeled_png,
        "cluster_info_exists": os.path.exists(final_yaml),
    }


def _finalize_cluster_info_draft(
    structure_path: str,
    *,
    case_dir: str,
    charge: int,
    target_spin: float,
    symmetry_group_override: str | None,
    reduction_symmetry_override: str | None,
    symmetry_detection_mode: str,
    family_scheme: str,
    benchmark_profile: str,
    config_reduction_mode: str,
    draft_csv_path: str | None = None,
    force: bool = False,
) -> dict:
    """Validate an edited draft CSV and write the authoritative cluster_info.yaml."""
    cluster_info = _parse_structure(
        structure_path,
        charge=charge,
        target_spin=target_spin,
        cluster_info_path=None,
        symmetry_group_override=symmetry_group_override,
        reduction_symmetry_override=reduction_symmetry_override,
        symmetry_detection_mode=symmetry_detection_mode,
        family_scheme=family_scheme,
        benchmark_profile=benchmark_profile,
        config_reduction_mode=config_reduction_mode,
    )

    inputs_dir = os.path.join(case_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(structure_path))[0]

    draft_csv = draft_csv_path or os.path.join(inputs_dir, f"{stem}_cluster_info_draft.csv")
    final_yaml = os.path.join(inputs_dir, f"{stem}_cluster_info.yaml")
    if not os.path.exists(draft_csv):
        raise FileNotFoundError(f"Cluster-info draft CSV not found: {draft_csv}")
    if os.path.exists(final_yaml) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing cluster_info.yaml without --force: {final_yaml}"
        )

    prepared_atoms = _load_validated_prepared_atoms_from_csv(
        draft_csv,
        elements=list(cluster_info.all_elements),
    )
    _write_cluster_info_yaml(final_yaml, cluster_info, prepared_atoms)
    return {
        "cluster_info": cluster_info,
        "prepared_atoms": prepared_atoms,
        "draft_csv": draft_csv,
        "final_yaml": final_yaml,
        "atom_count": len(prepared_atoms),
    }


def _build_prepared_atoms(cluster_info) -> list[_PreparedAtom]:
    elements = list(cluster_info.all_elements)
    positions = np.asarray(cluster_info.all_positions, dtype=float)
    connectivity = _build_connectivity(elements, positions)

    metal_info = {m.index: m for m in cluster_info.metals}
    bridge_info = {b.index: b for b in cluster_info.bridging_atoms}
    terminal_info = {t.donor_atom_index: t for t in cluster_info.terminal_ligands}

    role_by_index = {}
    for idx in range(len(elements)):
        if idx in metal_info:
            role_by_index[idx] = "metal"
        elif idx in bridge_info:
            role_by_index[idx] = bridge_info[idx].role or "bridging"
        elif idx in terminal_info:
            role_by_index[idx] = "terminal"
        else:
            role_by_index[idx] = "spectator"

    label_by_index = _assign_user_labels(elements, role_by_index)
    metal_label_by_index = {idx: label_by_index[idx] for idx in metal_info}

    prepared = []
    for idx, elem in enumerate(elements):
        role = role_by_index[idx]
        charge, ligand_type = _infer_charge_and_ligand_type(
            idx,
            elem,
            role,
            connectivity,
            elements,
            metal_info=metal_info,
            bridge_info=bridge_info,
            terminal_info=terminal_info,
        )
        bound_to = ""
        bridging_to = ""
        if idx in bridge_info:
            refs = [cluster_info.metals[m_idx].index for m_idx in bridge_info[idx].bridged_metals]
            labels = [metal_label_by_index[ref] for ref in refs]
            bridging_to = ",".join(labels)
        elif idx in terminal_info:
            metal_ref = cluster_info.metals[terminal_info[idx].metal_index].index
            bound_to = metal_label_by_index[metal_ref]

        bonded_neighbors = _display_neighbor_indices(
            idx,
            role,
            connectivity=connectivity,
            metal_info=metal_info,
            bridge_info=bridge_info,
            terminal_info=terminal_info,
            cluster_info=cluster_info,
            role_by_index=role_by_index,
            label_by_index=label_by_index,
        )
        bonded_to_labels = ",".join(label_by_index[j] for j in bonded_neighbors)
        neighbor_elements = _format_neighbor_elements(elements[j] for j in bonded_neighbors)
        auto_reason = _build_auto_reason(
            idx,
            elem,
            role,
            bound_to=bound_to,
            bridging_to=bridging_to,
            bonded_neighbors=bonded_neighbors,
            elements=elements,
            metal_indices=set(metal_info),
        )

        prepared.append(
            _PreparedAtom(
                atom_index=idx,
                xyz_serial=idx + 1,
                element=elem,
                user_label=label_by_index[idx],
                role=role,
                charge=charge,
                projection_role=DEFAULT_PROJECTION_ROLE.get(role, "exclude"),
                bound_to=bound_to,
                bridging_to=bridging_to,
                ligand_type=ligand_type,
                display_contacts=bonded_to_labels,
                neighbor_elements=neighbor_elements,
                auto_reason=auto_reason,
                note="auto-draft",
            )
        )
    return prepared


def _assign_user_labels(elements: list[str], role_by_index: dict[int, str]) -> dict[int, str]:
    grouped: dict[str, list[int]] = {}
    for idx, elem in enumerate(elements):
        grouped.setdefault(elem, []).append(idx)

    labels = {}
    for elem, indices in grouped.items():
        ordered = sorted(
            indices,
            key=lambda idx: (
                ROLE_PRIORITY.get(role_by_index[idx], 99),
                idx,
            ),
        )
        for n, idx in enumerate(ordered, 1):
            labels[idx] = f"{elem}{n}"
    return labels


def _infer_charge_and_ligand_type(
    atom_index: int,
    element: str,
    role: str,
    connectivity: dict[int, list[int]],
    elements: list[str],
    *,
    metal_info: dict[int, object],
    bridge_info: dict[int, object],
    terminal_info: dict[int, object],
) -> tuple[int, str]:
    if role == "metal":
        return 0, ""

    neighbors = connectivity.get(atom_index, [])
    non_metal_neighbors = [j for j in neighbors if j not in metal_info]
    has_carbon_neighbor = any(elements[j] == "C" for j in non_metal_neighbors)

    if role in {"bridging", "interstitial"}:
        if element in {"S", "O", "Se"}:
            return -2, {
                "S": "sulfide",
                "O": "oxide",
                "Se": "selenide",
            }[element]
        if element in {"F", "Cl", "Br", "I"}:
            return -1, "halide"
        if element == "H":
            return -1, "hydride"
        if element == "N":
            return -3, "nitride"
        if element == "C":
            return -4, "carbide"
        return 0, ""

    if role == "terminal":
        if element == "S":
            if has_carbon_neighbor:
                return -1, "thiolate"
            return -2, "terminal_sulfide"
        if element == "O":
            if has_carbon_neighbor:
                return -1, "alkoxide"
            return -2, "oxide"
        if element in {"F", "Cl", "Br", "I"}:
            return -1, "halide"
        if element == "H":
            return +1, "proton_like_h"
        if element == "N":
            return 0, "amine_like_n"
        if element == "C":
            return 0, "carbon_donor"
        return 0, ""

    return 0, ""


def _write_cluster_info_draft_csv(path: str, atoms: list[_PreparedAtom]) -> None:
    atoms = _sorted_prepared_atoms(atoms)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# APEX prepare cluster-info draft. Edit the formal fields, review the read-only fields.\n")
        fh.write("# Editable formal fields: user_label, role, charge, projection_role, bound_to, bridging_to, ligand_type\n")
        fh.write("# charge here means the integer charge label used for formal oxidation-state / ligand-charge analysis.\n")
        fh.write("# It is not a Mulliken, Lowdin, Bader, or other density-derived partial charge.\n")
        fh.write("# Valid role values: metal, bridging, terminal, interstitial, spectator\n")
        fh.write("# Valid projection_role values: metal_df, bridging_p, exclude\n")
        fh.write("# bound_to: single metal label such as Fe1. bridging_to: comma-separated metal labels such as \"Fe1,Fe2\".\n")
        fh.write("# Read-only review fields: display_contacts, neighbor_elements, auto_reason, note\n")
        writer = csv.writer(fh)
        writer.writerow(
            [
                "atom_index",
                "xyz_serial",
                "element",
                "user_label",
                "role",
                "charge",
                "projection_role",
                "bound_to",
                "bridging_to",
                "ligand_type",
                "display_contacts",
                "neighbor_elements",
                "auto_reason",
                "note",
            ]
        )
        for atom in atoms:
            writer.writerow(
                [
                    atom.atom_index,
                    atom.xyz_serial,
                    atom.element,
                    atom.user_label,
                    atom.role,
                    atom.charge,
                    atom.projection_role,
                    atom.bound_to,
                    atom.bridging_to,
                    atom.ligand_type,
                    atom.display_contacts,
                    atom.neighbor_elements,
                    atom.auto_reason,
                    atom.note,
                ]
            )


def _write_cluster_info_yaml(path: str, cluster_info, atoms: list[_PreparedAtom]) -> None:
    atoms = _sorted_prepared_atoms(atoms)
    payload = {
        "cluster": {
            "total_charge": int(cluster_info.total_charge),
            "target_spin": float(cluster_info.target_spin),
            "symmetry_group": cluster_info.symmetry_group,
            "reduction_symmetry": cluster_info.reduction_symmetry,
            "family_scheme": cluster_info.family_scheme,
            "benchmark_profile": cluster_info.benchmark_profile,
            "config_reduction_mode": cluster_info.config_reduction_mode,
        },
        "atoms": [],
    }
    for atom in atoms:
        row = {
            "atom_index": int(atom.atom_index),
            "element": atom.element,
            "label": atom.user_label,
            "role": atom.role,
            "charge": int(atom.charge),
            "projection_role": atom.projection_role,
        }
        if atom.bound_to:
            row["bound_to"] = atom.bound_to
        if atom.bridging_to:
            row["bridging_to"] = atom.bridging_to.split(",")
        if atom.ligand_type:
            row["ligand_type"] = atom.ligand_type
        payload["atoms"].append(row)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def _load_validated_prepared_atoms_from_csv(
    path: str,
    *,
    elements: list[str],
) -> list[_PreparedAtom]:
    rows = _read_cluster_info_draft_rows(path)
    if not rows:
        raise ValueError(f"No atom rows found in draft CSV: {path}")

    required = {
        "atom_index",
        "xyz_serial",
        "element",
        "user_label",
        "role",
        "charge",
        "projection_role",
        "bound_to",
        "bridging_to",
        "ligand_type",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(
            f"Draft CSV missing required columns: {', '.join(sorted(missing))}"
        )

    prepared: list[_PreparedAtom] = []
    seen_indices: set[int] = set()
    seen_labels: set[str] = set()

    for row in rows:
        atom_index = _parse_required_int(row, "atom_index")
        xyz_serial = _parse_required_int(row, "xyz_serial")
        element = (row.get("element") or "").strip()
        user_label = (row.get("user_label") or "").strip()
        role = (row.get("role") or "").strip()
        projection_role = (row.get("projection_role") or "").strip()
        bound_to = (row.get("bound_to") or "").strip()
        bridging_to = _normalize_csv_token_list(row.get("bridging_to") or "")
        ligand_type = (row.get("ligand_type") or "").strip()

        if atom_index in seen_indices:
            raise ValueError(f"Duplicate atom_index in draft CSV: {atom_index}")
        if not user_label:
            raise ValueError(f"Missing user_label for atom_index {atom_index}")
        if user_label in seen_labels:
            raise ValueError(f"Duplicate user_label in draft CSV: {user_label}")
        seen_indices.add(atom_index)
        seen_labels.add(user_label)

        if not (0 <= atom_index < len(elements)):
            raise ValueError(
                f"atom_index {atom_index} out of range for structure with {len(elements)} atoms"
            )
        if xyz_serial != atom_index + 1:
            raise ValueError(
                f"xyz_serial mismatch for atom_index {atom_index}: expected {atom_index + 1}, got {xyz_serial}"
            )
        expected_element = elements[atom_index]
        if element != expected_element:
            raise ValueError(
                f"Element mismatch at atom_index {atom_index}: expected {expected_element}, got {element}"
            )
        if role not in VALID_ROLES:
            raise ValueError(
                f"Unsupported role {role!r} for atom_index {atom_index}. "
                f"Valid values: {', '.join(sorted(VALID_ROLES))}"
            )
        if projection_role not in VALID_PROJECTION_ROLES:
            raise ValueError(
                f"Unsupported projection_role {projection_role!r} for atom_index {atom_index}. "
                f"Valid values: {', '.join(sorted(VALID_PROJECTION_ROLES))}"
            )

        prepared.append(
            _PreparedAtom(
                atom_index=atom_index,
                xyz_serial=xyz_serial,
                element=element,
                user_label=user_label,
                role=role,
                charge=_parse_required_int(row, "charge"),
                projection_role=projection_role,
                bound_to=bound_to,
                bridging_to=",".join(bridging_to),
                ligand_type=ligand_type,
                display_contacts=(row.get("display_contacts") or "").strip(),
                neighbor_elements=(row.get("neighbor_elements") or "").strip(),
                auto_reason=(row.get("auto_reason") or "").strip(),
                note=(row.get("note") or "").strip(),
            )
        )

    expected_indices = set(range(len(elements)))
    if seen_indices != expected_indices:
        missing_indices = sorted(expected_indices.difference(seen_indices))
        extra_indices = sorted(seen_indices.difference(expected_indices))
        parts = []
        if missing_indices:
            parts.append(f"missing atom_index rows: {missing_indices}")
        if extra_indices:
            parts.append(f"unexpected atom_index rows: {extra_indices}")
        raise ValueError("Draft CSV does not cover the full structure: " + "; ".join(parts))

    metal_labels = {atom.user_label for atom in prepared if atom.role == "metal"}
    if not metal_labels:
        raise ValueError("Draft CSV must contain at least one metal atom")

    for atom in prepared:
        if atom.bound_to and atom.bound_to not in metal_labels:
            raise ValueError(
                f"atom {atom.user_label} references unknown bound_to metal label {atom.bound_to!r}"
            )
        bridging_labels = [v for v in atom.bridging_to.split(",") if v]
        unknown = [label for label in bridging_labels if label not in metal_labels]
        if unknown:
            raise ValueError(
                f"atom {atom.user_label} references unknown bridging_to metal labels: {', '.join(unknown)}"
            )
        if atom.role == "metal":
            if atom.bound_to or atom.bridging_to:
                raise ValueError(f"metal atom {atom.user_label} must not define bound_to/bridging_to")
        elif atom.role in {"bridging", "interstitial"}:
            if len(bridging_labels) < 2:
                raise ValueError(
                    f"{atom.role} atom {atom.user_label} must define at least two bridging_to labels"
                )
            if atom.bound_to:
                raise ValueError(f"{atom.role} atom {atom.user_label} must not define bound_to")
        elif atom.role == "terminal":
            if not atom.bound_to:
                raise ValueError(f"terminal atom {atom.user_label} must define bound_to")
            if atom.bridging_to:
                raise ValueError(f"terminal atom {atom.user_label} must not define bridging_to")

    return _sorted_prepared_atoms(prepared)


def _read_cluster_info_draft_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return [dict(row) for row in reader]


def _parse_required_int(row: dict[str, str], key: str) -> int:
    raw = (row.get(key) or "").strip()
    if not raw:
        raise ValueError(f"Missing required integer field {key!r}")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for field {key!r}: {raw!r}") from exc


def _normalize_csv_token_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _write_labeled_structure_png(
    path: str,
    elements: list[str],
    positions: np.ndarray,
    atoms: list[_PreparedAtom],
) -> None:
    coords_2d = _project_to_2d(positions)
    connectivity = _build_connectivity(elements, positions)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)

    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j <= i:
                continue
            xi, yi = coords_2d[i]
            xj, yj = coords_2d[j]
            ax.plot([xi, xj], [yi, yj], color="#bdc3c7", linewidth=1.0, zorder=1)

    for atom in atoms:
        x, y = coords_2d[atom.atom_index]
        color = ROLE_COLORS.get(atom.role, "#7f8c8d")
        ax.scatter([x], [y], s=120, color=color, edgecolors="black", linewidths=0.5, zorder=2)
        ax.text(
            x + 0.03,
            y + 0.03,
            atom.user_label,
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=3,
        )

    handles = []
    labels = []
    for role in ["metal", "bridging", "terminal", "spectator"]:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=ROLE_COLORS[role],
                markeredgecolor="black",
                markersize=8,
            )
        )
        labels.append(role)
    ax.legend(handles, labels, loc="best", frameon=False, title="Auto roles")
    ax.set_axis_off()
    ax.set_title("APEX prepare — labeled structure draft", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _project_to_2d(positions: np.ndarray) -> np.ndarray:
    centered = positions - positions.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = centered @ basis
    if projected.shape[1] != 2:
        projected = centered[:, :2]
    return projected


def _sorted_prepared_atoms(atoms: list[_PreparedAtom]) -> list[_PreparedAtom]:
    def label_num(atom: _PreparedAtom) -> int:
        suffix = atom.user_label[len(atom.element):]
        try:
            return int(suffix)
        except ValueError:
            return atom.atom_index + 1

    return sorted(
        atoms,
        key=lambda atom: (
            ROLE_PRIORITY.get(atom.role, 99),
            atom.element,
            label_num(atom),
            atom.atom_index,
        ),
    )


def _display_neighbor_indices(
    atom_index: int,
    role: str,
    *,
    connectivity: dict[int, list[int]],
    metal_info: dict[int, object],
    bridge_info: dict[int, object],
    terminal_info: dict[int, object],
    cluster_info,
    role_by_index: dict[int, str],
    label_by_index: dict[int, str],
) -> list[int]:
    raw_neighbors = sorted(connectivity.get(atom_index, []))
    metal_indices = set(metal_info)

    if role == "metal":
        return _sort_display_neighbors(
            [j for j in raw_neighbors if j in bridge_info or j in terminal_info],
            role_by_index,
            label_by_index,
        )

    if role in {"bridging", "interstitial"}:
        bridged = []
        if atom_index in bridge_info:
            bridged = [
                cluster_info.metals[m_idx].index
                for m_idx in bridge_info[atom_index].bridged_metals
                if 0 <= m_idx < len(cluster_info.metals)
            ]
        return _sort_display_neighbors(set(bridged), role_by_index, label_by_index)

    if role == "terminal":
        display = []
        if atom_index in terminal_info:
            metal_ref = cluster_info.metals[terminal_info[atom_index].metal_index].index
            display.append(metal_ref)
        display.extend(j for j in raw_neighbors if j not in metal_indices)
        return _sort_display_neighbors(set(display), role_by_index, label_by_index)

    return _sort_display_neighbors(raw_neighbors, role_by_index, label_by_index)


def _sort_display_neighbors(
    indices: list[int] | set[int],
    role_by_index: dict[int, str],
    label_by_index: dict[int, str],
) -> list[int]:
    def label_num(idx: int) -> int:
        label = label_by_index[idx]
        elem = "".join(ch for ch in label if not ch.isdigit())
        suffix = label[len(elem):]
        try:
            return int(suffix)
        except ValueError:
            return idx + 1

    return sorted(
        set(indices),
        key=lambda idx: (
            ROLE_PRIORITY.get(role_by_index.get(idx, "spectator"), 99),
            label_by_index[idx][0],
            label_num(idx),
            idx,
        ),
    )


def _format_neighbor_elements(elements: _Iterable[str]) -> str:
    counts = _Counter(elements)
    parts = []
    for elem in sorted(counts):
        count = counts[elem]
        parts.append(f"{elem}x{count}" if count > 1 else elem)
    return ",".join(parts)


def _build_auto_reason(
    atom_index: int,
    element: str,
    role: str,
    *,
    bound_to: str,
    bridging_to: str,
    bonded_neighbors: list[int],
    elements: list[str],
    metal_indices: set[int],
) -> str:
    metal_neighbor_count = sum(1 for j in bonded_neighbors if j in metal_indices)
    neighbor_elems = [elements[j] for j in bonded_neighbors]

    if role == "metal":
        if neighbor_elems:
            return f"auto metal center; bonded to {','.join(neighbor_elems)}"
        return "auto metal center"
    if role == "bridging":
        if bridging_to:
            return f"auto bridging atom; bridges {bridging_to}"
        return f"auto bridging atom; bonded to {metal_neighbor_count} metals"
    if role == "interstitial":
        if bridging_to:
            return f"auto interstitial atom; shared by {bridging_to}"
        return f"auto interstitial atom; bonded to {metal_neighbor_count} metals"
    if role == "terminal":
        if bound_to:
            return f"auto terminal donor; bound to {bound_to}"
        return "auto terminal donor"
    if bonded_neighbors:
        return "spectator atom; no direct metal-center role assigned"
    return "spectator atom; isolated from metal-center role analysis"
