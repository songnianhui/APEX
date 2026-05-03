"""Shared structure parsing and symmetry utilities for transition-metal clusters."""

import os
import re
from collections import Counter as _Counter
from itertools import combinations as _combinations
import json

import numpy as np
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

from .models import (
    BridgingAtom as _BridgingAtom,
    ClusterInfo as _ClusterInfo,
    MetalCenter as _MetalCenter,
    TerminalLigand as _TerminalLigand,
)
from .cluster_info_io import (
    load_cluster_info_yaml as _load_cluster_info_yaml,
    resolve_cluster_metadata as _resolve_cluster_metadata,
)
from .element_data import (
    BRIDGING_ELEMENTS as DEFAULT_BRIDGING_ELEMENTS,
    COVALENT_RADII as DEFAULT_COVALENT_RADII,
    TRANSITION_METALS as DEFAULT_TRANSITION_METALS,
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

TRANSITION_METALS = set(DEFAULT_TRANSITION_METALS)
COVALENT_RADII = dict(DEFAULT_COVALENT_RADII)
BRIDGING_ELEMENTS = set(DEFAULT_BRIDGING_ELEMENTS)

# Bond detection parameters
BOND_TOLERANCE = 1.3
MAX_METAL_LIGAND_DIST = 3.0
MAX_METAL_METAL_DIST = 3.5


# ──────────────────────────────────────────────────────────────────
# Structure parsing
# ──────────────────────────────────────────────────────────────────

def parse_structure(filepath: str, charge: int = 0, target_spin: float = 0.0,
                    cluster_info_path: str = None,
                    custom_metals: list | None = None,
                    symmetry_group_override: str = None,
                    reduction_symmetry_override: str = None,
                    symmetry_detection_mode: str = "auto",
                    family_scheme: str = "",
                    benchmark_profile: str = "",
                    config_reduction_mode: str = "none",
                    bond_tolerance: float = BOND_TOLERANCE,
                    max_metal_metal_dist: float = MAX_METAL_METAL_DIST,
                    max_metal_ligand_dist: float = MAX_METAL_LIGAND_DIST) -> _ClusterInfo:
    """Parse a structure file and return a ClusterInfo object.

    Args:
        filepath: Path to structure file (XYZ, PDB, or any format ASE supports).
        charge: Total charge of the cluster.
        target_spin: Target total spin S quantum number.
        cluster_info_path: Optional annotation file path. When provided,
            explicit atom roles/charges/labels override automatic parsing.
        custom_metals: Optional extra element symbols to treat as metals in
            auto-detection mode for unusual benchmark or setup cases.
        symmetry_group_override: Optional user-specified symmetry label
            (e.g. "Td", "C3"). Consumed according to symmetry_detection_mode.
        reduction_symmetry_override: Optional user-specified downstream
            reduction symmetry (typically Cn) used by APEX_Filter.
        symmetry_detection_mode: "auto" | "override" | "hint".
            - auto: use detected symmetry only
            - override: use symmetry_group_override directly
            - hint: prefer auto, but fall back to symmetry_group_override if
              auto detection is weak (C1)
        family_scheme: Optional family-labeling convention.
        benchmark_profile: Optional literature-aligned enumeration profile.
        config_reduction_mode: Whether downstream config symmetry reduction
            should be applied by default.
        bond_tolerance: Multiplier for covalent radii sum in bond detection.
        max_metal_metal_dist: Maximum distance (A) for metal-metal bonding.
        max_metal_ligand_dist: Maximum distance (A) for metal-ligand bonding.

    Returns:
        ClusterInfo with identified metals, bridging atoms, and ligands.
    """
    from ase.io import read as ase_read

    atoms = ase_read(filepath)
    elements = list(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    cluster_payload = None
    cluster_meta = {}
    atom_annotations = {}
    strict_authority = False
    if cluster_info_path:
        cluster_payload = _load_cluster_info_yaml(
            cluster_info_path,
            elements=elements,
        )
        cluster_meta = cluster_payload.get("cluster") or {}
        merged_meta = _resolve_cluster_metadata(
            cluster_meta,
            total_charge=charge,
            target_spin=target_spin,
            symmetry_group=symmetry_group_override,
            reduction_symmetry=reduction_symmetry_override,
            family_scheme=family_scheme,
            benchmark_profile=benchmark_profile,
            config_reduction_mode=config_reduction_mode,
        )
        charge = merged_meta["total_charge"]
        target_spin = merged_meta["target_spin"]
        symmetry_group_override = merged_meta["symmetry_group"] or symmetry_group_override
        reduction_symmetry_override = merged_meta["reduction_symmetry"] or reduction_symmetry_override
        family_scheme = merged_meta["family_scheme"] or family_scheme
        benchmark_profile = merged_meta["benchmark_profile"] or benchmark_profile
        config_reduction_mode = merged_meta["config_reduction_mode"] or config_reduction_mode
        atom_annotations = cluster_payload["atom_annotations"]
        strict_authority = _has_complete_authority_annotations(atom_annotations, elements)

    # Build connectivity graph
    connectivity = _build_connectivity(elements, positions, bond_tolerance=bond_tolerance)

    if strict_authority:
        metals = _build_metal_centers_from_authority(
            elements,
            positions,
            connectivity,
            atom_annotations,
        )
        bridging_atoms = _build_bridging_atoms_from_authority(
            elements,
            positions,
            metals,
            atom_annotations,
        )
        terminal_ligands = _build_terminal_ligands_from_authority(
            elements,
            metals,
            atom_annotations,
        )
    else:
        # Identify metal centers
        metals = _identify_metal_centers(
            elements,
            positions,
            annotations=atom_annotations,
            custom_metals=custom_metals,
        )

        # Set metal neighbors from connectivity
        for metal in metals:
            metal.neighbors = connectivity.get(metal.index, [])
            metal.coordination = len(metal.neighbors)

        # Identify bridging atoms
        bridging_atoms = _identify_bridging_atoms(
            elements,
            positions,
            metals,
            connectivity,
            annotations=atom_annotations,
        )

        # Identify terminal ligands
        terminal_ligands = _identify_terminal_ligands(
            elements,
            positions,
            metals,
            connectivity,
            annotations=atom_annotations,
        )

    # Generate formula
    formula = _generate_formula(elements)

    # Detect approximate symmetry
    sym_details = _detect_symmetry_details(metals, positions)
    molecular_point_group = _detect_molecular_point_group(elements, positions)
    symmetry_group = molecular_point_group
    sym_axis_atoms = sym_details["symmetry_axis_atoms"]
    symmetry_source = "auto"

    mode = (symmetry_detection_mode or "auto").lower()
    if mode not in {"auto", "override", "hint"}:
        raise ValueError(
            f"Unsupported symmetry_detection_mode={symmetry_detection_mode!r}. "
            "Use 'auto', 'override', or 'hint'."
        )

    if symmetry_group_override:
        if mode == "override":
            symmetry_group = symmetry_group_override
            symmetry_source = "user_override"
        elif mode == "hint" and symmetry_group == "C1":
            symmetry_group = symmetry_group_override
            symmetry_source = "user_hint"
    if cluster_meta.get("symmetry_group"):
        symmetry_group = cluster_meta["symmetry_group"]
        symmetry_source = "cluster_info_yaml"

    metal_framework_symmetry = sym_details["symmetry_group"]
    reduction_symmetry = (
        cluster_meta.get("reduction_symmetry")
        or reduction_symmetry_override
        or _default_reduction_symmetry(benchmark_profile, metal_framework_symmetry)
    )
    resolved_family_scheme = family_scheme or _default_family_scheme(benchmark_profile)

    return _ClusterInfo(
        metals=metals,
        bridging_atoms=bridging_atoms,
        terminal_ligands=terminal_ligands,
        all_elements=elements,
        all_positions=np.array(positions),
        formula=formula,
        total_charge=charge,
        target_spin=target_spin,
        symmetry_group=symmetry_group,
        metal_framework_symmetry=metal_framework_symmetry,
        reduction_symmetry=reduction_symmetry,
        symmetry_axis_atoms=sym_axis_atoms,
        symmetry_source=symmetry_source,
        symmetry_confidence=sym_details["confidence"],
        symmetry_candidates=sym_details["candidates"],
        family_scheme=resolved_family_scheme,
        benchmark_profile=benchmark_profile or "",
        config_reduction_mode=config_reduction_mode or "none",
        cluster_info_path=os.path.abspath(cluster_info_path) if cluster_info_path else "",
        annotation_source="cluster_info_yaml" if cluster_info_path else "auto",
    )


def analyze_symmetry_report(cluster_info: _ClusterInfo) -> dict:
    """Build a geometry-focused symmetry report for a parsed cluster.

    The report is intentionally descriptive rather than authoritative:
    it quantifies how symmetric the metal framework and same-element shells are,
    and records the detector's current best labels.
    """
    report = {
        "formula": cluster_info.formula,
        "symmetry_group": cluster_info.symmetry_group,
        "metal_framework_symmetry": getattr(cluster_info, "metal_framework_symmetry", "C1"),
        "reduction_symmetry": getattr(cluster_info, "reduction_symmetry", cluster_info.symmetry_group),
        "symmetry_source": getattr(cluster_info, "symmetry_source", "auto"),
        "symmetry_confidence": getattr(cluster_info, "symmetry_confidence", 0.0),
        "symmetry_candidates": getattr(cluster_info, "symmetry_candidates", []),
        "family_scheme": getattr(cluster_info, "family_scheme", ""),
        "benchmark_profile": getattr(cluster_info, "benchmark_profile", ""),
        "config_reduction_mode": getattr(cluster_info, "config_reduction_mode", "none"),
        "metal_framework": {},
        "shells": [],
    }

    metals = cluster_info.metals or []
    if not metals:
        return report

    metal_positions = np.array([m.position for m in metals], dtype=float)
    metal_center = metal_positions.mean(axis=0)
    pair_distances = [
        float(np.linalg.norm(metal_positions[i] - metal_positions[j]))
        for i, j in _combinations(range(len(metal_positions)), 2)
    ]
    pair_mean = float(np.mean(pair_distances)) if pair_distances else 0.0
    pair_std = float(np.std(pair_distances)) if pair_distances else 0.0
    pair_rel_std = pair_std / pair_mean if pair_mean > 1e-12 else 0.0
    metal_framework = {
        "n_metals": len(metals),
        "elements": [m.element for m in metals],
        "labels": [m.label for m in metals],
        "center": metal_center.tolist(),
        "pair_distances": pair_distances,
        "pair_distance_mean": pair_mean,
        "pair_distance_std": pair_std,
        "pair_distance_rel_std": pair_rel_std,
    }
    if len(metals) == 4:
        metal_framework["tetrahedral_score"] = _tetrahedral_score(metal_positions)
    report["metal_framework"] = metal_framework

    all_elements = cluster_info.all_elements or []
    all_positions = np.array(cluster_info.all_positions, dtype=float) if cluster_info.all_positions is not None else None
    if all_positions is None or len(all_elements) != len(all_positions):
        return report

    for elem in sorted(set(all_elements)):
        idx = [i for i, symbol in enumerate(all_elements) if symbol == elem]
        if len(idx) < 3:
            continue
        shell_positions = all_positions[idx]
        radial = np.linalg.norm(shell_positions - metal_center, axis=1)
        entry = {
            "element": elem,
            "count": len(idx),
            "indices": idx,
            "radial_distance_mean": float(np.mean(radial)),
            "radial_distance_std": float(np.std(radial)),
        }
        if len(idx) == 4:
            entry["tetrahedral_score"] = _tetrahedral_score(shell_positions)
        report["shells"].append(entry)

    return report


def format_symmetry_report(report: dict) -> str:
    """Render a symmetry report into concise CLI-friendly text."""
    lines = []
    lines.append(f"Formula            : {report.get('formula', '')}")
    lines.append(
        "Detected symmetry : "
        f"{report.get('symmetry_group', 'C1')} "
        f"(reduction={report.get('reduction_symmetry', 'C1')}, "
        f"source={report.get('symmetry_source', 'auto')}, "
        f"confidence={report.get('symmetry_confidence', 0.0):.2f})"
    )
    if report.get("metal_framework_symmetry"):
        lines.append(
            f"Framework symmetry: {report.get('metal_framework_symmetry', 'C1')}"
        )
    if report.get("benchmark_profile"):
        lines.append(
            f"Benchmark profile : {report.get('benchmark_profile')}"
        )
    if report.get("family_scheme"):
        lines.append(
            f"Family scheme     : {report.get('family_scheme')}"
        )
    if report.get("config_reduction_mode"):
        lines.append(
            f"Config reduction  : {report.get('config_reduction_mode')}"
        )

    candidates = report.get("symmetry_candidates") or []
    if candidates:
        formatted = ", ".join(
            f"{c.get('label', '?')}:{c.get('confidence', 0.0):.2f}"
            for c in candidates[:5]
        )
        lines.append(f"Candidates         : {formatted}")

    mf = report.get("metal_framework", {})
    if mf:
        lines.append("")
        lines.append("Metal Framework")
        lines.append(f"  n_metals         : {mf.get('n_metals', 0)}")
        lines.append(f"  labels           : {', '.join(mf.get('labels', []))}")
        lines.append(
            "  pair distances   : "
            + ", ".join(f"{d:.6f}" for d in mf.get("pair_distances", []))
        )
        lines.append(
            f"  pair mean/std    : "
            f"{mf.get('pair_distance_mean', 0.0):.6f} / "
            f"{mf.get('pair_distance_std', 0.0):.6f}"
        )
        lines.append(
            f"  pair rel std     : "
            f"{mf.get('pair_distance_rel_std', 0.0):.6e}"
        )
        if "tetrahedral_score" in mf:
            lines.append(
                f"  tetrahedral score: {mf.get('tetrahedral_score', 0.0):.6f}"
            )

    shells = report.get("shells") or []
    if shells:
        lines.append("")
        lines.append("Same-Element Shells")
        for shell in shells:
            line = (
                f"  {shell['element']} x{shell['count']}: "
                f"r_mean={shell['radial_distance_mean']:.6f}, "
                f"r_std={shell['radial_distance_std']:.6f}"
            )
            if "tetrahedral_score" in shell:
                line += f", tetra_score={shell['tetrahedral_score']:.6f}"
            lines.append(line)

    return "\n".join(lines)


def symmetry_report_json(report: dict) -> str:
    """Serialize the symmetry report as indented JSON."""
    return json.dumps(report, indent=2, sort_keys=True)


def _identify_metal_centers(elements, positions, annotations=None, custom_metals=None):
    """Identify all transition metal centers in the structure.

    Args:
        elements: List of element symbols.
        positions: List of (x, y, z) positions.
        annotations: Optional atom-annotation map from cluster_info.yaml.
    """
    annotations = annotations or {}
    metal_set = set(TRANSITION_METALS)
    if custom_metals:
        metal_set.update(custom_metals)

    metals = []
    metal_count = {}  # track count per element for labeling
    for i, elem in enumerate(elements):
        annotation = annotations.get(i, {})
        explicit_role = annotation.get("role")
        is_metal = explicit_role == "metal" or (
            explicit_role is None and elem in metal_set
        )
        if is_metal:
            metal_count[elem] = metal_count.get(elem, 0) + 1
            label = annotation.get("label") or f"{elem}{metal_count[elem]}"
            metals.append(_MetalCenter(
                element=elem,
                index=i,
                position=np.array(positions[i]),
                label=label,
                role="metal",
                charge=int(annotation.get("charge", 0)),
                projection_role=annotation.get("projection_role", "metal_df"),
            ))
    return metals


def _has_complete_authority_annotations(annotations: dict[int, dict], elements: list[str]) -> bool:
    """Return True when annotations are complete enough for strict reconstruction."""
    if not annotations or len(annotations) != len(elements):
        return False

    roles = set()
    all_core_fields_present = True
    for idx, elem in enumerate(elements):
        annotation = annotations.get(idx, {})
        if annotation.get("element", elem) != elem:
            return False
        for key in ("label", "role", "charge", "projection_role"):
            if key not in annotation or annotation.get(key) in (None, ""):
                all_core_fields_present = False
        role = annotation.get("role")
        if role is None:
            continue
        roles.add(role)
        if role in {"bridging", "interstitial"}:
            if not annotation.get("bridging_to"):
                if all_core_fields_present:
                    raise ValueError(
                        f"Explicit cluster_info annotations are incomplete: "
                        f"{role} atom {annotation.get('label', idx)} is missing bridging_to"
                    )
                return False
        elif role == "terminal":
            if not annotation.get("bound_to"):
                if all_core_fields_present:
                    raise ValueError(
                        f"Explicit cluster_info annotations are incomplete: "
                        f"terminal atom {annotation.get('label', idx)} is missing bound_to"
                    )
                return False
    if not all_core_fields_present:
        return False
    return "metal" in roles


def _build_metal_centers_from_authority(elements, positions, connectivity, annotations):
    """Build metal centers strictly from explicit annotations."""
    metals = []
    for i, elem in enumerate(elements):
        annotation = annotations.get(i, {})
        if annotation.get("role") != "metal":
            continue
        metal = _MetalCenter(
            element=elem,
            index=i,
            position=np.array(positions[i]),
            label=annotation["label"],
            role="metal",
            charge=int(annotation["charge"]),
            projection_role=annotation["projection_role"],
        )
        metal.neighbors = connectivity.get(i, [])
        metal.coordination = len(metal.neighbors)
        metals.append(metal)
    if not metals:
        raise ValueError("Strict cluster_info authority mode found no atoms with role='metal'")
    return metals


def _build_bridging_atoms_from_authority(elements, positions, metals, annotations):
    """Build bridging/interstitial atoms strictly from explicit annotations."""
    bridging_atoms = []
    for i, elem in enumerate(elements):
        annotation = annotations.get(i, {})
        role = annotation.get("role")
        if role not in {"bridging", "interstitial"}:
            continue
        bridged = _resolve_metal_refs(annotation.get("bridging_to", []), metals)
        if len(bridged) < 2:
            raise ValueError(
                f"Strict cluster_info authority mode requires at least two bridging_to labels for atom {annotation.get('label', i)}"
            )
        metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
        bridging_atoms.append(
            _BridgingAtom(
                element=elem,
                index=i,
                position=np.array(positions[i]),
                bridged_metals=[metal_idx_to_metal_pos[j] for j in bridged],
                role=role,
                label=annotation["label"],
                charge=int(annotation["charge"]),
                ligand_type=annotation.get("ligand_type", ""),
                projection_role=annotation["projection_role"],
            )
        )
    return bridging_atoms


def _build_terminal_ligands_from_authority(elements, metals, annotations):
    """Build terminal donors strictly from explicit annotations."""
    terminal_ligands = []
    metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
    for i, elem in enumerate(elements):
        annotation = annotations.get(i, {})
        if annotation.get("role") != "terminal":
            continue
        bound_to = _resolve_metal_refs(annotation.get("bound_to"), metals)
        if len(bound_to) != 1:
            raise ValueError(
                f"Strict cluster_info authority mode requires exactly one bound_to label for atom {annotation.get('label', i)}"
            )
        metal_idx = bound_to[0]
        terminal_ligands.append(
            _TerminalLigand(
                name=annotation.get("ligand_type", elem),
                atom_indices=[i],
                donor_atom_index=i,
                charge=int(annotation["charge"]),
                metal_index=metal_idx_to_metal_pos[metal_idx],
                label=annotation["label"],
                role="terminal",
                ligand_type=annotation.get("ligand_type", ""),
                projection_role=annotation["projection_role"],
            )
        )
    return terminal_ligands


def _bond_distance(r1, r2, tol=BOND_TOLERANCE):
    """Check if two atoms are bonded based on covalent radii."""
    r_a = COVALENT_RADII.get(r1[0], 1.5)
    r_b = COVALENT_RADII.get(r2[0], 1.5)
    dist = np.linalg.norm(r1[1] - r2[1])
    return dist <= (r_a + r_b) * tol


def _build_connectivity(elements, positions, bond_tolerance=BOND_TOLERANCE):
    """Build a simple connectivity graph based on interatomic distances."""
    n = len(elements)
    connectivity = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if _bond_distance((elements[i], positions[i]),
                              (elements[j], positions[j]), tol=bond_tolerance):
                connectivity[i].append(j)
                connectivity[j].append(i)

    return connectivity


def _identify_bridging_atoms(elements, positions, metals, connectivity,
                             annotations=None):
    """Identify atoms that bridge two or more metal centers.

    Args:
        annotations: Optional atom-annotation map from cluster_info.yaml.
    """
    metal_indices = {m.index for m in metals}
    metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
    bridging_atoms = []
    annotations = annotations or {}
    bridging_set = set(BRIDGING_ELEMENTS)
    BRIDGING_ATOM_CHARGES = {
        "F": -1, "Cl": -1, "Br": -1, "I": -1,
        "O": -2, "S": -2, "Se": -2,
        "N": -3,
        "C": -4,
        "H": -1,
    }

    for i, elem in enumerate(elements):
        if i in metal_indices:
            continue
        annotation = annotations.get(i, {})
        explicit_role = annotation.get("role")
        if explicit_role in {"terminal", "spectator"}:
            continue
        if explicit_role not in {"bridging", "interstitial"} and elem not in bridging_set:
            continue

        # Check which metals this atom is bonded to
        if annotation.get("bridging_to"):
            bonded_metals = _resolve_metal_refs(
                annotation.get("bridging_to", []),
                metals,
            )
        else:
            bonded_metals = [j for j in connectivity.get(i, []) if j in metal_indices]

        if len(bonded_metals) >= 2 or explicit_role in {"bridging", "interstitial"}:
            role = annotation.get("role", "bridging")
            if role not in {"bridging", "interstitial"}:
                role = "bridging"
            if role == "bridging" and elem == "C" and len(bonded_metals) >= 4:
                role = "interstitial"

            bridging_atoms.append(_BridgingAtom(
                element=elem,
                index=i,
                position=np.array(positions[i]),
                bridged_metals=[metal_idx_to_metal_pos[j] for j in bonded_metals],
                role=role,
                label=annotation.get("label", f"{elem}{i + 1}"),
                charge=int(annotation.get("charge", BRIDGING_ATOM_CHARGES.get(elem, 0))),
                ligand_type=annotation.get("ligand_type", ""),
                projection_role=annotation.get("projection_role", "bridging_p"),
            ))

    return bridging_atoms


def _identify_terminal_ligands(elements, positions, metals, connectivity, annotations=None):
    """Identify terminal ligands attached to metal centers."""
    metal_indices = {m.index for m in metals}
    metal_idx_to_metal_pos = {m.index: k for k, m in enumerate(metals)}
    terminal_ligands = []
    annotations = annotations or {}

    # Common charges for terminal donor atoms
    LIGAND_ATOM_CHARGES = {
        "F": -1, "Cl": -1, "Br": -1, "I": -1,
        "O": -2, "S": -2, "Se": -2,
        "N": -3,
        "C": -4,
        "H": +1,
    }

    for metal in metals:
        for neighbor_idx in metal.neighbors:
            if neighbor_idx in metal_indices:
                continue
            elem = elements[neighbor_idx]
            annotation = annotations.get(neighbor_idx, {})
            explicit_role = annotation.get("role")
            explicit_bound_to = annotation.get("bound_to")

            # Check if this neighbor is already a bridging atom (bridges to another metal)
            other_metal_neighbors = [
                j for j in connectivity.get(neighbor_idx, [])
                if j in metal_indices and j != metal.index
            ]
            if explicit_role in {"bridging", "interstitial"}:
                continue  # It's a bridging atom, not terminal

            # Explicit terminal annotations should take precedence over noisy
            # geometry-based multi-metal contacts, provided the intended bound
            # metal is stated via ``bound_to``.
            if other_metal_neighbors and not (explicit_role == "terminal" and explicit_bound_to):
                continue

            if explicit_role not in {None, "terminal"}:
                continue

            if explicit_bound_to:
                bound_to = _resolve_metal_refs(explicit_bound_to, metals)
                if metal.index not in bound_to:
                    continue

            lig_charge = int(annotation.get("charge", LIGAND_ATOM_CHARGES.get(elem, 0)))

            terminal_ligands.append(_TerminalLigand(
                name=annotation.get("ligand_type", elem),
                atom_indices=[neighbor_idx],
                donor_atom_index=neighbor_idx,
                charge=lig_charge,
                metal_index=metal_idx_to_metal_pos[metal.index],
                label=annotation.get("label", f"{elem}{neighbor_idx + 1}"),
                role="terminal",
                ligand_type=annotation.get("ligand_type", ""),
                projection_role=annotation.get("projection_role", "exclude"),
            ))

    return terminal_ligands


def _resolve_metal_refs(refs, metals):
    """Resolve metal references by label, atom index, or metal list index."""
    if isinstance(refs, (str, int)):
        refs = [refs]

    resolved = []
    by_label = {m.label: m.index for m in metals if m.label}
    by_atom_index = {m.index: m.index for m in metals}
    by_list_index = {k: m.index for k, m in enumerate(metals)}

    for ref in refs:
        if isinstance(ref, str):
            if ref not in by_label:
                raise ValueError(f"Unknown metal label in cluster_info annotation: {ref}")
            resolved.append(by_label[ref])
            continue

        ref_idx = int(ref)
        if ref_idx in by_atom_index:
            resolved.append(by_atom_index[ref_idx])
        elif ref_idx in by_list_index:
            resolved.append(by_list_index[ref_idx])
        else:
            raise ValueError(f"Unknown metal reference in cluster_info annotation: {ref}")

    return resolved


def _generate_formula(elements):
    """Generate a Hill-order chemical formula."""
    counts = _Counter(elements)
    # Put C first, then H, then alphabetical
    parts = []
    if "C" in counts:
        parts.append(f"C{counts['C']}" if counts['C'] > 1 else "C")
        del counts["C"]
    if "H" in counts:
        parts.append(f"H{counts['H']}" if counts['H'] > 1 else "H")
        del counts["H"]

    for elem in sorted(counts):
        c = counts[elem]
        parts.append(f"{elem}{c}" if c > 1 else elem)

    return "".join(parts)

def _detect_molecular_point_group(elements, positions) -> str:
    """Detect the full-cluster point group using PySCF's symmetry analyzer.

    PySCF's Mole builder enables symmetry detection when ``symmetry=True`` and
    exposes the detected group via ``groupname``/``topgroup``.
    """
    try:
        from pyscf import gto

        atom_spec = "; ".join(
            f"{elem} {x:.12f} {y:.12f} {z:.12f}"
            for elem, (x, y, z) in zip(elements, positions)
        )
        mol = gto.M(
            atom=atom_spec,
            basis="sto-3g",
            symmetry=True,
            verbose=0,
        )
        return getattr(mol, "groupname", None) or getattr(mol, "topgroup", "C1")
    except Exception:
        return "C1"


def _detect_symmetry_details(metals, positions):
    """Detect approximate symmetry from the metal framework.

    The detector is deliberately conservative:
    - First, identify special high-symmetry motifs we can recognize robustly
      from the metal framework (currently a tetrahedral 4-metal skeleton).
    - Then, probe candidate C2/C3/C4 axes using rigid rotation + Hungarian
      matching on the metal positions.

    Returns a dict with symmetry label, representative axis atoms, confidence,
    and ranked candidate symmetries.
    """
    if len(metals) < 2:
        return {
            "symmetry_group": "C1",
            "symmetry_axis_atoms": [],
            "confidence": 0.0,
            "candidates": [],
        }
    if len(metals) < 3:
        return {
            "symmetry_group": "C1",
            "symmetry_axis_atoms": [],
            "confidence": 0.0,
            "candidates": [],
        }

    metal_positions = np.array([m.position for m in metals], dtype=float)
    centered = metal_positions - metal_positions.mean(axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))), 1.0)

    if len(metals) == 4:
        tetra_score = _tetrahedral_score(metal_positions)
        if tetra_score >= 0.85:
            axis = _best_c3_axis_for_tetrahedron(centered)
            return {
                "symmetry_group": "Td",
                "symmetry_axis_atoms": _representative_axis_atoms(centered, axis),
                "confidence": tetra_score,
                "candidates": [
                    {"label": "Td", "confidence": tetra_score},
                    {"label": "C3", "confidence": tetra_score},
                ],
            }

    candidates = []
    best = {
        "symmetry_group": "C1",
        "symmetry_axis_atoms": [],
        "confidence": 0.0,
    }

    for axis in _generate_candidate_axes(centered):
        for fold in (4, 3, 2):
            norm_rmsd = _rotation_match_rmsd(centered, axis, fold) / scale
            threshold = {2: 0.12, 3: 0.10, 4: 0.08}[fold]
            if norm_rmsd > threshold:
                continue

            confidence = max(0.0, 1.0 - norm_rmsd / threshold)
            label = f"C{fold}"
            candidate = {
                "label": label,
                "confidence": round(confidence, 4),
                "axis_atoms": _representative_axis_atoms(centered, axis),
            }
            candidates.append(candidate)

            better = (
                _symmetry_order(label) > _symmetry_order(best["symmetry_group"])
                or (
                    label == best["symmetry_group"]
                    and confidence > best["confidence"]
                )
            )
            if better:
                best = {
                    "symmetry_group": label,
                    "symmetry_axis_atoms": candidate["axis_atoms"],
                    "confidence": confidence,
                }

    candidates = sorted(
        { (c["label"], tuple(c["axis_atoms"])): c for c in candidates }.values(),
        key=lambda x: (_symmetry_order(x["label"]), x["confidence"]),
        reverse=True,
    )
    for candidate in candidates:
        candidate["axis_atoms"] = list(candidate["axis_atoms"])

    return {
        "symmetry_group": best["symmetry_group"],
        "symmetry_axis_atoms": best["symmetry_axis_atoms"],
        "confidence": round(best["confidence"], 4),
        "candidates": candidates,
    }


def _count_rotational_matches(projections, fold, tol_angle=0.3, tol_r=0.3):
    """Count how many positions form a rotational pattern of given fold."""
    if len(projections) < fold:
        return 0

    target_angle = 2 * np.pi / fold
    radii = [p[0] for p in projections]
    angles = sorted([p[1] for p in projections])

    # Check if we have `fold` points at similar radii with regular angular spacing
    # Use the median radius as reference
    if len(radii) < fold:
        return 0

    # Cluster by similar radius
    sorted_r = sorted(radii)
    # Find the largest group with similar radii
    best_count = 0
    for start in range(len(sorted_r)):
        count = 1
        for k in range(start + 1, len(sorted_r)):
            if sorted_r[k] - sorted_r[start] < tol_r:
                count += 1
            else:
                break
        best_count = max(best_count, count)

    if best_count >= fold:
        angle_diffs = []
        for i in range(fold):
            a1 = angles[i]
            a2 = angles[(i + 1) % fold] + (2 * np.pi if i == fold - 1 else 0.0)
            angle_diffs.append(abs((a2 - a1) - target_angle))
        if max(angle_diffs) < tol_angle:
            return fold
    return 0


def _infer_reduction_symmetry(symmetry_group: str) -> str:
    """Infer the Cn reduction symmetry consumed by downstream enumeration."""
    if not symmetry_group:
        return "C1"

    label = symmetry_group.strip()
    if label.startswith("C") and label[1:].isdigit():
        return label

    match = re.match(r"^[CD](\d+)", label)
    if match:
        return f"C{match.group(1)}"

    reduced = {
        "Td": "C3",
        "T": "C3",
        "Th": "C3",
        "D2d": "C2",
        "D2h": "C2",
        "D2": "C2",
        "S4": "C2",
    }
    return reduced.get(label, "C1")


def _default_reduction_symmetry(benchmark_profile: str, metal_framework_symmetry: str) -> str:
    """Choose the default family-grouping symmetry.

    Literature-aligned benchmark profiles take precedence. Otherwise use the
    metal-framework symmetry, not the full-molecule point group.
    """
    profile_map = {
        "chan_fe2s2_2017": "C1",
        "chan_fe4s4_2017": "C1",
        "chan_femoco_2026_llduc": "C3",
    }
    if benchmark_profile in profile_map:
        return profile_map[benchmark_profile]
    return _infer_reduction_symmetry(metal_framework_symmetry)


def _default_family_scheme(benchmark_profile: str) -> str:
    profile_map = {
        "chan_fe2s2_2017": "literature_fe2s2_dimer",
        "chan_fe4s4_2017": "literature_fe4s4_cubane",
        "chan_femoco_2026_llduc": "literature_femoco_bs",
    }
    return profile_map.get(benchmark_profile, "")


def _tetrahedral_score(positions: np.ndarray) -> float:
    """Return a simple [0,1] score for a tetrahedral 4-point framework."""
    if positions.shape != (4, 3):
        return 0.0

    dists = []
    for i, j in _combinations(range(4), 2):
        dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    dists = np.array(dists)
    mean = float(np.mean(dists))
    if mean < 1e-8:
        return 0.0
    rel_std = float(np.std(dists) / mean)
    return max(0.0, 1.0 - rel_std / 0.08)


def _best_c3_axis_for_tetrahedron(centered: np.ndarray) -> np.ndarray:
    """Pick a representative C3 axis for a tetrahedral 4-point set."""
    norms = np.linalg.norm(centered, axis=1)
    idx = int(np.argmax(norms))
    axis = centered[idx]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return axis / axis_norm


def _generate_candidate_axes(centered: np.ndarray) -> list[np.ndarray]:
    """Generate unique candidate rotation axes from a metal framework."""
    axes = []
    n = len(centered)

    for vec in centered:
        if np.linalg.norm(vec) > 1e-8:
            axes.append(vec)

    for i, j in _combinations(range(n), 2):
        diff = centered[j] - centered[i]
        if np.linalg.norm(diff) > 1e-8:
            axes.append(diff)
        cross = np.cross(centered[i], centered[j])
        if np.linalg.norm(cross) > 1e-8:
            axes.append(cross)

    if n >= 3:
        cov = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(cov)
        axes.extend(eigenvectors.T)

    unique_axes = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            continue
        axis_hat = axis / norm
        if all(abs(np.dot(axis_hat, prev)) < 0.98 for prev in unique_axes):
            unique_axes.append(axis_hat)

    if not unique_axes:
        unique_axes.append(np.array([0.0, 0.0, 1.0]))
    return unique_axes


def _rotation_match_rmsd(centered: np.ndarray, axis: np.ndarray, fold: int) -> float:
    """RMSD after rotating the framework by one symmetry step around axis."""
    angle = 2 * np.pi / fold
    rot = _rotation_matrix(axis, angle)
    rotated = (rot @ centered.T).T
    diff = rotated[:, None, :] - centered[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_ind, col_ind = _linear_sum_assignment(cost)
    matched = cost[row_ind, col_ind]
    return float(np.sqrt(np.mean(matched ** 2)))


def _rotation_matrix(axis, angle):
    """Rotation matrix for rotation by angle around axis (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    eye = np.eye(3)
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _representative_axis_atoms(centered: np.ndarray, axis: np.ndarray) -> list[int]:
    """Return representative metal indices associated with a candidate axis."""
    projections = centered @ axis
    if len(projections) == 0:
        return []
    i_max = int(np.argmax(projections))
    i_min = int(np.argmin(projections))
    if i_max == i_min:
        return [i_max]
    return [i_max, i_min]


def _symmetry_order(sym_label):
    """Return numeric order for symmetry comparison."""
    if sym_label.startswith("C") and len(sym_label) > 1 and sym_label[1:].isdigit():
        return int(sym_label[1:])
    return 1
