"""Shared helpers for user-facing final-state signatures."""

from __future__ import annotations

import re

import numpy as np

from shared.cluster_info_labels import resolve_metal_site_label as _resolve_metal_site_label
from shared.models import CAS as _CAS, ClusterInfo as _ClusterInfo
from shared.roman import to_roman as _to_roman


def parse_orbital_metal_mapping(cas: _CAS, cluster_info: _ClusterInfo) -> dict:
    """Map active-space orbital indices to metal site indices."""
    if not cas.orbital_labels:
        return {}

    metal_label_map = _build_metal_label_map(cluster_info)
    metal_elements: dict[str, list[int]] = {}
    for site_idx, metal in enumerate(cluster_info.metals):
        metal_elements.setdefault(metal.element, []).append(site_idx)

    mapping = {}
    for orb_idx, label in enumerate(cas.orbital_labels):
        metal_site = _parse_label_to_metal_site(label, metal_label_map, metal_elements)
        if metal_site is not None and not _is_spin_carrying_metal_orbital(label):
            metal_site = None
        mapping[orb_idx] = metal_site
    return mapping


def summarize_final_state_from_dm(cas, config, cluster_info, dm):
    """Summarize the final broken-symmetry state from a density matrix."""
    dm_a, dm_b = dm
    metal_orbital_map = parse_orbital_metal_mapping(cas, cluster_info)
    spin_diag = np.diag(dm_a - dm_b)

    site_spin_proxy = {}
    spin_tokens = []
    for site_idx, _metal in enumerate(cluster_info.metals):
        orb_indices = [i for i, mapped_site in metal_orbital_map.items() if mapped_site == site_idx]
        site_spin = float(sum(spin_diag[i] for i in orb_indices))
        metal_label = _resolve_metal_site_label(cluster_info, site_idx)
        site_spin_proxy[metal_label] = site_spin
        arrow = "↑" if site_spin >= 0 else "↓"
        spin_tokens.append(f"{metal_label}{arrow}")

    oxidation_tokens = []
    if config.oxidation:
        for site_idx, _metal in enumerate(cluster_info.metals):
            if site_idx in config.oxidation.assignments:
                metal_label = _resolve_metal_site_label(cluster_info, site_idx)
                oxidation_tokens.append(
                    f"{metal_label}({_to_roman(config.oxidation.assignments[site_idx])})"
                )
    oxidation_label = "+".join(oxidation_tokens) if oxidation_tokens else "ox:none"

    final_d_basin = {}
    d_tokens = []
    if config.d_orbital_assignments:
        for site_idx in sorted(config.d_orbital_assignments):
            metal_label = _resolve_metal_site_label(cluster_info, site_idx)
            spin_dir = config.spin_assignment.get(site_idx, +1)
            minority_dm = dm_b if spin_dir == +1 else dm_a
            metal_orbs = [i for i, mapped_site in metal_orbital_map.items() if mapped_site == site_idx]
            if not metal_orbs:
                continue
            diag = [float(minority_dm[i, i]) for i in metal_orbs]
            target_orb = metal_orbs[int(np.argmax(diag))]
            basin = _short_d_label(cas.orbital_labels[target_orb])
            final_d_basin[metal_label] = basin
            d_tokens.append(f"{metal_label}:{basin}")

    return {
        "final_site_spin_proxy": site_spin_proxy,
        "final_d_basin": final_d_basin,
        "final_state_signature": f"{''.join(spin_tokens)}|{oxidation_label}|{'+'.join(d_tokens) if d_tokens else 'd:none'}",
    }


def _build_metal_label_map(cluster_info: _ClusterInfo) -> dict[str, int]:
    label_map = {}
    element_counts = {}
    for site_idx, metal in enumerate(cluster_info.metals):
        if metal.label:
            label_map[metal.label] = site_idx
        element_counts[metal.element] = element_counts.get(metal.element, 0) + 1
        label_map[f"{metal.element}{element_counts[metal.element]}"] = site_idx
    return label_map


def _parse_label_to_metal_site(label: str, metal_label_map: dict, metal_elements: dict) -> int | None:
    if not label:
        return None

    token = label.split("_", 1)[0].split(":", 1)[-1].strip()
    if token in metal_label_map:
        return metal_label_map[token]

    match = re.search(r"([A-Z][a-z]?)(\d+)", token)
    if not match:
        return None

    elem = match.group(1)
    site_num = int(match.group(2)) - 1

    if elem not in metal_elements:
        return None

    sites = metal_elements[elem]
    if site_num < len(sites):
        return sites[site_num]
    return None


def _is_spin_carrying_metal_orbital(label: str) -> bool:
    if "_" not in label:
        return False
    orbital_part = label.split("_", 1)[1]
    return re.search(r"\d+d", orbital_part) is not None


def _short_d_label(label: str) -> str:
    if "_" not in label:
        return label
    orbital_part = label.split("_", 1)[1]
    match = re.search(r"\d+d(.+)$", orbital_part)
    if match:
        return f"d{match.group(1)}"
    return orbital_part
