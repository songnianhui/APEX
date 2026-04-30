"""Shared artifact/path resolution helpers for APEX packages."""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

_STRUCTURE_SUFFIXES = (".xyz", ".pdb", ".mol", ".mol2", ".cif")


def load_json_if_exists(path: str) -> Optional[dict]:
    """Load JSON when present, otherwise return None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_fcidump_summary(fcidump_path: str) -> Optional[dict]:
    """Load the matching ``*_fcidump_info.json`` sidecar when available."""
    fcidump_path = os.path.abspath(fcidump_path)
    fcidump_dir = os.path.dirname(fcidump_path)
    for name in sorted(os.listdir(fcidump_dir)):
        if not name.endswith("_fcidump_info.json"):
            continue
        path = os.path.join(fcidump_dir, name)
        data = load_json_if_exists(path)
        if data and os.path.abspath(data.get("fcidump_path", "")) == fcidump_path:
            return data
    return None


def candidate_structure_paths(case_dir: str) -> list[str]:
    """Return plausible original structure files inside an APEX case dir."""
    candidates = []
    preferred_dirs = [
        os.path.join(case_dir, "inputs"),
        case_dir,
    ]

    for base in preferred_dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.isfile(path) and name.lower().endswith(_STRUCTURE_SUFFIXES):
                candidates.append(os.path.abspath(path))

    return list(dict.fromkeys(candidates))


def resolve_structure_path(config_raw: dict, case_dir: str) -> Optional[str]:
    """Resolve the source structure path for topology-preserving parsing."""
    explicit = config_raw.get("structure_path") or config_raw.get("apex_cas_structure_path")
    if explicit:
        if not os.path.isabs(explicit):
            explicit = os.path.join(case_dir, explicit)
        explicit = os.path.abspath(explicit)
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Configured structure_path does not exist: {explicit}")
        return explicit

    candidates = candidate_structure_paths(case_dir)
    if len(candidates) == 1:
        return candidates[0]

    input_candidates = [p for p in candidates if os.path.dirname(p).endswith(os.sep + "inputs")]
    if len(input_candidates) == 1:
        return input_candidates[0]

    return None


def resolve_cluster_info_path(
    config_raw: dict,
    case_dir: str,
    config_dir: str | None = None,
) -> Optional[str]:
    """Resolve an explicit ``cluster_info.yaml`` path if configured."""
    explicit = config_raw.get("cluster_info_path")
    if not explicit:
        inputs_dir = os.path.join(case_dir, "inputs")
        if not os.path.isdir(inputs_dir):
            return None

        names = [
            os.path.join(inputs_dir, name)
            for name in sorted(os.listdir(inputs_dir))
            if name == "cluster_info.yaml" or name.endswith("_cluster_info.yaml")
        ]
        names = [os.path.abspath(path) for path in names if os.path.isfile(path)]
        if len(names) == 1:
            return names[0]
        return None

    candidates = []
    if os.path.isabs(explicit):
        candidates.append(explicit)
    else:
        if config_dir:
            candidates.append(os.path.join(config_dir, explicit))
        candidates.append(os.path.join(case_dir, explicit))
        candidates.append(os.path.join(case_dir, "inputs", explicit))

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Configured cluster_info_path does not exist: {explicit}")


def auto_detect_fcidump(case_dir: str) -> Optional[str]:
    """Auto-detect FCIDUMP file in ``case_dir/outputs/fcidump``."""
    fcidump_dir = os.path.join(case_dir, "outputs", "fcidump")
    if not os.path.isdir(fcidump_dir):
        return None

    for name in ["FCIDUMP", "fcidump", "FCIDUMP.txt"]:
        path = os.path.join(fcidump_dir, name)
        if os.path.isfile(path):
            return path
    return None


def resolve_fcidump_path(
    config_raw: dict,
    case_dir: str,
    config_dir: str | None = None,
) -> Optional[str]:
    """Resolve FCIDUMP path from config, supporting wildcard patterns."""
    explicit = config_raw.get("fcidump_path")
    if not explicit:
        return auto_detect_fcidump(case_dir)

    candidates = []
    if os.path.isabs(explicit):
        candidates.append(explicit)
    else:
        if config_dir:
            candidates.append(os.path.join(config_dir, explicit))
        candidates.append(os.path.join(case_dir, explicit))

    def _is_primary_fcidump(path: str) -> bool:
        name = os.path.basename(path)
        return (
            os.path.isfile(path)
            and not name.endswith(".ecore")
            and not name.endswith(".json")
        )

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if glob.has_magic(candidate):
            matches = [
                os.path.abspath(path)
                for path in sorted(glob.glob(candidate))
                if _is_primary_fcidump(path)
            ]
            if matches:
                return matches[0]
        elif _is_primary_fcidump(candidate):
            return candidate

    raise FileNotFoundError(
        f"Configured fcidump_path does not exist or matched no primary FCIDUMP file: {explicit}"
    )
