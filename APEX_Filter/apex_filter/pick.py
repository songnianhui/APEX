"""Selection mechanism for the APEX_Filter interactive pipeline.

Parses ``--pick`` arguments and applies selection strategies to per-step
result summaries.
"""

import csv
import json
import os
from collections import defaultdict


def parse_pick_arg(pick_str: str) -> dict:
    """Parse a ``--pick`` CLI string into a structured spec.

    Supported forms::

        "all"                -> {"mode": "all"}
        "top 5"              -> {"mode": "top", "n": 5}
        "top-per-family 24"  -> {"mode": "top_per_family", "n": 24}
        "labels L1,L2,L3"    -> {"mode": "labels", "labels": ["L1", "L2", "L3"]}
        "energy-window 0.1"  -> {"mode": "energy_window", "window": 0.1}
        "file picks.json"    -> {"mode": "file", "path": "picks.json"}
    """
    if not pick_str:
        return {"mode": "all"}

    parts = pick_str.strip().split(None, 1)
    mode = parts[0].lower()

    if mode == "all":
        return {"mode": "all"}
    elif mode == "top":
        if len(parts) < 2:
            raise ValueError("'top' requires a number, e.g. 'top 5'")
        return {"mode": "top", "n": int(parts[1])}
    elif mode in ("top-per-family", "top_per_family"):
        if len(parts) < 2:
            raise ValueError("'top-per-family' requires a number")
        return {"mode": "top_per_family", "n": int(parts[1])}
    elif mode == "labels":
        if len(parts) < 2:
            raise ValueError("'labels' requires a comma-separated list")
        return {"mode": "labels", "labels": [l.strip() for l in parts[1].split(",")]}
    elif mode in ("energy-window", "energy_window"):
        if len(parts) < 2:
            raise ValueError("'energy-window' requires a float value")
        return {"mode": "energy_window", "window": float(parts[1])}
    elif mode == "file":
        if len(parts) < 2:
            raise ValueError("'file' requires a path")
        return {"mode": "file", "path": parts[1]}
    else:
        raise ValueError(f"Unknown pick mode: '{mode}'")


def apply_pick(pick_spec: dict, summary: list[dict]) -> list[str]:
    """Apply a selection strategy to a step summary, returning chosen labels.

    Parameters
    ----------
    pick_spec : dict
        Output of ``parse_pick_arg()``.
    summary : list[dict]
        Each entry: ``{"label": str, "energy": float, "converged": bool, "family": str}``.
        Assumed sorted by *energy* ascending (caller's responsibility).

    Returns
    -------
    list[str]
        Selected config labels.
    """
    mode = pick_spec["mode"]

    # Only consider converged results unless explicitly using labels mode
    converged = [s for s in summary if s.get("converged", True)]

    if mode == "all":
        return [s["label"] for s in converged]

    elif mode == "top":
        n = pick_spec["n"]
        return [s["label"] for s in converged[:n]]

    elif mode == "top_per_family":
        n = pick_spec["n"]
        families = defaultdict(list)
        for s in converged:
            fam = s.get("family", "")
            families[fam].append(s)

        selected = []
        for fam_label, group in families.items():
            selected.extend(group[:n])
        # Sort by energy
        selected.sort(key=lambda s: s.get("energy", float("inf")))
        return [s["label"] for s in selected]

    elif mode == "labels":
        requested = set(pick_spec["labels"])
        # Return in summary order (energy-sorted), but include even if not converged
        return [s["label"] for s in summary if s["label"] in requested]

    elif mode == "energy_window":
        window = pick_spec["window"]
        if not converged:
            return []
        min_energy = converged[0].get("energy", 0.0)
        return [
            s["label"] for s in converged
            if s.get("energy", float("inf")) - min_energy <= window
        ]

    elif mode == "file":
        path = pick_spec["path"]
        file_labels = _load_labels_from_file(path)
        file_set = set(file_labels)
        return [s["label"] for s in summary if s["label"] in file_set]

    else:
        raise ValueError(f"Unknown pick mode: {mode}")


def _load_labels_from_file(path: str) -> list[str]:
    """Load picked labels from a JSON or CSV file."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return _load_labels_from_csv(path)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [d["label"] for d in data]
        return list(data)
    if isinstance(data, dict) and "labels" in data:
        return data["labels"]
    raise ValueError(f"Unexpected format in {path}")


def _is_truthy_flag(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "keep", "select", "selected", "include", "run"}


def _load_labels_from_csv(path: str) -> list[str]:
    """Load picked labels from a CSV worklist or candidate table."""
    with open(path, newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        rows = list(reader)
    if not rows:
        return []
    if "label" not in {name.lower(): name for name in reader.fieldnames or []}:
        raise ValueError(f"CSV pick file must contain a 'label' column: {path}")

    normalized_fields = {name.lower(): name for name in (reader.fieldnames or [])}
    label_key = normalized_fields["label"]
    flag_keys = [
        normalized_fields[name]
        for name in ("keep", "select", "selected", "include", "use", "run")
        if name in normalized_fields
    ]
    if not flag_keys:
        return [row[label_key] for row in rows if row.get(label_key)]

    selected = []
    for row in rows:
        label = row.get(label_key)
        if not label:
            continue
        if any(_is_truthy_flag(row.get(flag_key)) for flag_key in flag_keys):
            selected.append(label)
    return selected
