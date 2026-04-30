"""Tests for selection-file parsing."""

import csv
import json

from apex_filter.pick import apply_pick, parse_pick_arg


def test_apply_pick_file_json_labels(tmp_path):
    path = tmp_path / "picks.json"
    with open(path, "w") as f:
        json.dump({"labels": ["B", "A"]}, f)

    summary = [
        {"label": "A", "converged": True},
        {"label": "B", "converged": True},
        {"label": "C", "converged": True},
    ]
    labels = apply_pick(parse_pick_arg(f"file {path}"), summary)
    assert labels == ["A", "B"]


def test_apply_pick_file_csv_worklist(tmp_path):
    path = tmp_path / "selection_worklist.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keep", "label", "family"])
        writer.writeheader()
        writer.writerow({"keep": "yes", "label": "B", "family": "BS8"})
        writer.writerow({"keep": "", "label": "A", "family": "BS7"})
        writer.writerow({"keep": "1", "label": "C", "family": "BS9"})

    summary = [
        {"label": "A", "converged": True},
        {"label": "B", "converged": True},
        {"label": "C", "converged": True},
    ]
    labels = apply_pick(parse_pick_arg(f"file {path}"), summary)
    assert labels == ["B", "C"]


def test_apply_pick_file_csv_without_keep_column_uses_all_labels(tmp_path):
    path = tmp_path / "selection_candidates.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "family"])
        writer.writeheader()
        writer.writerow({"label": "B", "family": "BS8"})
        writer.writerow({"label": "A", "family": "BS7"})

    summary = [
        {"label": "A", "converged": True},
        {"label": "B", "converged": True},
        {"label": "C", "converged": True},
    ]
    labels = apply_pick(parse_pick_arg(f"file {path}"), summary)
    assert labels == ["A", "B"]
