"""Regression tests for the internal Step 3+ ``--pick`` helper seam."""

import csv
import json

from apex_filter.pick import _apply_pick, _parse_pick_arg


def test_internal_pick_helper_reads_json_label_files(tmp_path):
    path = tmp_path / "picks.json"
    with open(path, "w") as f:
        json.dump({"labels": ["B", "A"]}, f)

    summary = [
        {"label": "A", "converged": True},
        {"label": "B", "converged": True},
        {"label": "C", "converged": True},
    ]
    labels = _apply_pick(_parse_pick_arg(f"file {path}"), summary)
    assert labels == ["A", "B"]


def test_internal_pick_helper_reads_csv_worklists(tmp_path):
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
    labels = _apply_pick(_parse_pick_arg(f"file {path}"), summary)
    assert labels == ["B", "C"]


def test_internal_pick_helper_uses_all_labels_when_keep_column_is_absent(tmp_path):
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
    labels = _apply_pick(_parse_pick_arg(f"file {path}"), summary)
    assert labels == ["A", "B"]
