"""Tests for apex_cas.prepare draft annotation helpers."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from apex_cas import BridgingAtom, ClusterInfo, MetalCenter, TerminalLigand
from apex_cas.prepare import (
    _build_prepared_atoms,
    _load_validated_prepared_atoms_from_csv,
    _write_cluster_info_draft_csv,
    _write_cluster_info_yaml,
)


class TestPrepareDraft(unittest.TestCase):
    @staticmethod
    def _cluster_info():
        return ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0, position=np.array([0.0, 0.0, 0.0]), label="Fe1"),
                MetalCenter(element="Fe", index=5, position=np.array([2.7, 0.0, 0.0]), label="Fe2"),
            ],
            bridging_atoms=[
                BridgingAtom(
                    element="S",
                    index=1,
                    position=np.array([1.35, 1.1, 0.0]),
                    bridged_metals=[0, 1],
                    role="bridging",
                    label="S1",
                    charge=-2,
                    projection_role="bridging_p",
                ),
                BridgingAtom(
                    element="S",
                    index=4,
                    position=np.array([1.35, -1.1, 0.0]),
                    bridged_metals=[0, 1],
                    role="bridging",
                    label="S2",
                    charge=-2,
                    projection_role="bridging_p",
                ),
            ],
            terminal_ligands=[
                TerminalLigand(
                    name="thiolate",
                    atom_indices=[2],
                    donor_atom_index=2,
                    charge=-1,
                    metal_index=0,
                    label="S3",
                    role="terminal",
                    ligand_type="thiolate",
                    projection_role="exclude",
                ),
                TerminalLigand(
                    name="thiolate",
                    atom_indices=[3],
                    donor_atom_index=3,
                    charge=-1,
                    metal_index=0,
                    label="S4",
                    role="terminal",
                    ligand_type="thiolate",
                    projection_role="exclude",
                ),
                TerminalLigand(
                    name="thiolate",
                    atom_indices=[6],
                    donor_atom_index=6,
                    charge=-1,
                    metal_index=1,
                    label="S5",
                    role="terminal",
                    ligand_type="thiolate",
                    projection_role="exclude",
                ),
                TerminalLigand(
                    name="thiolate",
                    atom_indices=[7],
                    donor_atom_index=7,
                    charge=-1,
                    metal_index=1,
                    label="S6",
                    role="terminal",
                    ligand_type="thiolate",
                    projection_role="exclude",
                ),
            ],
            all_elements=[
                "Fe", "S", "S", "S", "S", "Fe", "S", "S",
                "C", "H", "H", "H", "C", "H", "H", "H",
                "C", "H", "H", "H", "C", "H", "H", "H",
            ],
            all_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.35, 1.1, 0.0],
                [-1.2, 1.0, 0.0],
                [-1.2, -1.0, 0.0],
                [1.35, -1.1, 0.0],
                [2.7, 0.0, 0.0],
                [3.9, 1.0, 0.0],
                [3.9, -1.0, 0.0],
                [-2.4, 1.2, 0.0],
                [-3.0, 1.8, 0.8],
                [-3.0, 1.8, -0.8],
                [-2.6, 0.2, 0.0],
                [-2.4, -1.2, 0.0],
                [-3.0, -1.8, 0.8],
                [-3.0, -1.8, -0.8],
                [-2.6, -0.2, 0.0],
                [5.1, 1.2, 0.0],
                [5.7, 1.8, 0.8],
                [5.7, 1.8, -0.8],
                [5.3, 0.2, 0.0],
                [5.1, -1.2, 0.0],
                [5.7, -1.8, 0.8],
                [5.7, -1.8, -0.8],
                [5.3, -0.2, 0.0],
            ]),
            total_charge=-2,
            target_spin=0.0,
            symmetry_group="C1",
            reduction_symmetry="C1",
            family_scheme="",
            benchmark_profile="",
            config_reduction_mode="none",
        )

    def test_prepared_atoms_include_bonded_to_and_reason(self):
        ci = self._cluster_info()
        prepared = _build_prepared_atoms(ci)
        by_label = {atom.user_label: atom for atom in prepared}

        self.assertEqual(by_label["Fe1"].display_contacts, "S1,S2,S3,S4")
        self.assertEqual(by_label["Fe1"].neighbor_elements, "Sx4")
        self.assertIn("auto metal center", by_label["Fe1"].auto_reason)

        self.assertEqual(by_label["S1"].bridging_to, "Fe1,Fe2")
        self.assertEqual(by_label["S1"].display_contacts, "Fe1,Fe2")
        self.assertIn("bridges Fe1,Fe2", by_label["S1"].auto_reason)

        self.assertEqual(by_label["S3"].bound_to, "Fe1")
        self.assertTrue(by_label["S3"].display_contacts.startswith("Fe1"))
        self.assertIn("bound to Fe1", by_label["S3"].auto_reason)

    def test_draft_csv_round_trips_into_validated_yaml_atoms(self):
        ci = self._cluster_info()
        prepared = _build_prepared_atoms(ci)

        with tempfile.TemporaryDirectory() as tmpdir:
            draft_csv = Path(tmpdir) / "cluster_info_draft.csv"
            final_yaml = Path(tmpdir) / "cluster_info.yaml"
            _write_cluster_info_draft_csv(str(draft_csv), prepared)

            validated = _load_validated_prepared_atoms_from_csv(
                str(draft_csv),
                elements=list(ci.all_elements),
            )
            _write_cluster_info_yaml(str(final_yaml), ci, validated)

            payload = yaml.safe_load(final_yaml.read_text(encoding="utf-8"))
            self.assertEqual(payload["cluster"]["total_charge"], -2)
            self.assertEqual(len(payload["atoms"]), len(ci.all_elements))

            atom2 = next(row for row in payload["atoms"] if row["atom_index"] == 2)
            self.assertEqual(atom2["label"], "S3")
            self.assertEqual(atom2["bound_to"], "Fe1")

    def test_validation_rejects_unknown_bound_to_label(self):
        ci = self._cluster_info()
        prepared = _build_prepared_atoms(ci)

        with tempfile.TemporaryDirectory() as tmpdir:
            draft_csv = Path(tmpdir) / "cluster_info_draft.csv"
            _write_cluster_info_draft_csv(str(draft_csv), prepared)

            with draft_csv.open("r", encoding="utf-8", newline="") as fh:
                comment_lines = []
                data_lines = []
                for line in fh:
                    if line.lstrip().startswith("#"):
                        comment_lines.append(line)
                    else:
                        data_lines.append(line)
            reader = csv.DictReader(data_lines)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
            for row in rows:
                if row["atom_index"] == "2":
                    row["bound_to"] = "Fe9"
            with draft_csv.open("w", encoding="utf-8", newline="") as fh:
                for line in comment_lines:
                    fh.write(line)
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "unknown bound_to metal label"):
                _load_validated_prepared_atoms_from_csv(
                    str(draft_csv),
                    elements=list(ci.all_elements),
                )

