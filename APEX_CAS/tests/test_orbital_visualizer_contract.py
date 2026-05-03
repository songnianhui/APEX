"""Contract tests for orbital visualization orchestration and internal artifact writers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyscf import gto

from apex_cas.orbital_visualizer import (
    _generate_orbital_cubes,
    _generate_orbital_report,
    plot_orbitals,
)


class TestOrbitalVisualizerContract(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    def test_internal_orbital_report_writer_writes_markdown_projection_columns(self):
        coeff = np.eye(self.mol.nao_nr())
        occupations = np.array([1.95, 0.05])
        labels = ["H1_1s", "H2_1s"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "orbital_report.md"
            _generate_orbital_report(
                self.mol,
                coeff,
                occupations,
                labels,
                output_path=str(path),
                active_indices=[0, 1],
                cas_type="uno",
                selection_method="projection",
                projection_weights=np.array([0.33, 0.12]),
                projection_weights_metal=np.array([0.20, 0.05]),
                projection_weights_bridging=np.array([0.13, 0.07]),
            )
            text = path.read_text()

        self.assertIn("# Orbital Report", text)
        self.assertIn("| idx | occ | block | label | selected | character | proj_weight | proj_wt_M | proj_wt_B | ao_contrib |", text)
        self.assertIn("| cas_type | uno |", text)
        self.assertIn("| selection_method | projection |", text)
        self.assertIn("H1_1s", text)

    def test_internal_orbital_cube_writer_uses_labelled_filenames(self):
        coeff = np.eye(self.mol.nao_nr())
        labels = ["Fe1_3dz^2", "S1_3px"]

        with tempfile.TemporaryDirectory() as tmpdir, patch("pyscf.tools.cubegen.orbital") as mock_orbital:
            paths = _generate_orbital_cubes(
                self.mol,
                coeff,
                indices=[0, 1],
                labels=labels,
                output_dir=tmpdir,
                prefix="demo_orb",
                nx=10,
                ny=10,
                nz=10,
            )

        self.assertTrue(paths[0].endswith("demo_orb_0000_Fe1_3dz2.cube"))
        self.assertTrue(paths[1].endswith("demo_orb_0001_S1_3px.cube"))
        self.assertEqual(mock_orbital.call_count, 2)

    def test_plot_orbitals_respects_projection_threshold_and_generates_gallery(self):
        coeff = np.eye(self.mol.nao_nr())
        cas = SimpleNamespace(
            mo_coeff_full=coeff,
            occupations_full=np.array([2.0, 1.0]),
            orbital_labels_full=["old0", "old1"],
            active_indices=[1],
            cpt_cas_type="uno",
            selection_method="projection",
            projection_weights=np.array([0.20, 0.35]),
            projection_weights_metal=np.array([0.20, 0.10]),
            projection_weights_bridging=np.array([0.00, 0.25]),
        )

        def _fake_cube_writer(_mol, _coeff, indices=None, labels=None, output_dir="", prefix="orb", nx=80, ny=80, nz=80):
            output = []
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for idx in indices or []:
                label = labels[idx]
                cube_path = Path(output_dir) / f"{prefix}_{idx:04d}_{label}.cube"
                cube_path.write_text("cube")
                output.append(str(cube_path))
            return output

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "apex_cas.orbital_visualizer._ensure_chemical_labels",
            return_value=["Fe1_3dx2-y2", "S1_3px"],
        ), patch(
            "apex_cas.orbital_visualizer._generate_orbital_cubes",
            side_effect=_fake_cube_writer,
        ):
            result = plot_orbitals(
                cas,
                self.mol,
                tmpdir,
                cluster_info=None,
                generate_cubes=True,
                cube_grid="10x10x10",
                stem="demo",
                pw_plot_threshold=0.1,
                render_png=True,
                png_isovalue=0.05,
            )

            report_path = Path(result["report_path"])
            gallery_path = Path(result["html_gallery_path"])
            cube_dir = Path(result["cube_dir"])
            gallery_text = gallery_path.read_text()

            self.assertTrue(report_path.name == "demo_orbital_report.md")
            self.assertTrue(gallery_path.name == "demo_orbital_gallery.html")
            self.assertTrue((Path(tmpdir) / "demo_orbital_gallery_server.py").exists())
            cube_files = sorted(path.name for path in cube_dir.glob("*.cube"))
            self.assertIn('stick: {radius: 0.045, colorscheme: "Jmol"}', gallery_text)
            self.assertIn('addStyle({elem: "H"}, {sphere: {scale: 0.14, colorscheme: "Jmol"}});', gallery_text)
            self.assertIn('addStyle({elem: "C"}, {sphere: {scale: 0.20, colorscheme: "Jmol"}});', gallery_text)
            self.assertIn('addStyle({elem: "S"}, {sphere: {scale: 0.24, colorscheme: "Jmol"}});', gallery_text)
            self.assertIn('addStyle({elem: "Fe"}, {sphere: {scale: 0.27, colorscheme: "Jmol"}});', gallery_text)

        self.assertEqual(cube_files, ["demo_orb_0000_Fe1_3dx2-y2.cube", "demo_orb_0001_S1_3px.cube"])

