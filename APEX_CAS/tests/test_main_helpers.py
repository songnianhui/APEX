"""Tests for lightweight workflow helpers in apex_cas.main."""

import importlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from apex_cas import CAS, ComputationSettings
from shared.dmrg_solvers import resolve_dmrgci_twodot_to_onedot
from apex_cas.main import (
    _build_settings,
    _parse_compute_cluster,
    _restore_mean_field_from_chkfile,
    _resolve_fcidump_selection_path,
    _resolve_fcidump_spin_projection,
)
from shared.orbital_methods.localization import build_localization_params_from_settings
from shared.models import ClusterInfo


class TestMainHelpers(unittest.TestCase):
    def test_build_localization_params_boys_uses_settings(self):
        settings = ComputationSettings()
        settings.boys_conv_tol = 1e-9
        settings.boys_conv_tol_grad = 1e-5
        settings.boys_max_cycle = 88

        params = build_localization_params_from_settings(settings, "boys")

        self.assertEqual(params["conv_tol"], 1e-9)
        self.assertEqual(params["conv_tol_grad"], 1e-5)
        self.assertEqual(params["max_cycle"], 88)

    def test_build_localization_params_pm_uses_settings(self):
        settings = ComputationSettings()
        settings.pm_pop_method = "mulliken"
        settings.pm_conv_tol = 1e-8
        settings.pm_conv_tol_grad = 1e-6
        settings.pm_max_cycle = 123
        settings.pm_exponent = 4
        settings.pm_init_guess = "cholesky"

        params = build_localization_params_from_settings(settings, "pm")

        self.assertEqual(params["pop_method"], "mulliken")
        self.assertEqual(params["conv_tol"], 1e-8)
        self.assertEqual(params["conv_tol_grad"], 1e-6)
        self.assertEqual(params["max_cycle"], 123)
        self.assertEqual(params["exponent"], 4)
        self.assertEqual(params["init_guess"], "cholesky")

    def test_resolve_fcidump_spin_projection_defaults_to_saved_target_spin(self):
        args = SimpleNamespace(spin_projection=None)
        cas = CAS()
        cas.target_spin = 1.5

        self.assertEqual(_resolve_fcidump_spin_projection(args, cas), 1.5)

    def test_resolve_fcidump_spin_projection_falls_back_to_zero(self):
        args = SimpleNamespace(spin_projection=None)
        cas = CAS()
        cas.target_spin = 0.0

        self.assertEqual(_resolve_fcidump_spin_projection(args, cas), 0.0)

    def test_resolve_fcidump_selection_path_prefers_default_case_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            orbitals_dir = os.path.join(output_dir, "orbitals")
            os.makedirs(orbitals_dir, exist_ok=True)
            selection_path = os.path.join(orbitals_dir, "demo_selection.txt")
            with open(selection_path, "w") as f:
                f.write("n-electrons: 2\nn-orbital: 1\n0\n")

            args = SimpleNamespace(active_file=None)
            resolved = _resolve_fcidump_selection_path(args, tmpdir, output_dir, "demo")

        self.assertTrue(resolved.endswith("demo_selection.txt"))

    def test_resolve_dmrgci_twodot_to_onedot_keeps_valid_default(self):
        self.assertEqual(resolve_dmrgci_twodot_to_onedot(26, 30), 26)

    def test_resolve_dmrgci_twodot_to_onedot_clamps_invalid_equal_max_iter(self):
        self.assertEqual(resolve_dmrgci_twodot_to_onedot(30, 30), 29)

    def test_resolve_dmrgci_twodot_to_onedot_preserves_zero(self):
        self.assertEqual(resolve_dmrgci_twodot_to_onedot(0, 30), 0)

    def test_build_settings_clears_preset_basis_overrides_when_default_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = os.path.join(tmpdir, "cas.yaml")
            with open(settings_path, "w") as fh:
                fh.write("basis_set_default: tzp-dkh\n")

            args = SimpleNamespace(
                cas_settings=settings_path,
                charge=None,
                total_spin=None,
                scf_spin=None,
                symmetry_group=None,
                reduction_symmetry=None,
                symmetry_mode=None,
                no_cubes=False,
                cube_grid=None,
            )

            settings, *_ = _build_settings(args)

        self.assertEqual(settings.basis_set_default, "tzp-dkh")
        self.assertEqual(settings.basis_set_per_element, {})

    def test_build_settings_reads_scf_spin_into_formal_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = os.path.join(tmpdir, "cas.yaml")
            with open(settings_path, "w") as fh:
                fh.write("scf_spin: 4.5\n")

            args = SimpleNamespace(
                cas_settings=settings_path,
                charge=None,
                total_spin=None,
                scf_spin=None,
                symmetry_group=None,
                reduction_symmetry=None,
                symmetry_mode=None,
                no_cubes=False,
                cube_grid=None,
            )

            settings, *_ = _build_settings(args)

        self.assertEqual(settings.scf_spin, 4.5)

    def test_build_settings_unknown_preset_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = os.path.join(tmpdir, "cas.yaml")
            with open(settings_path, "w") as fh:
                fh.write("preset: nonexistent\n")

            args = SimpleNamespace(
                cas_settings=settings_path,
                charge=None,
                total_spin=None,
                scf_spin=None,
                symmetry_group=None,
                reduction_symmetry=None,
                symmetry_mode=None,
                no_cubes=False,
                cube_grid=None,
            )

            with self.assertRaises(KeyError):
                _build_settings(args)

    def test_parse_compute_cluster_requires_finalized_cluster_info_path(self):
        args = SimpleNamespace(structure="demo.xyz")
        with self.assertRaises(FileNotFoundError):
            _parse_compute_cluster(
                args,
                case_dir="demo_case",
                charge=-2,
                total_spin=0.0,
                symmetry_group_override=None,
                reduction_symmetry_override=None,
                symmetry_mode="auto",
                family_scheme="",
                benchmark_profile="",
                config_reduction_mode="none",
                cluster_info_path=None,
            )

    def test_parse_compute_cluster_requires_authoritative_cluster_info(self):
        args = SimpleNamespace(structure="demo.xyz")
        cluster = ClusterInfo(annotation_source="auto")

        with patch("apex_cas.main._parse_structure", return_value=cluster):
            with self.assertRaises(ValueError):
                _parse_compute_cluster(
                    args,
                    case_dir="demo_case",
                    charge=-2,
                    total_spin=0.0,
                    symmetry_group_override=None,
                    reduction_symmetry_override=None,
                    symmetry_mode="auto",
                    family_scheme="",
                    benchmark_profile="",
                    config_reduction_mode="none",
                    cluster_info_path="inputs/demo_cluster_info.yaml",
                )

    def test_restore_mean_field_from_chkfile_recovers_converged_from_scf_summary(self):
        settings = ComputationSettings()
        mf_stub = SimpleNamespace(converged=False, chkfile=None)
        cas_builder = importlib.import_module("apex_cas.CAS_builder")

        with patch("pyscf.lib.chkfile.load_mol", return_value="mol"), patch(
            "pyscf.lib.chkfile.load",
            return_value={"e_tot": -1.23},
        ), patch.object(
            cas_builder,
            "_build_mf_object",
            return_value=mf_stub,
        ), patch(
            "apex_cas.state_io._load_scf_summary",
            return_value={"converged": True},
        ):
            mol, mf = _restore_mean_field_from_chkfile(
                "/tmp/demo_case/outputs/scf/demo.chk",
                settings,
            )

        self.assertEqual(mol, "mol")
        self.assertTrue(mf.converged)
        self.assertEqual(mf.chkfile, "/tmp/demo_case/outputs/scf/demo.chk")
