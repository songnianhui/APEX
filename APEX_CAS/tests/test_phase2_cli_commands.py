"""Tests for new scf/buildcas CLI workflow commands."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from apex_cas.main import create_parser, run_buildcas, run_compute, run_fcidump, run_scf


class TestPhase2CLICommands(unittest.TestCase):
    def test_parser_includes_scf_and_buildcas(self):
        parser = create_parser()
        help_text = parser.format_help()
        self.assertIn("scf", help_text)
        self.assertIn("buildcas", help_text)

    def test_run_scf_uses_scf_only_workflow(self):
        args = SimpleNamespace(
            structure="demo.xyz",
            case_dir="demo_case",
            cas_settings=None,
            charge=None,
            total_spin=None,
            scf_spin=None,
            symmetry_group=None,
            reduction_symmetry=None,
            symmetry_mode=None,
            yes=False,
        )
        fake_settings = object()
        with patch("apex_cas.main._resolve_case_dir", return_value="demo_case"), patch(
            "apex_cas.main._build_settings",
            return_value=(
                fake_settings,
                -2,
                0.0,
                "boys",
                {},
                {},
                "uno",
                None,
                None,
                None,
                None,
                "auto",
                "",
                "",
                "none",
            ),
        ), patch(
            "apex_cas.main._parse_compute_cluster",
            return_value="cluster",
        ), patch(
            "apex_cas.main._run_scf_only",
            return_value=("mol", "mf", "/tmp/demo.chk"),
        ) as mocked_run, patch(
            "apex_cas.main._print_scf_next_steps"
        ), patch(
            "apex_cas.CAS_builder._get_output_stem",
            return_value="demo",
        ), patch(
            "apex_cas.state_io.save_scf_summary"
        ) as mocked_save:
            run_scf(args)

        mocked_run.assert_called_once()
        mocked_save.assert_called_once()

    def test_run_buildcas_uses_saved_scf_workflow(self):
        args = SimpleNamespace(
            structure="demo.xyz",
            case_dir="demo_case",
            cas_settings=None,
            charge=None,
            total_spin=None,
            scf_spin=None,
            symmetry_group=None,
            reduction_symmetry=None,
            symmetry_mode=None,
        )
        fake_settings = object()
        fake_cas = SimpleNamespace(
            n_electrons=10,
            n_orbitals=10,
            mo_coeff_full=None,
        )
        with patch("apex_cas.main._resolve_case_dir", return_value="demo_case"), patch(
            "apex_cas.main._build_settings",
            return_value=(
                fake_settings,
                -2,
                0.0,
                "boys",
                {"generate_cubes": False, "cube_grid": "40x40x40", "pw_plot_threshold": None, "render_png": False, "png_isovalue": 0.05},
                {"projection_threshold": 0.3},
                "uno",
                None,
                None,
                None,
                None,
                "auto",
                "",
                "",
                "none",
            ),
        ), patch(
            "apex_cas.main._parse_compute_cluster",
            return_value="cluster",
        ), patch(
            "apex_cas.main._build_cas_from_saved_scf",
            return_value=(fake_cas, "mol", "mf", "/tmp/demo.chk"),
        ) as mocked_build, patch(
            "apex_cas.main._synchronize_cas_labels"
        ), patch(
            "apex_cas.main._save_and_validate_compute_outputs"
        ) as mocked_save, patch(
            "apex_cas.main._generate_compute_visualizations"
        ) as mocked_viz:
            run_buildcas(args)

        mocked_build.assert_called_once()
        mocked_save.assert_called_once()
        mocked_viz.assert_called_once()

    def test_run_compute_fresh_uses_split_scf_then_buildcas_flow(self):
        args = SimpleNamespace(
            structure="demo.xyz",
            case_dir="demo_case",
            cas_settings=None,
            charge=None,
            total_spin=None,
            scf_spin=None,
            symmetry_group=None,
            reduction_symmetry=None,
            symmetry_mode=None,
            yes=False,
        )
        fake_settings = object()
        fake_cas = SimpleNamespace(
            n_electrons=10,
            n_orbitals=10,
            mo_coeff_full=None,
        )
        with patch("apex_cas.main._resolve_case_dir", return_value="demo_case"), patch(
            "apex_cas.main._build_settings",
            return_value=(
                fake_settings,
                -2,
                0.0,
                "boys",
                {"generate_cubes": False, "cube_grid": "40x40x40", "pw_plot_threshold": None, "render_png": False, "png_isovalue": 0.05},
                {"projection_threshold": 0.3},
                "uno",
                None,
                None,
                None,
                None,
                "auto",
                "",
                "",
                "none",
            ),
        ), patch(
            "apex_cas.main._parse_compute_cluster",
            return_value="cluster",
        ), patch(
            "apex_cas.main._run_scf_only",
            return_value=("mol", "mf", "/tmp/demo.chk"),
        ) as mocked_run_scf, patch(
            "apex_cas.state_io.save_scf_summary"
        ) as mocked_save_scf, patch(
            "apex_cas.CAS_builder.build_cas_from_mean_field",
            return_value=fake_cas,
        ) as mocked_build_cas, patch(
            "apex_cas.main._synchronize_cas_labels"
        ), patch(
            "apex_cas.main._save_and_validate_compute_outputs"
        ) as mocked_save_outputs, patch(
            "apex_cas.main._generate_compute_visualizations"
        ) as mocked_viz:
            run_compute(args)

        mocked_run_scf.assert_called_once()
        mocked_save_scf.assert_called_once()
        mocked_build_cas.assert_called_once()
        mocked_save_outputs.assert_called_once()
        mocked_viz.assert_called_once()

    def test_run_fcidump_writes_stage_summary(self):
        args = SimpleNamespace(
            case_dir="demo_case",
            spin_projection=0.5,
            output=None,
            reference_fcidump=None,
            active_file=None,
            zero_ecore=True,
        )
        fake_cas = SimpleNamespace(
            n_electrons=9,
            n_orbitals=9,
            cpt_cas_type="uno",
            target_spin=0.5,
            orbital_labels_full=[f"orb_{i}" for i in range(60)],
            occupations_full=[1.0] * 60,
            mo_coeff_full="coeff",
        )
        with patch(
            "apex_cas.state_io.load_cas_state",
            return_value=(fake_cas, "mol", "mf"),
        ), patch(
            "apex_cas.state_io.find_chkfile",
            return_value="/tmp/demo.chk",
        ), patch(
            "apex_cas.main._resolve_fcidump_selection_path",
            return_value="/tmp/demo_selection.txt",
        ), patch(
            "apex_cas.selection_io.load_active_selection",
            return_value=([1, 2, 3], 5),
        ), patch(
            "apex_cas.FCIDUMP_generator.generate_fcidump_from_selection",
            return_value="/tmp/FCIDUMP.demo",
        ) as mocked_gen, patch(
            "apex_cas.state_io.save_fcidump_summary"
        ) as mocked_save, patch(
            "apex_cas.main._print_selected_orbitals"
        ):
            run_fcidump(args)

        mocked_gen.assert_called_once()
        mocked_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
