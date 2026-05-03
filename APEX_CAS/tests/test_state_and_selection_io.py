"""Regression tests for state/selection I/O helpers and readback entry points."""

import os
import json
import tempfile
import unittest
from types import SimpleNamespace

from apex_cas import (
    CAS,
    ComputationSettings,
)
from apex_cas.selection_io import _generate_selection_file
from shared.chkfiles import find_chkfile
from shared.selection_io import load_active_selection
from shared.settings_payloads import (
    ACTIVE_SPACE_CC_RECORD_ONLY_KEYS,
    build_base_settings_payload,
    extend_settings_payload,
    find_effective_parameter_leaks,
    missing_normalized_settings_sections,
    normalize_settings_payload,
)
from apex_cas.state_io import _decode_h5_string
from apex_cas.state_io import (
    _read_mf_settings_from_h5,
    _read_mf_settings_from_scf_summary,
    _load_scf_summary,
    _save_cas_state,
    _save_dmrg_summary,
    _save_fcidump_summary,
    _save_scf_summary,
)


class TestSelectionIO(unittest.TestCase):
    def test_generate_and_load_selection_roundtrip(self):
        cas = CAS()
        cas.active_indices = [8, 2, 5]
        cas.n_electrons = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "selection.txt")
            _generate_selection_file(cas, path)
            indices, n_electrons = load_active_selection(path)

        self.assertEqual(indices, [2, 5, 8])


class TestSettingsPayloadHelpers(unittest.TestCase):
    def test_build_base_settings_payload_accepts_dataclass_settings(self):
        settings = ComputationSettings(conv_tol=1e-9, max_cycle=123)
        payload = build_base_settings_payload(
            settings,
            control_source="/tmp/method_controls.yaml",
            theory="UHF",
            stabilize_cycles=4,
        )
        self.assertEqual(payload["conv_tol"], 1e-9)
        self.assertEqual(payload["max_cycle"], 123)
        self.assertEqual(payload["control_source"], "/tmp/method_controls.yaml")
        self.assertEqual(payload["theory"], "UHF")
        self.assertEqual(payload["stabilize_cycles"], 4)

    def test_build_base_settings_payload_accepts_mapping_settings(self):
        payload = build_base_settings_payload(
            {"conv_tol": 1e-8, "max_cycle": 200},
            theory="DMRG basis",
            cc_conv_tol=1e-10,
        )
        self.assertEqual(payload["conv_tol"], 1e-8)
        self.assertEqual(payload["max_cycle"], 200)
        self.assertEqual(payload["theory"], "DMRG basis")
        self.assertEqual(payload["cc_conv_tol"], 1e-10)

    def test_build_base_settings_payload_accepts_empty_source_settings(self):
        payload = build_base_settings_payload(
            None,
            control_source="/tmp/method_controls.yaml",
            theory="DMRG",
            bond_dim=500,
        )
        self.assertEqual(payload["control_source"], "/tmp/method_controls.yaml")
        self.assertEqual(payload["theory"], "DMRG")
        self.assertEqual(payload["bond_dim"], 500)

    def test_extend_settings_payload_preserves_none(self):
        self.assertIsNone(extend_settings_payload(None, theory="DMRG"))

    def test_extend_settings_payload_returns_copied_payload_with_overrides(self):
        payload = {"theory": "DMRG basis", "bond_dim": 500}
        extended = extend_settings_payload(payload, source_method="UCCSD-NO", bond_dim=800)
        self.assertEqual(
            extended,
            {
                "theory": "DMRG basis",
                "bond_dim": 800,
                "source_method": "UCCSD-NO",
            },
        )
        self.assertEqual(payload, {"theory": "DMRG basis", "bond_dim": 500})

    def test_load_selection_raises_value_error_when_header_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "selection.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# malformed selection\n")
                fh.write("n-orbital: 2\n")
                fh.write("1 2\n")

            with self.assertRaises(ValueError):
                load_active_selection(path)

class TestStateIO(unittest.TestCase):
    def test_normalize_settings_payload_preserves_flat_fields_and_adds_nested_authority(self):
        payload = {
            "theory": "DMRG",
            "control_source": "/tmp/method_controls.yaml",
            "bond_dim": 500,
            "schedule_mode": "benchmark",
        }

        normalized = normalize_settings_payload(payload)

        self.assertEqual(normalized["theory"], "DMRG")
        self.assertEqual(normalized["bond_dim"], 500)
        self.assertEqual(normalized["requested_config"]["bond_dim"], 500)
        self.assertEqual(normalized["effective_method"]["theory"], "DMRG")
        self.assertEqual(normalized["effective_method"]["schedule_mode"], "benchmark")
        self.assertEqual(normalized["effective_parameters"]["bond_dim"], 500)
        self.assertNotIn("control_source", normalized["requested_config"])

    def test_normalize_settings_payload_merges_existing_nested_blocks(self):
        payload = {
            "theory": "UHF",
            "conv_tol": 1e-8,
            "requested_config": {"conv_tol": 1e-8},
            "effective_method": {"theory": "UHF"},
            "effective_parameters": {"conv_tol": 1e-8},
        }

        normalized = normalize_settings_payload(payload)

        self.assertEqual(normalized["requested_config"], {"theory": "UHF", "conv_tol": 1e-8})
        self.assertEqual(normalized["effective_method"], {"theory": "UHF"})
        self.assertEqual(normalized["effective_parameters"], {"conv_tol": 1e-8})

    def test_normalize_settings_payload_classifies_method_identity_fields(self):
        payload = {
            "scf_method": "uks",
            "xc_functional": "BP86",
            "cpt_cas_type": "uno",
            "symm_type": "sz",
            "dmrg_mode": "benchmark",
            "conv_tol": 1e-10,
            "bond_dim": 500,
        }

        normalized = normalize_settings_payload(payload)

        assert normalized["effective_method"] == {
            "scf_method": "uks",
            "xc_functional": "BP86",
            "cpt_cas_type": "uno",
            "symm_type": "sz",
            "dmrg_mode": "benchmark",
        }
        assert normalized["effective_parameters"] == {
            "conv_tol": 1e-10,
            "bond_dim": 500,
        }

    def test_normalize_settings_payload_excludes_record_only_keys_from_effective_parameters(self):
        payload = {
            "theory": "UCCSDT",
            "basis_set": "cc-pVDZ",
            "conv_tol": 1e-8,
        }

        normalized = normalize_settings_payload(payload, record_only_keys=ACTIVE_SPACE_CC_RECORD_ONLY_KEYS)

        self.assertEqual(normalized["requested_config"]["basis_set"], "cc-pVDZ")
        self.assertNotIn("basis_set", normalized["effective_parameters"])
        self.assertEqual(normalized["effective_parameters"]["conv_tol"], 1e-8)

    def test_normalize_settings_payload_drops_record_only_keys_from_old_effective_block(self):
        payload = {
            "theory": "UCCSDT",
            "basis_set": "cc-pVDZ",
            "requested_config": {"basis_set": "cc-pVDZ"},
            "effective_parameters": {"basis_set": "cc-pVDZ", "conv_tol": 1e-8},
        }

        normalized = normalize_settings_payload(payload, record_only_keys=ACTIVE_SPACE_CC_RECORD_ONLY_KEYS)

        self.assertEqual(normalized["requested_config"]["basis_set"], "cc-pVDZ")
        self.assertNotIn("basis_set", normalized["effective_parameters"])
        self.assertEqual(normalized["effective_parameters"]["conv_tol"], 1e-8)

    def test_find_effective_parameter_leaks_reports_method_identity_and_record_only_keys(self):
        payload = {
            "effective_parameters": {
                "theory": "DMRG",
                "basis_set": "cc-pVDZ",
                "bond_dim": 500,
            }
        }

        leaks = find_effective_parameter_leaks(
            payload,
            record_only_keys=ACTIVE_SPACE_CC_RECORD_ONLY_KEYS,
        )

        self.assertEqual(leaks, {"theory", "basis_set"})

    def test_find_effective_parameter_leaks_accepts_flat_payloads(self):
        payload = {
            "theory": "UHF",
            "conv_tol": 1e-8,
        }

        leaks = find_effective_parameter_leaks(payload)

        self.assertEqual(leaks, set())

    def test_missing_normalized_settings_sections_reports_absent_blocks(self):
        self.assertEqual(
            missing_normalized_settings_sections({"requested_config": {}}),
            set(),
        )

    def test_missing_normalized_settings_sections_accepts_flat_payloads(self):
        self.assertEqual(
            missing_normalized_settings_sections({"theory": "UHF", "conv_tol": 1e-8}),
            set(),
        )

    def test_missing_normalized_settings_sections_reports_all_sections_for_none(self):
        self.assertEqual(
            missing_normalized_settings_sections(None),
            {"requested_config", "effective_method", "effective_parameters"},
        )

    def test_decode_h5_string_decodes_bytes_without_byte_prefix(self):
        self.assertEqual(_decode_h5_string(b"Fe1_3dxy"), "Fe1_3dxy")
        self.assertEqual(_decode_h5_string("S1_3pz"), "S1_3pz")

    def test_find_chkfile_single_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk = os.path.join(tmpdir, "test.chk")
            with open(chk, "wb") as f:
                f.write(b"abc")

            resolved = find_chkfile(tmpdir)

        self.assertEqual(os.path.basename(resolved), "test.chk")

    def test_find_chkfile_multiple_candidates_uses_largest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small = os.path.join(tmpdir, "small.chk")
            large = os.path.join(tmpdir, "large.chk")
            with open(small, "wb") as f:
                f.write(b"abc")
            with open(large, "wb") as f:
                f.write(b"abcdefghi")

            resolved = find_chkfile(tmpdir)

        self.assertEqual(os.path.basename(resolved), "large.chk")

    def test_save_and_load_scf_summary_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            mf = SimpleNamespace(
                e_tot=-123.456,
                converged=True,
                with_solvent=object(),
                scf_summary={"e_solvent": -0.12},
            )
            settings = ComputationSettings()
            settings.solvation_model = "none"
            settings.scf_method = "uks"
            settings.xc_functional = "BP86"
            path = _save_scf_summary(
                mf,
                output_dir,
                stem="demo",
                settings=settings,
                charge=-2,
                target_spin=0.0,
            )
            payload = _load_scf_summary(output_dir, "demo")

        self.assertTrue(path.endswith("demo_scf_info.json"))
        self.assertEqual(payload["energy"], -123.456)
        self.assertEqual(payload["E_solvent"], -0.12)
        self.assertEqual(payload["charge"], -2)
        self.assertIn("settings", payload)
        self.assertIn("scf", payload["settings"])
        self.assertEqual(payload["basis_set_default"], "def2-TZVP")
        self.assertEqual(
            payload["basis_set_per_element"],
            settings.basis_set_per_element,
        )
        self.assertEqual(payload["scf_method"], settings.scf_method)
        self.assertEqual(payload["xc_functional"], settings.xc_functional)
        self.assertEqual(payload["relativistic"], settings.relativistic)
        self.assertEqual(payload["solvation_model"], settings.solvation_model)
        self.assertEqual(payload["conv_tol"], settings.conv_tol)
        self.assertEqual(payload["max_cycle"], settings.max_cycle)
        self.assertEqual(payload["scf_spin"], settings.scf_spin)
        self.assertEqual(payload["settings"]["scf"]["scf_method"], "uks")
        self.assertEqual(payload["settings"]["scf"]["xc_functional"], "BP86")
        self.assertEqual(payload["settings"]["scf"]["solvation_model"], "none")
        self.assertNotIn("solvation_epsilon", payload["settings"]["scf"])
        self.assertEqual(payload["settings"]["scf"]["requested_config"]["scf_method"], "uks")
        self.assertEqual(payload["settings"]["scf"]["effective_method"]["scf_method"], "uks")
        self.assertEqual(payload["settings"]["scf"]["effective_parameters"]["max_cycle"], 2000)

    def test_read_mf_settings_from_scf_summary_recovers_runtime_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            mf = SimpleNamespace(
                e_tot=-123.456,
                converged=True,
                with_solvent=object(),
                scf_summary={"e_solvent": -0.12},
            )
            settings = ComputationSettings()
            settings.scf_method = "uks"
            settings.xc_functional = "BP86"
            settings.relativistic = "sf-x2c"
            settings.solvation_model = "ddcosmo"
            settings.solvation_epsilon = 6.0
            settings.density_fit = True
            settings.grids_level = 5
            settings.frac_occ = True
            settings.smearing_method = "fermi"
            settings.smearing_sigma = 0.02

            _save_scf_summary(
                mf,
                output_dir,
                stem="demo",
                settings=settings,
                charge=-2,
                target_spin=0.0,
            )

            restored = _read_mf_settings_from_scf_summary(output_dir, "demo")

        self.assertIsNotNone(restored)
        self.assertEqual(restored.scf_method, "uks")
        self.assertEqual(restored.xc_functional, "BP86")
        self.assertEqual(restored.relativistic, "sf-x2c")
        self.assertEqual(restored.solvation_model, "ddcosmo")
        self.assertEqual(restored.solvation_epsilon, 6.0)
        self.assertTrue(restored.density_fit)
        self.assertEqual(restored.grids_level, 5)
        self.assertTrue(restored.frac_occ)
        self.assertEqual(restored.smearing_method, "fermi")
        self.assertAlmostEqual(restored.smearing_sigma, 0.02)

    def test_read_mf_settings_from_h5_recovers_metadata_settings(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "demo_cas_data.h5")
            with h5py.File(h5_path, "w") as f:
                meta = f.create_group("metadata")
                meta.attrs["scf_method"] = "uks"
                meta.attrs["xc_functional"] = "BP86"
                meta.attrs["relativistic"] = "sf-x2c"
                meta.attrs["solvation_model"] = "ddcosmo"
                meta.attrs["solvation_epsilon"] = 5.0
                meta.attrs["density_fit"] = True
                meta.attrs["grids_level"] = 4
                meta.attrs["frac_occ"] = True
                meta.attrs["smearing_method"] = "fermi"
                meta.attrs["smearing_sigma"] = 0.03

            restored = _read_mf_settings_from_h5(h5_path)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.scf_method, "uks")
        self.assertEqual(restored.xc_functional, "BP86")
        self.assertEqual(restored.relativistic, "sf-x2c")
        self.assertEqual(restored.solvation_model, "ddcosmo")
        self.assertEqual(restored.solvation_epsilon, 5.0)
        self.assertTrue(restored.density_fit)
        self.assertEqual(restored.grids_level, 4)
        self.assertTrue(restored.frac_occ)
        self.assertEqual(restored.smearing_method, "fermi")
        self.assertAlmostEqual(restored.smearing_sigma, 0.03)

    def test_save_cas_state_writes_separate_cas_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            mf = SimpleNamespace(
                e_tot=-123.456,
                converged=True,
                scf_summary={},
            )
            _save_scf_summary(
                mf,
                output_dir,
                stem="demo",
                settings=None,
                charge=-2,
                target_spin=0.0,
            )

            cas = CAS()
            cas.mo_coeff_full = None
            cas.occupations_full = None
            cas.mo_coeff_alpha = None
            cas.mo_coeff_beta = None
            cas.occupations = None
            cas.orbital_labels_full = []
            cas.orbital_labels = []
            cas.n_electrons = 9
            cas.n_orbitals = 9
            cas.cpt_cas_type = "uno"
            cas.source_method = "UKS-BP86/UNO"
            cas.description = "test"
            cas.selection_method = "noon"
            cas.active_indices = [1, 2, 3]
            cas.projection_weights = None
            cas.projection_weights_metal = None
            cas.projection_weights_bridging = None
            settings_payload = {
                "cpt_cas_type": "uno",
                "localization_method": "pm",
                "projection_threshold": 0.3,
            }

            _save_cas_state(
                cas,
                mol=None,
                mf=mf,
                output_dir=output_dir,
                stem="demo",
                settings=None,
                charge=-2,
                target_spin=0.0,
                settings_payload=settings_payload,
            )

            scf_info_path = os.path.join(output_dir, "scf", "demo_scf_info.json")
            cas_info_path = os.path.join(output_dir, "scf", "demo_cas_info.json")

            self.assertTrue(os.path.isfile(scf_info_path))
            self.assertTrue(os.path.isfile(cas_info_path))

            with open(scf_info_path) as f:
                scf_payload = json.load(f)
            with open(cas_info_path) as f:
                cas_payload = json.load(f)

        self.assertEqual(scf_payload["energy"], -123.456)
        self.assertTrue(scf_payload["converged"])
        self.assertNotIn("n_electrons", scf_payload)
        self.assertEqual(cas_payload["results"]["n_electrons"], 9)
        self.assertEqual(cas_payload["results"]["n_orbitals"], 9)
        self.assertEqual(cas_payload["requested_config"]["cas_build"]["cpt_cas_type"], "uno")
        self.assertEqual(cas_payload["requested_config"]["cas_build"]["localization_method"], "pm")
        self.assertEqual(cas_payload["requested_config"]["cas_build"]["projection_threshold"], 0.3)
        self.assertEqual(cas_payload["effective_method"]["cpt_cas_type"], "uno")
        self.assertEqual(cas_payload["effective_method"]["localization"]["method"], "pm")
        self.assertEqual(cas_payload["effective_method"]["selection"]["method"], "noon")
        self.assertEqual(cas_payload["effective_method"]["selection"]["parameters"]["occ_lo"], 0.02)
        self.assertEqual(cas_payload["effective_parameters"]["localization"]["pop_method"], "mulliken")
        self.assertEqual(cas_payload["effective_parameters"]["selection"]["occ_lo"], 0.02)

    def test_save_cas_state_writes_molecule_and_mapping_groups(self):
        import h5py
        import numpy as np

        class _FakeMol:
            elements = ["Fe", "S"]
            _atom = [("Fe", (0.0, 0.0, 0.0)), ("S", (1.0, 0.0, 0.0))]

            def atom_coords(self):
                return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

            def dumps(self):
                return '{"fake": "mol"}'

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            mf = SimpleNamespace(
                e_tot=-123.456,
                converged=True,
                scf_summary={},
            )
            settings = ComputationSettings()
            settings.scf_method = "uks"
            settings.xc_functional = "BP86"
            settings.relativistic = "sf-x2c"
            settings.solvation_model = "ddcosmo"
            settings.solvation_epsilon = 4.0
            settings.basis_set_default = "tzp-dkh"
            settings.basis_set_per_element = {"Fe": "tzp-dkh", "S": "tzp-dkh"}
            settings.density_fit = True
            settings.grids_level = 5

            cas = CAS()
            cas.mo_coeff_full = None
            cas.occupations_full = None
            cas.mo_coeff_alpha = None
            cas.mo_coeff_beta = None
            cas.occupations = None
            cas.orbital_labels_full = ["Fe1_3dxy", "S1_3pz"]
            cas.orbital_labels = ["Fe1_3dxy"]
            cas.n_electrons = 9
            cas.n_orbitals = 9
            cas.cpt_cas_type = "uno"
            cas.source_method = "UKS-BP86/UNO"
            cas.description = "test"
            cas.selection_method = "noon"
            cas.active_indices = [1]
            cas.projection_weights = None
            cas.projection_weights_metal = None
            cas.projection_weights_bridging = None

            _save_cas_state(
                cas,
                mol=_FakeMol(),
                mf=mf,
                output_dir=output_dir,
                stem="demo",
                settings=settings,
                charge=-2,
                target_spin=0.0,
                settings_payload={"localization_method": "pm"},
            )

            h5_path = os.path.join(output_dir, "orbitals", "demo_cas_data.h5")
            with h5py.File(h5_path, "r") as f:
                self.assertIn("molecule", f)
                self.assertIn("active_space_mapping", f)
                self.assertEqual(f["molecule"].attrs["charge"], -2)
                self.assertEqual(f["molecule"].attrs["basis_set_default"], "tzp-dkh")
                self.assertEqual(f["molecule"].attrs["serialized_solver_mol"], '{"fake": "mol"}')
                self.assertIn("atom_symbols", f["molecule"])
                self.assertIn("atom_positions", f["molecule"])
                self.assertIn("active_indices", f["active_space_mapping"])
                self.assertIn("orbital_labels", f["active_space_mapping"])
                self.assertIn("orbital_labels_full", f["active_space_mapping"])
                self.assertIn("settings_json", f["metadata"].attrs)
                self.assertIn("basis_set_per_element_json", f["metadata"].attrs)
                settings_json = json.loads(f["metadata"].attrs["settings_json"])
                self.assertEqual(settings_json["requested_config"]["localization_method"], "pm")
                self.assertEqual(settings_json["effective_method"]["cpt_cas_type"], "uno")
                self.assertEqual(settings_json["effective_method"]["localization"]["method"], "pm")
                self.assertEqual(settings_json["effective_method"]["selection"]["method"], "noon")
                self.assertEqual(settings_json["effective_parameters"]["selection"]["occ_lo"], 0.02)
                self.assertNotIn("projection_threshold", settings_json["effective_parameters"])

    def test_save_fcidump_summary_writes_stage_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            path = _save_fcidump_summary(
                output_dir,
                "demo",
                fcidump_path=os.path.join(output_dir, "fcidump", "FCIDUMP.demo"),
                selection_file=os.path.join(output_dir, "orbitals", "demo_selection.txt"),
                n_electrons=23,
                n_orbitals=16,
                ms2=1,
                target_spin=0.5,
                zero_ecore=True,
                frozen_core_indices=[0, 1, 2],
                settings_payload={"spin_projection": 0.5, "zero_ecore": True},
            )
            with open(path) as f:
                payload = json.load(f)

        self.assertTrue(path.endswith("demo_fcidump_info.json"))
        self.assertEqual(payload["n_electrons"], 23)
        self.assertEqual(payload["n_orbitals"], 16)
        self.assertEqual(payload["ms2"], 1)
        self.assertEqual(payload["frozen_core_indices"], [0, 1, 2])
        self.assertEqual(payload["settings"]["fcidump"]["spin_projection"], 0.5)
        self.assertEqual(payload["settings"]["fcidump"]["requested_config"]["spin_projection"], 0.5)
        self.assertEqual(payload["settings"]["fcidump"]["effective_parameters"]["zero_ecore"], True)

    def test_save_dmrg_summary_writes_stage_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dmrg_dir = os.path.join(tmpdir, "outputs", "fcidump", "dmrg")
            path = _save_dmrg_summary(
                dmrg_dir,
                "demo_sz_M500",
                fcidump_path=os.path.join(tmpdir, "outputs", "fcidump", "FCIDUMP.demo"),
                h5_path=os.path.join(dmrg_dir, "demo_sz_M500_dmrg_results.h5"),
                noon_plot_path=os.path.join(dmrg_dir, "demo_sz_M500_noon_plot.png"),
                bond_dim=500,
                symm_type="sz",
                n_orb=16,
                n_elec=23,
                ms2=1,
                e_active=-10.0,
                e_core=-20.0,
                e_total=-30.0,
                wall_time_s=12.5,
                spin_squared=None,
                settings_payload={"bond_dim": 500, "symm_type": "sz", "stack_mem_gb": None},
            )
            with open(path) as f:
                payload = json.load(f)

        self.assertTrue(path.endswith("demo_sz_M500_dmrg_info.json"))
        self.assertEqual(payload["bond_dim"], 500)
        self.assertEqual(payload["symm_type"], "sz")
        self.assertEqual(payload["e_total"], -30.0)
        self.assertNotIn("spin_squared", payload)
        self.assertEqual(payload["settings"]["dmrg"]["bond_dim"], 500)
        self.assertEqual(payload["settings"]["dmrg"]["requested_config"]["bond_dim"], 500)
        self.assertEqual(payload["settings"]["dmrg"]["effective_method"]["symm_type"], "sz")
