"""Tests for extracted state/selection I/O helpers."""

import os
import json
import tempfile
import unittest
from types import SimpleNamespace

from apex_cas import (
    CAS,
    ComputationSettings,
    save_cas_state,
    save_dmrg_summary,
    save_fcidump_summary,
    find_chkfile,
    generate_selection_file,
    load_active_selection,
    load_scf_summary,
    save_scf_summary,
)
from apex_cas.state_io import _decode_h5_string


class TestSelectionIO(unittest.TestCase):
    def test_generate_and_load_selection_roundtrip(self):
        cas = CAS()
        cas.active_indices = [8, 2, 5]
        cas.n_electrons = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "selection.txt")
            generate_selection_file(cas, path)
            indices, n_electrons = load_active_selection(path)

        self.assertEqual(indices, [2, 5, 8])
        self.assertEqual(n_electrons, 10)

class TestStateIO(unittest.TestCase):
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
            path = save_scf_summary(
                mf,
                output_dir,
                stem="demo",
                settings=settings,
                charge=-2,
                target_spin=0.0,
            )
            payload = load_scf_summary(output_dir, "demo")

        self.assertTrue(path.endswith("demo_scf_info.json"))
        self.assertEqual(payload["energy"], -123.456)
        self.assertEqual(payload["E_solvent"], -0.12)
        self.assertEqual(payload["charge"], -2)
        self.assertIn("settings", payload)
        self.assertIn("scf", payload["settings"])
        self.assertEqual(payload["settings"]["scf"]["scf_method"], "uks")
        self.assertEqual(payload["settings"]["scf"]["xc_functional"], "BP86")
        self.assertEqual(payload["settings"]["scf"]["solvation_model"], "none")
        self.assertNotIn("solvation_epsilon", payload["settings"]["scf"])

    def test_save_cas_state_writes_separate_cas_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            mf = SimpleNamespace(
                e_tot=-123.456,
                converged=True,
                scf_summary={},
            )
            save_scf_summary(
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

            save_cas_state(
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
        self.assertEqual(cas_payload["n_electrons"], 9)
        self.assertEqual(cas_payload["n_orbitals"], 9)
        self.assertEqual(cas_payload["settings"]["cas_build"]["localization_method"], "pm")

    def test_save_fcidump_summary_writes_stage_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            path = save_fcidump_summary(
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
                reference_fcidump=None,
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

    def test_save_dmrg_summary_writes_stage_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dmrg_dir = os.path.join(tmpdir, "outputs", "fcidump", "dmrg")
            path = save_dmrg_summary(
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


if __name__ == "__main__":
    unittest.main()
