"""Acceptance-style checks for the committed Fe2S2 ox APEX_CAS outputs."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
from shared.settings_payloads import (
    find_effective_parameter_leaks,
    missing_normalized_settings_sections,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "examples" / "fe2s2"
OUTPUTS_DIR = CASE_DIR / "outputs"
STEM = "C4H12Fe2S6_uks_BP86_tzp-dkh"
def _assert_no_method_identity_leakage(settings: dict, *, record_only_keys: set[str] | None = None) -> None:
    assert not find_effective_parameter_leaks(
        settings,
        record_only_keys=record_only_keys,
    )


def _assert_normalized_settings_sections_present(settings: dict) -> None:
    assert not missing_normalized_settings_sections(settings)


def test_fe2s2_scf_and_cas_json_contract():
    scf_info = json.loads((OUTPUTS_DIR / "scf" / f"{STEM}_scf_info.json").read_text())
    cas_info = json.loads((OUTPUTS_DIR / "scf" / f"{STEM}_cas_info.json").read_text())

    assert scf_info["converged"] is True
    assert scf_info["charge"] == -2
    assert scf_info["target_spin"] == 0.0
    assert scf_info["basis_set_default"] == "tzp-dkh"
    assert scf_info["basis_set_per_element"] == {}
    assert scf_info["scf_method"] == "uks"
    assert scf_info["xc_functional"] == "BP86"
    assert scf_info["relativistic"] == "sf-x2c"
    assert scf_info["solvation_model"] == "none"
    assert scf_info["conv_tol"] == 1e-12
    assert scf_info["max_cycle"] == 200
    assert scf_info["scf_spin"] == 5.0
    assert "settings" in scf_info
    _assert_normalized_settings_sections_present(scf_info["settings"]["scf"])
    assert scf_info["settings"]["scf"]["scf_method"] == "uks"
    assert scf_info["settings"]["scf"]["xc_functional"] == "BP86"
    assert scf_info["settings"]["scf"]["requested_config"]["scf_method"] == "uks"
    assert scf_info["settings"]["scf"]["effective_method"]["scf_method"] == "uks"
    _assert_no_method_identity_leakage(scf_info["settings"]["scf"])
    assert scf_info["settings"]["scf"]["effective_parameters"]["max_cycle"] == 200

    assert cas_info["results"]["n_electrons"] == 10
    assert cas_info["results"]["n_orbitals"] == 10
    assert cas_info["effective_method"]["cpt_cas_type"] == "uno"
    assert cas_info["effective_method"]["selection"]["method"] == "noon"
    assert len(cas_info["results"]["active_indices"]) == 10


def test_fe2s2_cas_hdf5_contract():
    path = OUTPUTS_DIR / "orbitals" / f"{STEM}_cas_data.h5"
    with h5py.File(path, "r") as h5:
        assert set(h5.keys()) == {
            "active_space_mapping",
            "metadata",
            "mo_coeff_alpha",
            "mo_coeff_beta",
            "mo_coeff_full",
            "molecule",
            "occupations",
            "occupations_full",
            "orbital_labels",
            "orbital_labels_full",
        }
        assert "settings_json" in h5["metadata"].attrs
        assert "basis_set_default" in h5["metadata"].attrs
        settings = json.loads(h5["metadata"].attrs["settings_json"])
        _assert_normalized_settings_sections_present(settings)
        assert settings["requested_config"]["cpt_cas_type"] == "uno"
        assert settings["requested_config"]["projection_threshold"] == 0.3
        assert settings["effective_method"]["localization"]["method"] == "pm"
        assert settings["effective_method"]["selection"]["method"] == "noon"
        _assert_no_method_identity_leakage(settings)
        assert settings["effective_parameters"]["selection"]["occ_lo"] == 0.02
        assert "projection_threshold" not in settings["effective_parameters"]
        assert "active_indices" in h5["metadata"]
        assert "projection_weights" in h5["metadata"]
        assert "active_indices" in h5["active_space_mapping"]
        assert "orbital_labels" in h5["active_space_mapping"]
        assert "atom_symbols" in h5["molecule"]
        assert "atom_positions" in h5["molecule"]


def test_fe2s2_fcidump_and_testcas_contract():
    fcidump_info = json.loads((OUTPUTS_DIR / "fcidump" / f"{STEM}_fcidump_info.json").read_text())
    dmrg_info = json.loads((OUTPUTS_DIR / "fcidump" / "dmrg" / f"{STEM}_sz_M500_dmrg_info.json").read_text())

    assert Path(fcidump_info["fcidump_path"]).name == f"FCIDUMP.{STEM}"
    assert fcidump_info["n_electrons"] == 30
    assert fcidump_info["n_orbitals"] == 20
    assert "selection_file" in fcidump_info
    _assert_normalized_settings_sections_present(fcidump_info["settings"]["fcidump"])
    assert fcidump_info["settings"]["fcidump"]["requested_config"]["zero_ecore"] is True
    _assert_no_method_identity_leakage(fcidump_info["settings"]["fcidump"])
    assert fcidump_info["settings"]["fcidump"]["effective_parameters"]["spin_projection"] == 0.0

    assert dmrg_info["bond_dim"] == 500
    assert dmrg_info["symm_type"] == "sz"
    assert Path(dmrg_info["results_h5"]).name == f"{STEM}_sz_M500_dmrg_results.h5"
    _assert_normalized_settings_sections_present(dmrg_info["settings"]["dmrg"])
    assert dmrg_info["settings"]["dmrg"]["requested_config"]["bond_dim"] == 500
    assert dmrg_info["settings"]["dmrg"]["effective_method"]["symm_type"] == "sz"
    _assert_no_method_identity_leakage(dmrg_info["settings"]["dmrg"])
    assert dmrg_info["settings"]["dmrg"]["effective_parameters"]["bond_dim"] == 500

    h5_path = OUTPUTS_DIR / "fcidump" / "dmrg" / f"{STEM}_sz_M500_dmrg_results.h5"
    with h5py.File(h5_path, "r") as h5:
        assert set(h5.keys()) == {"dmrg_1rdm", "metadata", "noon", "noon_labels"}
        meta = h5["metadata"].attrs
        assert meta["bond_dim"] == 500
        assert meta["symm_type"] == "sz"
        assert meta["n_elec"] == 30
        assert meta["n_orb"] == 20


def test_fe2s2_orbital_report_selection_and_fcidump_files_exist():
    report = OUTPUTS_DIR / "orbitals" / f"{STEM}_orbital_report.md"
    selection = OUTPUTS_DIR / "orbitals" / f"{STEM}_selection.txt"
    fcidump = OUTPUTS_DIR / "fcidump" / f"FCIDUMP.{STEM}"
    ecore = OUTPUTS_DIR / "fcidump" / f"FCIDUMP.{STEM}.ecore"

    assert report.exists()
    assert selection.exists()
    assert fcidump.exists()
    assert ecore.exists()

    report_text = report.read_text()
    selection_text = selection.read_text()
    assert "# Orbital Report" in report_text
    assert "n-electrons: 30" in selection_text
    assert "n-orbital: 20" in selection_text
