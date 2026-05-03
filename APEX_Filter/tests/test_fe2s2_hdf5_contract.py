"""Schema-level regression checks for the committed Fe2S2 ox HDF5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
from shared.settings_payloads import (
    ACTIVE_SPACE_CC_RECORD_ONLY_KEYS,
    find_effective_parameter_leaks,
    missing_normalized_settings_sections,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "examples" / "fe2s2"
SESSION_DIR = CASE_DIR / "filter_session"
CONTROL_SOURCE = str(SESSION_DIR / "method_controls.yaml")
def _assert_no_method_identity_leakage(settings: dict, *, record_only_keys: set[str] | None = None) -> None:
    assert not find_effective_parameter_leaks(
        settings,
        record_only_keys=record_only_keys,
    )


def _assert_normalized_settings_sections_present(settings: dict) -> None:
    assert not missing_normalized_settings_sections(settings)


def _load_settings_json(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return json.loads(h5["metadata"].attrs["settings_json"])


def test_fe2s2_step3_hdf5_contract():
    path = SESSION_DIR / "step3_uhf" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_uhf.h5"
    with h5py.File(path, "r") as h5:
        assert set(h5.keys()) == {
            "active_space_mapping",
            "density_matrices",
            "diagnostics",
            "metadata",
            "molecule",
            "orbitals",
        }
        assert "dm_a" in h5["density_matrices"]
        assert "dm_b" in h5["density_matrices"]
        assert "active_indices" in h5["active_space_mapping"]
        settings = json.loads(h5["metadata"].attrs["settings_json"])
        _assert_normalized_settings_sections_present(settings)
        assert settings["theory"] == "UHF"
        assert settings["control_source"] == CONTROL_SOURCE
        assert settings["requested_config"]["theory"] == "UHF"
        assert settings["effective_method"]["theory"] == "UHF"
        _assert_no_method_identity_leakage(settings)
        assert "stabilize_cycles" in settings["effective_parameters"]


def test_fe2s2_step6_hdf5_contract():
    path = SESSION_DIR / "step6_ccsdt" / "scripts" / "Fe1↑Fe2↓_2xFe(III)_d:none_ccsdt_results.h5"
    with h5py.File(path, "r") as h5:
        assert set(h5.keys()) == {
            "active_space_mapping",
            "amplitudes",
            "density_matrices",
            "diagnostics",
            "metadata",
            "molecule",
            "orbitals",
        }
        assert "dm1a_mo" in h5["density_matrices"]
        assert "dm1b_mo" in h5["density_matrices"]
        settings = json.loads(h5["metadata"].attrs["settings_json"])
        _assert_normalized_settings_sections_present(settings)
        assert settings["theory"] == "UCCSDT"
        assert settings["control_source"] == CONTROL_SOURCE
        assert settings["requested_config"]["theory"] == "UCCSDT"
        assert settings["requested_config"]["basis_set"] == "cc-pVDZ"
        assert settings["effective_method"]["theory"] == "UCCSDT"
        _assert_no_method_identity_leakage(settings, record_only_keys=set(ACTIVE_SPACE_CC_RECORD_ONLY_KEYS))
        assert "conv_tol" in settings["effective_parameters"]


def test_fe2s2_step7_hdf5_contract():
    path = SESSION_DIR / "step7_dmrg_basis" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_dmrg_basis.h5"
    with h5py.File(path, "r") as h5:
        assert set(h5.keys()) == {
            "active_space_mapping",
            "metadata",
            "molecule",
            "orbitals",
        }
        assert "ordering" in h5["orbitals"]
        assert "pairs" in h5["orbitals"]
        settings = json.loads(h5["metadata"].attrs["settings_json"])
        _assert_normalized_settings_sections_present(settings)
        assert settings["theory"] == "DMRG basis"
        assert settings["control_source"] == CONTROL_SOURCE
        assert settings["requested_config"]["theory"] == "DMRG basis"
        assert settings["effective_method"]["theory"] == "DMRG basis"
        _assert_no_method_identity_leakage(settings)
        assert "cc_conv_tol" in settings["effective_parameters"]


def test_fe2s2_step8_hdf5_contract_for_full_m_ladder():
    result_dir = SESSION_DIR / "step8_dmrg" / "results"
    h5_paths = sorted(result_dir.glob("*_dmrg.h5"))
    assert len(h5_paths) == 10

    expected_bond_dims = {100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400}
    ladders_by_state: dict[str, set[int]] = {}

    for path in h5_paths:
        with h5py.File(path, "r") as h5:
            assert set(h5.keys()) == {
                "active_space_mapping",
                "basis_state",
                "density_matrices",
                "dmrg_diagnostics",
                "metadata",
                "molecule",
                "schedule",
            }
            meta = h5["metadata"]
            sched = h5["schedule"]
            settings = json.loads(meta.attrs["settings_json"])
            _assert_normalized_settings_sections_present(settings)
            assert settings["theory"] == "DMRG"
            assert settings["control_source"] == CONTROL_SOURCE
            assert settings["requested_config"]["theory"] == "DMRG"
            assert settings["effective_method"]["theory"] == "DMRG"
            _assert_no_method_identity_leakage(settings)
            label = str(meta.attrs["label"])
            bond_dim = int(meta.attrs["bond_dim"])
            ladders_by_state.setdefault(label, set()).add(bond_dim)
            assert settings["effective_parameters"]["bond_dim"] == bond_dim
            assert "n_sweeps" in settings["effective_parameters"]
            assert "twosite_to_onesite" in settings["effective_parameters"]
            assert "dav_max_iter" in settings["effective_parameters"]
            assert "dav_def_max_size" in settings["effective_parameters"]
            assert "dav_rel_conv_thrd" in settings["effective_parameters"]
            assert "dav_type" in settings["effective_parameters"]
            assert int(meta.attrs["n_sweeps"]) == 30
            assert int(sched.attrs["n_sweeps"]) == 30
            assert len(sched["bond_dims"]) == 30
            assert len(sched["noises"]) == 30
            assert len(sched["thresholds"]) == 30

    assert set(ladders_by_state) == {"Fe1↑Fe2↓|2xFe(III)|d:none"}
    assert all(bond_dims == expected_bond_dims for bond_dims in ladders_by_state.values())


def test_fe2s2_hdf5_authority_audit_across_step3_to_step8():
    audited_paths = [
        SESSION_DIR / "step3_uhf" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_uhf.h5",
        SESSION_DIR / "step3_uhf" / "results" / "Fe1↓Fe2↑_2xFe(III)_d:none_uhf.h5",
        SESSION_DIR / "step6_ccsdt" / "scripts" / "Fe1↑Fe2↓_2xFe(III)_d:none_ccsdt_results.h5",
        SESSION_DIR / "step6_ccsdt" / "scripts" / "Fe1↓Fe2↑_2xFe(III)_d:none_ccsdt_results.h5",
        SESSION_DIR / "step7_dmrg_basis" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_dmrg_basis.h5",
        SESSION_DIR / "step7_dmrg_basis" / "results" / "Fe1↓Fe2↑_2xFe(III)_d:none_dmrg_basis.h5",
        *sorted((SESSION_DIR / "step8_dmrg" / "results").glob("*_dmrg.h5")),
    ]
    audited_paths = [path for path in audited_paths if path.exists()]
    assert len(audited_paths) == 14

    for path in audited_paths:
        settings = _load_settings_json(path)
        _assert_normalized_settings_sections_present(settings)
        _assert_no_method_identity_leakage(
            settings,
            record_only_keys=set(ACTIVE_SPACE_CC_RECORD_ONLY_KEYS)
            if settings["effective_method"].get("theory") == "UCCSDT"
            else None,
        )
