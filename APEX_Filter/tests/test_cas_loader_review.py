"""Regression tests for Step 1 authority-preserving ClusterInfo loading."""

import numpy as np

import apex_filter.CAS_loader as cas_loader
from apex_filter.CAS_loader import (
    _reconcile_cas_with_fcidump_selection,
    _load_cluster_info,
)
from shared.models import BridgingAtom, CAS, ClusterInfo, MetalCenter
from shared.apex_cas_provenance import (
    build_effective_settings_from_apex_cas,
    load_apex_cas_provenance,
)
from shared.artifact_paths import (
    resolve_cluster_info_path,
    resolve_fcidump_path,
    resolve_structure_path,
)


def test_resolve_structure_path_prefers_unique_inputs_file(tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")

    resolved = resolve_structure_path({}, str(case_dir))

    assert resolved == str(structure.resolve())


def test_resolve_structure_path_honors_explicit_relative_config(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    structure = case_dir / "custom.pdb"
    structure.write_text("HEADER\n")

    resolved = resolve_structure_path({"structure_path": "custom.pdb"}, str(case_dir))

    assert resolved == str(structure.resolve())


def test_internal_load_cluster_info_prefers_parse_structure(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")
    cluster_info_path = inputs_dir / "cluster_cluster_info.yaml"
    cluster_info_path.write_text("cluster: {}\n")

    rich_cluster = ClusterInfo(
        metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
        bridging_atoms=[BridgingAtom(element="S", index=1, position=np.ones(3), bridged_metals=[0])],
        terminal_ligands=[],
        all_elements=["Fe", "S"],
        all_positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        formula="FeS",
        total_charge=-1,
        target_spin=1.5,
        symmetry_group="C3",
        cluster_info_path=str(cluster_info_path.resolve()),
        annotation_source="cluster_info_yaml",
    )

    monkeypatch.setattr(
        cas_loader,
        "_parse_structure",
        lambda *args, **kwargs: rich_cluster,
    )

    cluster = _load_cluster_info(
        case_dir=str(case_dir),
        config_raw={},
        charge=-1,
        spin=1.5,
        symmetry_group="C1",
    )

    assert len(cluster.bridging_atoms) == 1
    assert cluster.formula == "FeS"
    assert cluster.symmetry_group == "C1"


def test_load_apex_cas_provenance_uses_internal_chkfile_authority(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    scf_dir = case_dir / "outputs" / "scf"
    scf_dir.mkdir(parents=True)
    chk_path = scf_dir / "demo.chk"
    chk_path.write_bytes(b"x")

    monkeypatch.setattr(
        "shared.apex_cas_provenance._find_chkfile",
        lambda path: str(chk_path),
    )
    monkeypatch.setattr(
        "shared.apex_cas_provenance._load_json_if_exists",
        lambda path: {"path": path},
    )

    provenance = load_apex_cas_provenance(str(case_dir))

    assert provenance["stem"] == "demo"
    assert provenance["scf_info"]["path"].endswith("outputs/scf/demo_scf_info.json")
    assert provenance["cas_info"]["path"].endswith("outputs/scf/demo_cas_info.json")


def test_internal_load_cluster_info_requires_finalized_cluster_info(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")
    try:
        _load_cluster_info(
            case_dir=str(case_dir),
            config_raw={},
            charge=-1,
            spin=1.5,
            symmetry_group="C1",
        )
    except FileNotFoundError as exc:
        assert "finalized cluster_info.yaml" in str(exc)
    else:
        raise AssertionError("Expected missing cluster_info authority file to raise")


def test_internal_load_cluster_info_explicit_cluster_info_parse_failure_raises(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")
    cluster_info = inputs_dir / "cluster_info.yaml"
    cluster_info.write_text("cluster: {}\n")

    class FakeMol:
        _atom = [("Fe", (0.0, 0.0, 0.0))]
        formula = "Fe"

    monkeypatch.setattr(
        cas_loader,
        "_parse_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        _load_cluster_info(
            case_dir=str(case_dir),
            config_raw={"cluster_info_path": "inputs/cluster_info.yaml"},
            charge=0,
            spin=0.0,
            symmetry_group="C1",
        )
    except RuntimeError as exc:
        assert "finalized structure and cluster_info authority files" in str(exc)
    else:
        raise AssertionError("Expected explicit cluster_info parse failure to raise")


def test_resolve_cluster_info_path_auto_detects_unique_inputs_yaml(tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    cluster_info = inputs_dir / "cluster_cluster_info.yaml"
    cluster_info.write_text("cluster: {}\n")

    resolved = resolve_cluster_info_path({}, str(case_dir))

    assert resolved == str(cluster_info.resolve())


def test_internal_load_cluster_info_auto_detected_cluster_info_parse_failure_raises(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")
    cluster_info = inputs_dir / "cluster_cluster_info.yaml"
    cluster_info.write_text("cluster: {}\n")

    class FakeMol:
        _atom = [("Fe", (0.0, 0.0, 0.0))]
        formula = "Fe"

    monkeypatch.setattr(
        cas_loader,
        "_parse_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        _load_cluster_info(
            case_dir=str(case_dir),
            config_raw={},
            charge=0,
            spin=0.0,
            symmetry_group="C1",
        )
    except RuntimeError as exc:
        assert "finalized structure and cluster_info authority files" in str(exc)
    else:
        raise AssertionError("Expected auto-detected cluster_info parse failure to raise")


def test_resolve_fcidump_path_supports_glob_and_skips_ecore(tmp_path):
    case_dir = tmp_path / "case"
    fcidump_dir = case_dir / "outputs" / "fcidump"
    fcidump_dir.mkdir(parents=True)
    main = fcidump_dir / "FCIDUMP.demo"
    sidecar = fcidump_dir / "FCIDUMP.demo.ecore"
    info = fcidump_dir / "demo_fcidump_info.json"
    main.write_text("&FCI NORB=1,NELEC=2,MS2=0,\n")
    sidecar.write_text("0.0\n")
    info.write_text("{}\n")

    resolved = resolve_fcidump_path(
        {"fcidump_path": "outputs/fcidump/FCIDUMP.*"},
        str(case_dir),
    )

    assert resolved == str(main.resolve())


def test_reconcile_cas_with_fcidump_selection_uses_final_selection(tmp_path):
    case_dir = tmp_path / "case"
    fcidump_dir = case_dir / "outputs" / "fcidump"
    orbitals_dir = case_dir / "outputs" / "orbitals"
    fcidump_dir.mkdir(parents=True)
    orbitals_dir.mkdir(parents=True)

    fcidump_path = fcidump_dir / "FCIDUMP.demo"
    fcidump_path.write_text("&FCI NORB=16,NELEC=23,MS2=1,\n")

    selection_file = orbitals_dir / "demo_selection.txt"
    selection_file.write_text("n-electrons: 23\nn-orbital: 4\n1\n3\n5\n7\n")

    summary = fcidump_dir / "demo_fcidump_info.json"
    summary.write_text(
        '{'
        f'"fcidump_path":"{fcidump_path}",'
        f'"selection_file":"{selection_file}",'
        '"n_electrons":23,'
        '"n_orbitals":4'
        '}'
    )

    cas = CAS(
        n_electrons=9,
        n_orbitals=9,
        occupations_full=np.arange(10, dtype=float),
        orbital_labels_full=[f"orb_{i}" for i in range(10)],
        mo_coeff_full=np.eye(10),
    )
    fcid = type("FCI", (), {"norb": 16, "nelec": 23, "ms2": 1})()

    reconciled = _reconcile_cas_with_fcidump_selection(cas, str(fcidump_path), fcid)

    assert reconciled.n_electrons == 23
    assert reconciled.n_orbitals == 4
    assert reconciled.active_indices == [1, 3, 5, 7]
    assert reconciled.orbital_labels == ["orb_1", "orb_3", "orb_5", "orb_7"]
    assert np.allclose(reconciled.occupations, np.array([1.0, 3.0, 5.0, 7.0]))


def test_extract_effective_settings_prefers_apex_cas_sidecars(monkeypatch):
    fake_provenance = {
            "stem": "demo",
            "scf_info": {
                "settings": {
                    "scf": {
                        "scf_spin": 4.5,
                        "conv_tol": 1e-10,
                        "max_cycle": 100,
                        "solvation_model": "none",
                    }
                }
            },
            "cas_info": {
                "requested_config": {
                    "cas_build": {
                        "pm_pop_method": "mulliken",
                        "pm_conv_tol": 1e-8,
                        "pm_max_cycle": 500,
                        "newton_max_cycle": 10,
                        "newton_conv_tol": 1e-12,
                    }
                }
            },
        }

    from shared.models import ComputationSettings

    settings, provenance = build_effective_settings_from_apex_cas(
        config_raw={},
        case_dir="/tmp/fake_case",
        settings_cls=ComputationSettings,
        provenance_loader=lambda case_dir: fake_provenance,
    )

    assert settings.conv_tol == 1e-10
    assert settings.max_cycle == 100
    assert settings.solvation_model == "none"
    assert settings.scf_spin == 4.5
    assert settings.pm_pop_method == "mulliken"
    assert settings.pm_conv_tol == 1e-8
    assert settings.pm_max_cycle == 500
    assert settings.newton_max_cycle == 10
    assert settings.newton_conv_tol == 1e-12
    assert provenance["effective_settings_source"]["apex_cas_sidecar_keys"] == [
        "conv_tol",
        "max_cycle",
        "newton_conv_tol",
        "newton_max_cycle",
        "pm_conv_tol",
        "pm_max_cycle",
        "pm_pop_method",
        "scf_spin",
        "solvation_model",
    ]
    assert provenance["stem"] == "demo"


def test_extract_effective_settings_translates_localization_payload_and_clears_basis_map(monkeypatch):
    fake_provenance = {
            "stem": "demo",
            "scf_info": {
                "settings": {
                    "scf": {
                        "basis_set_default": "tzp-dkh",
                        "basis_set_per_element": {},
                        "conv_tol": 1e-12,
                        "max_cycle": 200,
                        "init_guess": "atom",
                    }
                }
            },
            "cas_info": {
                "effective_method": {
                    "localization": {
                        "method": "pm",
                        "parameters": {
                            "pop_method": "mulliken",
                            "conv_tol": 1e-8,
                            "conv_tol_grad": None,
                            "max_cycle": 100,
                            "exponent": 2,
                            "init_guess": "atomic",
                        },
                    }
                }
            },
        }

    from shared.models import ComputationSettings

    settings, provenance = build_effective_settings_from_apex_cas(
        config_raw={},
        case_dir="/tmp/fake_case",
        settings_cls=ComputationSettings,
        provenance_loader=lambda case_dir: fake_provenance,
    )

    assert settings.basis_set_per_element == {}
    assert settings.conv_tol == 1e-12
    assert settings.max_cycle == 200
    assert settings.init_guess == "atom"
    assert settings.pm_pop_method == "mulliken"
    assert settings.pm_conv_tol == 1e-8
    assert settings.pm_max_cycle == 100
    assert settings.pm_init_guess == "atomic"
    assert provenance["effective_settings_source"]["apex_cas_sidecar_keys"] == [
        "basis_set_default",
        "basis_set_per_element",
        "conv_tol",
        "init_guess",
        "max_cycle",
        "pm_conv_tol",
        "pm_conv_tol_grad",
        "pm_exponent",
        "pm_init_guess",
        "pm_max_cycle",
        "pm_pop_method",
    ]


def test_extract_effective_settings_prefers_effective_localization_payload_over_legacy_cas_build(monkeypatch):
    fake_provenance = {
            "stem": "demo",
            "scf_info": {
                "settings": {
                    "scf": {
                        "conv_tol": 1e-12,
                    }
                }
            },
            "cas_info": {
                "requested_config": {
                    "cas_build": {
                        "pm_pop_method": "lowdin",
                        "pm_conv_tol": 1e-6,
                        "pm_max_cycle": 50,
                    }
                },
                "effective_method": {
                    "localization": {
                        "method": "pm",
                        "parameters": {
                            "pop_method": "mulliken",
                            "conv_tol": 1e-8,
                            "conv_tol_grad": 1e-4,
                            "max_cycle": 100,
                            "exponent": 4,
                            "init_guess": "atomic",
                        },
                    }
                },
            },
        }

    from shared.models import ComputationSettings

    settings, provenance = build_effective_settings_from_apex_cas(
        config_raw={},
        case_dir="/tmp/fake_case",
        settings_cls=ComputationSettings,
        provenance_loader=lambda case_dir: fake_provenance,
    )

    assert settings.pm_pop_method == "mulliken"
    assert settings.pm_conv_tol == 1e-8
    assert settings.pm_conv_tol_grad == 1e-4
    assert settings.pm_max_cycle == 100
    assert settings.pm_exponent == 4
    assert settings.pm_init_guess == "atomic"
    assert provenance["effective_settings_source"]["apex_cas_sidecar_keys"] == [
        "conv_tol",
        "pm_conv_tol",
        "pm_conv_tol_grad",
        "pm_exponent",
        "pm_init_guess",
        "pm_max_cycle",
        "pm_pop_method",
    ]


def test_extract_effective_settings_ignores_retired_cas_info_settings_block(monkeypatch):
    fake_provenance = {
            "stem": "demo",
            "scf_info": {"settings": {"scf": {}}},
            "cas_info": {
                "settings": {
                    "cas_build": {
                        "pm_pop_method": "lowdin",
                        "pm_conv_tol": 1e-6,
                        "pm_max_cycle": 50,
                    }
                }
            },
        }

    from shared.models import ComputationSettings

    settings, provenance = build_effective_settings_from_apex_cas(
        config_raw={},
        case_dir="/tmp/fake_case",
        settings_cls=ComputationSettings,
        provenance_loader=lambda case_dir: fake_provenance,
    )

    assert settings.pm_pop_method == "mulliken"
    assert settings.pm_conv_tol == 1e-08
    assert settings.pm_max_cycle == 100
    assert provenance["effective_settings_source"]["apex_cas_sidecar_keys"] == []
