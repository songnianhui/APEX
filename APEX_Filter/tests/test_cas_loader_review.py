"""Regression tests for topology-preserving ClusterInfo loading."""

import numpy as np

from apex_filter.CAS_loader import (
    _extract_effective_settings,
    _reconcile_cas_with_fcidump_selection,
    _resolve_cluster_info_path,
    _resolve_fcidump_path,
    _resolve_structure_path,
    load_cluster_info,
)
from apex_filter.models import BridgingAtom, CAS, ClusterInfo, MetalCenter


def test_resolve_structure_path_prefers_unique_inputs_file(tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")

    resolved = _resolve_structure_path({}, str(case_dir))

    assert resolved == str(structure.resolve())


def test_resolve_structure_path_honors_explicit_relative_config(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    structure = case_dir / "custom.pdb"
    structure.write_text("HEADER\n")

    resolved = _resolve_structure_path({"structure_path": "custom.pdb"}, str(case_dir))

    assert resolved == str(structure.resolve())


def test_load_cluster_info_prefers_parse_structure(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")

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
    )

    monkeypatch.setattr(
        "apex_cas.parse_structure",
        lambda *args, **kwargs: rich_cluster,
    )

    cluster = load_cluster_info(
        case_dir=str(case_dir),
        config_raw={},
        mol=None,
        charge=-1,
        spin=1.5,
        symmetry_group="C1",
    )

    assert len(cluster.bridging_atoms) == 1
    assert cluster.formula == "FeS"
    assert cluster.symmetry_group == "C1"


def test_load_cluster_info_falls_back_to_mol_reconstruction(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    structure = inputs_dir / "cluster.xyz"
    structure.write_text("1\n\nH 0 0 0\n")

    class FakeMol:
        _atom = [("Fe", (0.0, 0.0, 0.0)), ("S", (1.0, 0.0, 0.0))]
        formula = "FeS"

    monkeypatch.setattr(
        "apex_cas.parse_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cluster = load_cluster_info(
        case_dir=str(case_dir),
        config_raw={},
        mol=FakeMol(),
        charge=-1,
        spin=1.5,
        symmetry_group="C1",
    )

    assert len(cluster.metals) == 1
    assert cluster.bridging_atoms == []
    assert cluster.formula == "FeS"


def test_load_cluster_info_explicit_cluster_info_parse_failure_raises(monkeypatch, tmp_path):
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
        "apex_cas.parse_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        load_cluster_info(
            case_dir=str(case_dir),
            config_raw={"cluster_info_path": "inputs/cluster_info.yaml"},
            mol=FakeMol(),
            charge=0,
            spin=0.0,
            symmetry_group="C1",
        )
    except RuntimeError as exc:
        assert "explicit cluster_info_path" in str(exc)
    else:
        raise AssertionError("Expected explicit cluster_info parse failure to raise")


def test_resolve_cluster_info_path_auto_detects_unique_inputs_yaml(tmp_path):
    case_dir = tmp_path / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    cluster_info = inputs_dir / "cluster_cluster_info.yaml"
    cluster_info.write_text("cluster: {}\n")

    resolved = _resolve_cluster_info_path({}, str(case_dir))

    assert resolved == str(cluster_info.resolve())


def test_load_cluster_info_auto_detected_cluster_info_parse_failure_raises(monkeypatch, tmp_path):
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
        "apex_cas.parse_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        load_cluster_info(
            case_dir=str(case_dir),
            config_raw={},
            mol=FakeMol(),
            charge=0,
            spin=0.0,
            symmetry_group="C1",
        )
    except RuntimeError as exc:
        assert "cluster_info_path" in str(exc)
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

    resolved = _resolve_fcidump_path(
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
    monkeypatch.setattr(
        "apex_filter.CAS_loader._load_apex_cas_provenance",
        lambda case_dir: {
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
                "settings": {
                    "cas_build": {
                        "pm_pop_method": "mulliken",
                        "pm_conv_tol": 1e-8,
                        "pm_max_cycle": 500,
                        "newton_max_cycle": 10,
                        "newton_conv_tol": 1e-12,
                    }
                }
            },
        },
    )

    settings, provenance = _extract_effective_settings({}, "/tmp/fake_case")

    assert settings.scf_spin == 4.5
    assert settings.conv_tol == 1e-10
    assert settings.max_cycle == 100
    assert settings.pm_pop_method == "mulliken"
    assert settings.pm_conv_tol == 1e-8
    assert settings.pm_max_cycle == 500
    assert settings.newton_max_cycle == 10
    assert settings.newton_conv_tol == 1e-12
    assert provenance["stem"] == "demo"
