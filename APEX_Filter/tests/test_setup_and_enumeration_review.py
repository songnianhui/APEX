"""Regression tests for Step 1/Step 2 setup and enumeration contracts."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.setting_utils import load_cas_settings_file
from shared.structure_parser import parse_structure
from apex_filter.elec_spin_config_generator import (
    _summarize_enumeration_layers,
    canonicalize_config_spin_labels,
    generate_all_configs,
)
from shared.models import (
    CAS,
    ClusterInfo,
    ComputationSettings,
    ElectronicConfig,
    MetalCenter,
    OxidationAssignment,
    SpinIsomer,
)
from apex_filter.session import SessionManager
from apex_filter.selection_guidance import _write_selection_artifacts
from apex_filter.steps_setup import _validate_active_space_inputs
from apex_filter.steps_enumeration import step_enumerate


def test_validate_active_space_inputs_rejects_mismatched_dimensions():
    cas = CAS(n_electrons=10, n_orbitals=8)
    fcid = SimpleNamespace(norb=7, nelec=10)

    try:
        _validate_active_space_inputs(cas, fcid)
    except ValueError as exc:
        assert "orbital mismatch" in str(exc)
    else:
        raise AssertionError("Expected CAS/FCIDUMP mismatch to raise ValueError")


def test_step1_state_persistence_normalizes_config_path(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    cluster = ClusterInfo(metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")])
    cas = CAS(n_electrons=2, n_orbitals=2)
    settings = ComputationSettings()
    rel_config = "relative/filter.yaml"

    provenance = {"stem": "demo", "scf_info": {"settings": {"scf": {"conv_tol": 1e-10}}}}
    sm.save_load_state(
        cluster,
        cas,
        str(tmp_path / "FCIDUMP.demo"),
        settings,
        rel_config,
        apex_cas_provenance=provenance,
    )

    assert sm.load()["config_path"].endswith("relative/filter.yaml")
    assert sm.load()["config_path"].startswith("/")
    step1 = tmp_path / "session" / "step1_load"
    with open(step1 / "fcidump_ref.json") as fh:
        fcidump_ref = fh.read()
    with open(step1 / "cas_meta.json") as fh:
        cas_meta = fh.read()
    with open(step1 / "settings.json") as fh:
        settings_payload = fh.read()
    assert "FCIDUMP.demo" in fcidump_ref
    assert "\"stage\"" not in cas_meta
    assert "\"level\"" not in cas_meta
    assert "bootstrap_settings_snapshot" in settings_payload
    assert "apex_cas_provenance" in settings_payload


def test_step2_enumeration_roundtrip_preserves_integer_site_keys(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    spin_isomer = SpinIsomer(
        label="BS1_1-1",
        spin_assignment={0: -1, 1: +1},
        n_minority=1,
        family="BS1_1",
        Sz=1.5,
    )
    oxidation = OxidationAssignment(assignments={0: 3, 1: 2}, description="Fe(III)/Fe(II)")
    config = ElectronicConfig(
        spin_isomer=spin_isomer,
        oxidation=oxidation,
        d_orbital_assignments={0: 2},
        minority_spin_sites=[0],
        spin_assignment={0: -1, 1: +1},
        config_id=0,
        label="BS1_1-1|Fe(III)/Fe(II)|d0:2",
    )

    sm.save_enumeration([config], [spin_isomer], [], 1, {"raw_spin_patterns": 1})
    loaded = sm.load_enumeration()
    loaded_cfg = loaded["configs"][0]

    assert loaded_cfg.spin_assignment == {0: -1, 1: +1}
    assert loaded_cfg.spin_isomer.spin_assignment == {0: -1, 1: +1}
    assert loaded_cfg.oxidation.assignments == {0: 3, 1: 2}
    assert loaded_cfg.d_orbital_assignments == {0: 2}
    assert loaded["stats"]["raw_spin_patterns"] == 1


def test_step2_worklist_can_be_prefilled_with_keep_flags(tmp_path):
    step_dir = tmp_path / "step2_enumerate"
    _write_selection_artifacts(
        str(step_dir),
        step_name="Step 2 enumerate",
        next_step_name="uhf",
        summary=[
            {"label": "L1", "family": "BS1", "energy": None, "converged": None},
            {"label": "L2", "family": "BS2", "energy": None, "converged": None},
        ],
        keep_default="1",
    )

    worklist = [
        line for line in (step_dir / "selection_worklist.csv").read_text().strip().splitlines()
        if line and not line.startswith("#")
    ]
    assert worklist[1].startswith("1,")
    assert worklist[2].startswith("1,")


def test_step_enumerate_uses_method_controls(tmp_path, monkeypatch):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()
    sm.mark_step_completed("step1_load")

    method_controls = tmp_path / "session" / "method_controls.yaml"
    method_controls.write_text(
        "enumerate:\n"
        "  target_sz: 1.5\n"
        "  forced_oxidation:\n"
        "    0: 3\n"
        "  max_configs: 7\n",
        encoding="utf-8",
    )

    cluster = ClusterInfo(
        metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
        target_spin=0.0,
    )
    monkeypatch.setattr(SessionManager, "load_load_state", lambda self: {"cluster_info": cluster})

    captured = {}

    spin_isomer = SpinIsomer(
        label="BS1_1-1",
        spin_assignment={0: -1},
        n_minority=1,
        family="BS1_1",
        Sz=1.5,
    )
    oxidation = OxidationAssignment(assignments={0: 3}, description="Fe(III)")
    config = ElectronicConfig(
        spin_isomer=spin_isomer,
        oxidation=oxidation,
        d_orbital_assignments={},
        minority_spin_sites=[0],
        spin_assignment={0: -1},
        config_id=0,
        label="BS1_1-1|Fe(III)|d:none",
    )

    def fake_generate_all_configs(ci, target_Sz=None, max_configs=None, forced_oxidation=None):
        captured["target_Sz"] = target_Sz
        captured["max_configs"] = max_configs
        captured["forced_oxidation"] = forced_oxidation
        return [config]

    monkeypatch.setattr(
        "apex_filter.steps_enumeration._generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr(
        "apex_filter.steps_enumeration._canonicalize_config_spin_labels",
        lambda configs, ci: (configs, [spin_isomer], []),
    )
    monkeypatch.setattr(
        "apex_filter.steps_enumeration._reduce_configs_by_symmetry",
        lambda configs, ci: configs,
    )
    monkeypatch.setattr(
        "apex_filter.steps_enumeration._summarize_enumeration_layers",
        lambda raw, reduced, spin_isomers, families: {
            "raw_spin_patterns": 1,
            "spin_families": 1,
            "spin_x_oxidation": 1,
            "spin_x_oxidation_x_d_before_reduction": 1,
            "total_configs_after_reduction": 1,
        },
    )

    step_enumerate(sm.session_dir)

    assert captured["target_Sz"] == 1.5
    assert captured["max_configs"] == 7
    assert captured["forced_oxidation"] == {0: 3}


def test_session_internal_rebuild_ccsd_summary_keeps_all_saved_npzs(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    iso1 = type("Iso", (), {"family": "BS1_1"})()
    iso2 = type("Iso", (), {"family": "BS1_2"})()
    cfg1 = type("Cfg", (), {"label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1", "spin_isomer": iso1})()
    cfg2 = type("Cfg", (), {"label": "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:d1", "spin_isomer": iso2})()

    np.savez(
        sm.step_artifact_dir("step4_ccsd", "scripts") + "/Fe1↓Fe2↑_Fe1(II)+Fe2(III)_Fe1:d1_ccsd_results.npz",
        ccsd_total=-1.0,
        ccsd_corr=-0.1,
        ccsd_converged=True,
        spin_sq=4.9,
        two_s=3.53,
        two_sz_fe1=-4.2,
        two_sz_fe2=4.2,
    )
    np.savez(
        sm.step_artifact_dir("step4_ccsd", "scripts") + "/Fe1↑Fe2↓_Fe1(III)+Fe2(II)_Fe2:d1_ccsd_results.npz",
        ccsd_total=-2.0,
        ccsd_corr=-0.2,
        ccsd_converged=True,
        spin_sq=4.8,
        two_s=3.50,
        two_sz_fe1=4.1,
        two_sz_fe2=-4.1,
    )

    rebuilt = sm._rebuild_ccsd_summary(
        [cfg1, cfg2],
        upstream_summary=[
            {"label": cfg1.label, "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"},
            {"label": cfg2.label, "display_label": "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:dz^2"},
        ],
        current_results=[
            {
                "label": cfg1.label,
                "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
                "method": "UCCSD",
                "energy": -1.0,
                "correlation_energy": -0.1,
                "converged": True,
                "family": "BS1_1",
            }
        ],
    )

    assert [row["label"] for row in rebuilt] == [cfg2.label, cfg1.label]
    assert rebuilt[0]["display_label"] == "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:dz^2"
    assert rebuilt[1]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"
    assert rebuilt[0]["s_squared"] == pytest.approx(4.8)
    assert rebuilt[0]["two_s"] == pytest.approx(3.50)
    assert rebuilt[0]["two_sz_fe1"] == pytest.approx(4.1)
    assert rebuilt[0]["two_sz_fe2"] == pytest.approx(-4.1)


def test_generate_all_configs_respects_target_sz_override(monkeypatch):
    cluster = ClusterInfo(
        metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
        target_spin=1.5,
    )
    oxidation = OxidationAssignment(assignments={0: 3}, description="Fe(III)")
    seen = []

    def fake_enum_ox(*args, **kwargs):
        return [oxidation]

    def fake_enum_isomers(cluster_info, target_Sz=None, oxidation_states=None):
        seen.append(target_Sz)
        return [
            SpinIsomer(
                label="BS1-1",
                spin_assignment={0: -1},
                n_minority=1,
                family="BS1",
                Sz=target_Sz,
            )
        ]

    monkeypatch.setattr(
        "apex_filter.elec_spin_config_generator._enumerate_oxidation_assignments",
        fake_enum_ox,
    )
    monkeypatch.setattr(
        "apex_filter.elec_spin_config_generator._enumerate_spin_isomers",
        fake_enum_isomers,
    )
    monkeypatch.setattr(
        "apex_filter.elec_spin_config_generator._get_d_orbital_choices_for_cluster",
        lambda *args, **kwargs: {},
    )

    configs = generate_all_configs(cluster, target_Sz=2.5)

    assert seen == [2.5]
    assert configs[0].spin_isomer.Sz == 2.5


def test_canonicalize_config_spin_labels_aligns_configs_and_families():
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.array([0.0, 0.0, 0.0]), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.array([1.0, 0.0, 0.0]), label="Fe2"),
            MetalCenter(element="Fe", index=2, position=np.array([2.0, 0.0, 0.0]), label="Fe3"),
        ],
        symmetry_group="C1",
    )
    ox = OxidationAssignment(assignments={0: 3, 1: 3, 2: 3}, description="3xFe(III)")
    configs = [
        ElectronicConfig(
            spin_isomer=SpinIsomer(
                label="BS1-1",
                spin_assignment={0: -1, 1: +1, 2: +1},
                n_minority=1,
                family="BS1",
                Sz=1.5,
            ),
            oxidation=ox,
            spin_assignment={0: -1, 1: +1, 2: +1},
            minority_spin_sites=[0],
            label="raw-1",
        ),
        ElectronicConfig(
            spin_isomer=SpinIsomer(
                label="BS1-2",
                spin_assignment={0: +1, 1: -1, 2: +1},
                n_minority=1,
                family="BS1",
                Sz=1.5,
            ),
            oxidation=ox,
            spin_assignment={0: +1, 1: -1, 2: +1},
            minority_spin_sites=[1],
            label="raw-2",
        ),
    ]

    relabeled_configs, spin_isomers, families = canonicalize_config_spin_labels(configs, cluster)

    assert len(spin_isomers) == 2
    assert len(families) == 2
    assert relabeled_configs[0].spin_isomer.family == "BS1_1"
    assert relabeled_configs[1].spin_isomer.family == "BS1_2"
    assert relabeled_configs[0].spin_assignment == {0: -1, 1: +1, 2: +1}
    assert relabeled_configs[1].minority_spin_sites == [1]


def test_chan_benchmark_profiles_match_reference_counts():
    root = Path(__file__).resolve().parents[2]
    cases = [
        (
            "fe2s2",
            root / "examples" / "fe2s2" / "inputs" / "fe2s2.xyz",
            root / "examples" / "fe2s2" / "inputs" / "fe2s2_cas_settings.yaml",
            (2, 2, 2),
        ),
        (
            "fe4s4",
            root / "examples" / "fe4s4" / "inputs" / "fe4s4.xyz",
            root / "examples" / "fe4s4" / "inputs" / "fe4s4_cas_settings.yaml",
            (3, 3, 24),
        ),
        (
            "femoco",
            root / "examples" / "femoco" / "inputs" / "femoco.xyz",
            root / "examples" / "femoco" / "inputs" / "femoco_cas_settings.yaml",
            (35, 10, 78750),
        ),
    ]

    for _, xyz, yaml, expected in cases:
        settings = load_cas_settings_file(str(yaml))
        cluster = parse_structure(
            str(xyz),
            charge=settings.get("charge", 0),
            target_spin=settings.get("spin", 0.0),
            family_scheme=settings.get("family_scheme", ""),
            benchmark_profile=settings.get("benchmark_profile", ""),
            config_reduction_mode=settings.get("config_reduction_mode", "none"),
        )
        configs = generate_all_configs(cluster, target_Sz=settings.get("spin"))
        configs, spin_isomers, families = canonicalize_config_spin_labels(configs, cluster)
        assert (len(spin_isomers), len(families), len(configs)) == expected


def test_enumeration_layer_summary_uses_uniform_vocabulary_for_benchmarks():
    root = Path(__file__).resolve().parents[2]
    cases = [
        (
            root / "examples" / "fe2s2" / "inputs" / "fe2s2.xyz",
            root / "examples" / "fe2s2" / "inputs" / "fe2s2_cas_settings.yaml",
            {
                "raw_spin_patterns": 2,
                "spin_families": 2,
                "spin_x_oxidation": 2,
                "spin_x_oxidation_x_d_before_reduction": 2,
                "total_configs_after_reduction": 2,
            },
        ),
        (
            root / "examples" / "fe4s4" / "inputs" / "fe4s4.xyz",
            root / "examples" / "fe4s4" / "inputs" / "fe4s4_cas_settings.yaml",
            {
                "raw_spin_patterns": 6,
                "spin_families": 3,
                "spin_x_oxidation": 24,
                "spin_x_oxidation_x_d_before_reduction": 24,
                "total_configs_after_reduction": 24,
            },
        ),
        (
            root / "examples" / "femoco" / "inputs" / "femoco.xyz",
            root / "examples" / "femoco" / "inputs" / "femoco_cas_settings.yaml",
            {
                "raw_spin_patterns": 35,
                "spin_families": 10,
                "spin_x_oxidation": 630,
                "spin_x_oxidation_x_d_before_reduction": 78750,
                "total_configs_after_reduction": 78750,
            },
        ),
        (
            root / "examples" / "fe4s4h4" / "inputs" / "fe4s4h4.xyz",
            root / "examples" / "fe4s4h4" / "inputs" / "fe4s4h4_cas_settings.yaml",
            {
                "raw_spin_patterns": 6,
                "spin_families": 3,
                "spin_x_oxidation": 24,
                "spin_x_oxidation_x_d_before_reduction": 600,
                "total_configs_after_reduction": 600,
            },
        ),
    ]

    for xyz, yaml, expected in cases:
        settings = load_cas_settings_file(str(yaml))
        cluster = parse_structure(
            str(xyz),
            charge=settings.get("charge", 0),
            target_spin=settings.get("spin", 0.0),
            family_scheme=settings.get("family_scheme", ""),
            benchmark_profile=settings.get("benchmark_profile", ""),
            config_reduction_mode=settings.get("config_reduction_mode", "none"),
        )
        configs = generate_all_configs(cluster, target_Sz=settings.get("spin"))
        raw_configs = copy.deepcopy(configs)
        configs, spin_isomers, families = canonicalize_config_spin_labels(configs, cluster)
        summary = _summarize_enumeration_layers(raw_configs, configs, spin_isomers, families)
        for key, value in expected.items():
            assert summary[key] == value
        if xyz.name == "fe4s4.xyz":
            assert cluster.family_scheme == "literature_fe4s4_cubane"
        if xyz.name == "fe4s4h4.xyz":
            assert cluster.family_scheme == "literature_fe4s4_cubane"


def test_fe4s4h4_shares_cubane_family_scheme_with_fe4s4_benchmark():
    root = Path(__file__).resolve().parents[2]
    settings = load_cas_settings_file(
        str(root / "examples" / "fe4s4h4" / "inputs" / "fe4s4h4_cas_settings.yaml")
    )
    cluster = parse_structure(
        str(root / "examples" / "fe4s4h4" / "inputs" / "fe4s4h4.xyz"),
        charge=settings.get("charge", 0),
        target_spin=settings.get("spin", 0.0),
        family_scheme=settings.get("family_scheme", ""),
        benchmark_profile=settings.get("benchmark_profile", ""),
        config_reduction_mode=settings.get("config_reduction_mode", "none"),
    )
    configs = generate_all_configs(cluster, target_Sz=settings.get("spin"))
    _, spin_isomers, families = canonicalize_config_spin_labels(configs, cluster)

    assert cluster.family_scheme == "literature_fe4s4_cubane"
    assert [si.label for si in spin_isomers] == ["BS1", "BS2", "BS3"]
    assert [fam.label for fam in families] == ["BS1", "BS2", "BS3"]
