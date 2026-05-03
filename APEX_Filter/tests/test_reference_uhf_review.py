"""Regression tests for Step 3 reference-UHF construction and initial guesses."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from shared.models import CAS, ClusterInfo, MetalCenter
from shared.settings_payloads import build_base_settings_payload
from apex_filter.session import SessionManager, _sanitize_label
from apex_filter.reference_uhf import (
    _compute_high_spin_ms2,
    _apply_d_orbital_encoding,
    _parse_orbital_metal_mapping,
    _sanitize_ms2_for_nelec,
    _swap_orbital_spin,
    _warn_if_spin_sites_unmapped,
    converge_reference_uhf,
)
from apex_filter.selection_guidance import _attach_display_labels
from apex_filter.steps_reference_uhf import step_uhf


def test_apply_d_orbital_encoding_preserves_minority_electron_count():
    dm_a = np.zeros((6, 6))
    dm_b = np.zeros((6, 6))

    # Metal site 0 occupies orbitals 0-4; minority beta occupation totals 1.
    dm_b[0, 0] = 0.2
    dm_b[1, 1] = 0.3
    dm_b[2, 2] = 0.1
    dm_b[3, 3] = 0.4

    metal_orbital_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: None}

    _apply_d_orbital_encoding(dm_a, dm_b, site_idx=0, d_orb_idx=2, spin_dir=+1,
                              metal_orbital_map=metal_orbital_map, cas=None)

    assert np.isclose(np.trace(dm_b[:5, :5]), 1.0)
    assert dm_b[2, 2] == 1.0
    assert np.allclose(np.diag(dm_b)[:5], [0.0, 0.0, 1.0, 0.0, 0.0])


def test_parse_orbital_metal_mapping_only_marks_metal_d_orbitals():
    cas = CAS(
        n_electrons=10,
        n_orbitals=5,
        orbital_labels=["Fe1_3dxy", "Fe1_4s", "S2_3pz", "Mo1_4dz2", "Fe2_3px"],
    )
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Mo", index=1, position=np.zeros(3), label="Mo1"),
            MetalCenter(element="Fe", index=2, position=np.zeros(3), label="Fe2"),
        ]
    )

    mapping = _parse_orbital_metal_mapping(cas, cluster)

    assert mapping[0] == 0
    assert mapping[1] is None
    assert mapping[2] is None
    assert mapping[3] == 1
    assert mapping[4] is None


def test_parse_orbital_metal_mapping_accepts_prefixed_labels():
    cas = CAS(
        n_electrons=6,
        n_orbitals=2,
        orbital_labels=["88: Fe1_3dx2-y2", "17: Mo1_4dz2"],
    )
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=10, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Mo", index=11, position=np.zeros(3), label="Mo1"),
        ]
    )

    mapping = _parse_orbital_metal_mapping(cas, cluster)

    assert mapping[0] == 0
    assert mapping[1] == 1


def test_swap_orbital_spin_uses_original_rows_and_columns():
    dm_a = np.array([[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]])
    dm_b = np.array([[4.0, 1.2, 1.3], [1.2, 5.0, 1.4], [1.3, 1.4, 6.0]])

    _swap_orbital_spin(dm_a, dm_b, 1)

    assert np.allclose(dm_a[1, :], [1.2, 5.0, 1.4])
    assert np.allclose(dm_a[:, 1], [1.2, 5.0, 1.4])
    assert np.allclose(dm_b[1, :], [0.2, 2.0, 0.4])
    assert np.allclose(dm_b[:, 1], [0.2, 2.0, 0.4])
    assert np.allclose(dm_a[[0, 2]][:, [0, 2]], [[1.0, 0.3], [0.3, 3.0]])
    assert np.allclose(dm_b[[0, 2]][:, [0, 2]], [[4.0, 1.3], [1.3, 6.0]])


def test_warn_if_spin_sites_unmapped_logs_missing_sites(caplog):
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
    )
    config = type(
        "Cfg",
        (),
        {"spin_assignment": {0: -1, 1: +1}, "label": "BS1_1-1"},
    )()

    with caplog.at_level("WARNING"):
        _warn_if_spin_sites_unmapped(config, {0: 0, 1: None}, cluster)

    assert "Fe2" in caplog.text


def test_sanitize_ms2_for_nelec_enforces_parity_and_bounds():
    assert _sanitize_ms2_for_nelec(10, 9) == 8
    assert _sanitize_ms2_for_nelec(9, 0) == 1
    assert _sanitize_ms2_for_nelec(6, 12) == 6


def test_compute_high_spin_ms2_uses_config_oxidation(monkeypatch):
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
    )
    oxidation = type("Ox", (), {"assignments": {0: 2, 1: 3}})()
    config = type("Cfg", (), {"oxidation": oxidation})()

    def fake_get_local_spin(element, ox):
        return {2: 2.0, 3: 2.5}[ox]

    monkeypatch.setattr("shared.chem_knowledge.get_local_spin", fake_get_local_spin)

    assert _compute_high_spin_ms2(cluster, config) == 9


def test_internal_attach_display_labels_prefers_final_state_signature_without_upstream():
    rows = [
        {
            "label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
        }
    ]

    _attach_display_labels(rows, None)

    assert rows[0]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"


def test_internal_attach_display_labels_replaces_stale_display_label_equal_to_label():
    rows = [
        {
            "label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
        }
    ]

    _attach_display_labels(rows, None)

    assert rows[0]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"


def test_converge_reference_uhf_uses_high_spin_then_target_spin(monkeypatch):
    created_mols = []
    solver_builds = []
    kernel_log = []

    class FakeMol:
        def __init__(self, spin):
            self.spin = spin
            self.nelectron = 23

    class FakeMF:
        def __init__(self, tag, converged=True):
            self.tag = tag
            self.converged = converged
            self.level_shift = 0.0
            self.damp = 0.0
            self.max_cycle = None
            self.e_tot = -1.23 if tag == "hs" else -2.34
            self.mo_coeff = (np.eye(2), np.eye(2))
            self.mo_energy = (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
            self.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
            self._dm = (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 1.0]]),
            )

        def kernel(self, dm0=None):
            kernel_log.append(
                {
                    "tag": self.tag,
                    "level_shift": self.level_shift,
                    "damp": self.damp,
                    "max_cycle": self.max_cycle,
                    "dm0_is_none": dm0 is None,
                }
            )
            return self.e_tot

        def make_rdm1(self):
            return self._dm

        def spin_square(self):
            return (0.75, 1.0)

    def fake_build_fake_mol(norb, nelec, ms2, ecore=0.0):
        created_mols.append({"norb": norb, "nelec": nelec, "ms2": ms2, "ecore": ecore})
        return FakeMol(ms2)

    def fake_build_solver(fcidump_data, mol_fake, conv_tol=1e-8, max_cycle=2000):
        tag = "hs" if not solver_builds else "bs"
        solver_builds.append(
            {"tag": tag, "spin": mol_fake.spin, "conv_tol": conv_tol, "max_cycle": max_cycle}
        )
        return FakeMF(tag=tag, converged=True)

    monkeypatch.setattr("apex_filter.reference_uhf._build_fake_mol", fake_build_fake_mol)
    monkeypatch.setattr("apex_filter.reference_uhf._build_reference_uhf_solver", fake_build_solver)
    monkeypatch.setattr(
        "apex_filter.reference_uhf._parse_orbital_metal_mapping",
        lambda cas, cluster: {0: 0, 1: 1},
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._warn_if_spin_sites_unmapped",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._compute_high_spin_ms2",
        lambda cluster, config: 9,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._swap_orbital_spin",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._apply_d_orbital_encoding",
        lambda *args, **kwargs: None,
    )

    cas = CAS(
        n_electrons=23,
        n_orbitals=2,
        orbital_labels=["Fe1_3dxy", "Fe2_3dxy"],
        occupations=np.array([1.0, 1.0]),
    )
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ],
        target_spin=0.5,
    )
    spin_isomer = type("Iso", (), {"Sz": 0.5})()
    oxidation = type("Ox", (), {"assignments": {0: 2, 1: 3}})()
    config = type(
        "Cfg",
        (),
        {
            "label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "spin_assignment": {0: -1, 1: +1},
            "spin_isomer": spin_isomer,
            "oxidation": oxidation,
            "d_orbital_assignments": {0: 0},
        },
    )()
    fcid = type("FCI", (), {"norb": 2, "nelec": 23, "ecore": 0.0})()

    result = converge_reference_uhf(cas, config, fcid, cluster, conv_tol=1e-10, max_cycle=500)

    assert result.converged is True
    assert [entry["ms2"] for entry in created_mols] == [9, 1]
    assert solver_builds[0]["conv_tol"] == 1e-10
    assert solver_builds[0]["max_cycle"] == 500
    assert solver_builds[1]["conv_tol"] == 1e-10
    assert solver_builds[1]["max_cycle"] == 20
    assert kernel_log[0] == {
        "tag": "hs",
        "level_shift": 0.0,
        "damp": 0.0,
        "max_cycle": None,
        "dm0_is_none": True,
    }
    assert kernel_log[1]["tag"] == "bs"
    assert kernel_log[1]["level_shift"] == 0.3
    assert kernel_log[1]["damp"] == 0.2
    assert kernel_log[1]["max_cycle"] is None
    assert kernel_log[2]["tag"] == "bs"
    assert kernel_log[2]["level_shift"] == 0.0
    assert kernel_log[2]["damp"] == 0.0
    assert kernel_log[2]["max_cycle"] == 500


def test_converge_reference_uhf_preserves_spin_square_when_not_converged(monkeypatch):
    class FakeMol:
        def __init__(self, spin):
            self.spin = spin
            self.nelectron = 23

    class FakeMF:
        def __init__(self, converged):
            self.converged = converged
            self.level_shift = 0.0
            self.damp = 0.0
            self.max_cycle = None
            self.e_tot = -2.34
            self.mo_coeff = (np.eye(2), np.eye(2))
            self.mo_energy = (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
            self.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
            self._dm = (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 1.0]]),
            )

        def kernel(self, dm0=None):
            return self.e_tot

        def make_rdm1(self):
            return self._dm

        def spin_square(self):
            return (4.67, 4.44)

    builds = []

    def fake_build_fake_mol(norb, nelec, ms2, ecore=0.0):
        return FakeMol(ms2)

    def fake_build_solver(fcidump_data, mol_fake, conv_tol=1e-8, max_cycle=2000):
        builds.append(mol_fake.spin)
        # high-spin converges; BS target remains unconverged
        return FakeMF(converged=(len(builds) == 1))

    monkeypatch.setattr("apex_filter.reference_uhf._build_fake_mol", fake_build_fake_mol)
    monkeypatch.setattr("apex_filter.reference_uhf._build_reference_uhf_solver", fake_build_solver)
    monkeypatch.setattr(
        "apex_filter.reference_uhf._parse_orbital_metal_mapping",
        lambda cas, cluster: {0: 0, 1: 1},
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._warn_if_spin_sites_unmapped",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._compute_high_spin_ms2",
        lambda cluster, config: 9,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._swap_orbital_spin",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._apply_d_orbital_encoding",
        lambda *args, **kwargs: None,
    )

    cas = CAS(
        n_electrons=23,
        n_orbitals=2,
        orbital_labels=["Fe1_3dxy", "Fe2_3dxy"],
        occupations=np.array([1.0, 1.0]),
    )
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ],
        target_spin=0.5,
    )
    spin_isomer = type("Iso", (), {"Sz": 0.5})()
    oxidation = type("Ox", (), {"assignments": {0: 2, 1: 3}})()
    config = type(
        "Cfg",
        (),
        {
            "label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "spin_assignment": {0: -1, 1: +1},
            "spin_isomer": spin_isomer,
            "oxidation": oxidation,
            "d_orbital_assignments": {0: 0},
        },
    )()
    fcid = type("FCI", (), {"norb": 2, "nelec": 23, "ecore": 0.0})()

    result = converge_reference_uhf(cas, config, fcid, cluster)

    assert result.converged is False
    assert result.energy == -2.34
    assert result.s_squared == 4.67


def test_converge_reference_uhf_can_apply_newton_refinement(monkeypatch):
    class FakeMol:
        def __init__(self, spin):
            self.spin = spin
            self.nelectron = 23

    class FakeMF:
        def __init__(self, converged, energy, spin_sq):
            self.converged = converged
            self.level_shift = 0.0
            self.damp = 0.0
            self.max_cycle = None
            self.e_tot = energy
            self.mo_coeff = (np.eye(2), np.eye(2))
            self.mo_energy = (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
            self.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
            self._dm = (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 1.0]]),
            )
            self.callback = None
            self.newton_called = False

        def kernel(self, *args, **kwargs):
            if callable(self.callback):
                self.callback(
                    {
                        "cycle": 0,
                        "e_tot": self.e_tot,
                        "last_hf_e": self.e_tot + 1e-4,
                        "norm_gorb": 1e-3,
                        "norm_ddm": 1e-2,
                    }
                )
            return self.e_tot

        def make_rdm1(self):
            return self._dm

        def spin_square(self):
            return (4.67, 4.44)

        def newton(self):
            self.newton_called = True
            return FakeMF(converged=True, energy=-2.5, spin_sq=4.5)

    builds = []

    def fake_build_fake_mol(norb, nelec, ms2, ecore=0.0):
        return FakeMol(ms2)

    def fake_build_solver(fcidump_data, mol_fake, conv_tol=1e-8, max_cycle=2000):
        builds.append(mol_fake.spin)
        return FakeMF(converged=(len(builds) == 1), energy=-2.34 if len(builds) > 1 else -1.23, spin_sq=4.67)

    monkeypatch.setattr("apex_filter.reference_uhf._build_fake_mol", fake_build_fake_mol)
    monkeypatch.setattr("apex_filter.reference_uhf._build_reference_uhf_solver", fake_build_solver)
    monkeypatch.setattr(
        "apex_filter.reference_uhf._parse_orbital_metal_mapping",
        lambda cas, cluster: {0: 0, 1: 1},
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._warn_if_spin_sites_unmapped",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._compute_high_spin_ms2",
        lambda cluster, config: 9,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._swap_orbital_spin",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apex_filter.reference_uhf._apply_d_orbital_encoding",
        lambda *args, **kwargs: None,
    )

    cas = CAS(
        n_electrons=23,
        n_orbitals=2,
        orbital_labels=["Fe1_3dxy", "Fe2_3dxy"],
        occupations=np.array([1.0, 1.0]),
    )
    cluster = ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ],
        target_spin=0.5,
    )
    spin_isomer = type("Iso", (), {"Sz": 0.5})()
    oxidation = type("Ox", (), {"assignments": {0: 2, 1: 3}})()
    config = type(
        "Cfg",
        (),
        {
            "label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1",
            "spin_assignment": {0: -1, 1: +1},
            "spin_isomer": spin_isomer,
            "oxidation": oxidation,
            "d_orbital_assignments": {0: 0},
        },
    )()
    fcid = type("FCI", (), {"norb": 2, "nelec": 23, "ecore": 0.0})()

    result = converge_reference_uhf(
        cas, config, fcid, cluster, newton_refine=True, newton_max_cycle=6
    )

    assert result.converged is True
    assert result.energy == -2.5
    assert result.diagnostics["newton_used"] is True
    assert len(result.diagnostics["newton_history"]) == 1


def test_step_uhf_rejects_energy_rank_pick_before_energies_exist(monkeypatch, tmp_path):
    class FakeSessionManager:
        def __init__(self, session_dir):
            self.session_dir = session_dir

        @property
        def method_controls_path(self):
            return str(Path(self.session_dir) / "method_controls.yaml")

        def require_previous(self, *args, **kwargs):
            return None

        def resolve_method_controls(self, section, defaults, cli_overrides):
            merged = dict(defaults)
            merged.update(cli_overrides)
            return merged

        def load_load_state(self):
            return {
                "cas": CAS(n_electrons=2, n_orbitals=2),
                "cluster_info": ClusterInfo(),
                "fcidump_data": object(),
            }

        def load_enumeration(self):
            return {"configs": []}

    monkeypatch.setattr(
        "apex_filter.steps_reference_uhf._SessionManager",
        FakeSessionManager,
    )

    try:
        step_uhf(str(tmp_path / "session"), pick="top 5")
    except ValueError as exc:
        assert "only pick modes 'all', 'labels', and 'file'" in str(exc)
    else:
        raise AssertionError("Expected step_uhf to reject energy-ranked pick modes")


def test_step_uhf_forwards_stabilization_controls(monkeypatch, tmp_path):
    captured = {}

    class FakeSessionManager:
        def __init__(self, session_dir):
            self.session_dir = session_dir

        @property
        def method_controls_path(self):
            return str(Path(self.session_dir) / "method_controls.yaml")

        def require_previous(self, *args, **kwargs):
            return None

        def resolve_method_controls(self, section, defaults, cli_overrides):
            merged = dict(defaults)
            merged.update(cli_overrides)
            return merged

        def load_load_state(self):
            return {
                "cas": CAS(n_electrons=2, n_orbitals=2),
                "cluster_info": ClusterInfo(),
                "fcidump_data": object(),
            }

        def load_enumeration(self):
            spin_isomer = type("Iso", (), {"family": "BS1"})()
            cfg = type("Cfg", (), {"label": "L1", "spin_isomer": spin_isomer})()
            return {"configs": [cfg]}

        def _build_step_settings_payload(self, source_settings, *, theory: str, **overrides):
            return build_base_settings_payload(
                source_settings,
                control_source=self.method_controls_path,
                theory=theory,
                **overrides,
            )

        def save_step_picked(self, step_name, labels):
            assert step_name == "step3_uhf"
            captured["picked"] = labels

        def save_uhf_result(self, label, result, **kwargs):
            captured["result_label"] = label

        def save_step_summary(self, step_name, filename, results, mark_completed=True):
            assert step_name == "step3_uhf"
            assert filename == "uhf_summary.json"
            captured["summary"] = results

        def _rebuild_uhf_summary(self, configs, current_results=None):
            return current_results or []

    def fake_converge(cas, cfg, fcid, ci, **kwargs):
        captured["kwargs"] = kwargs
        return type(
            "Res",
            (),
            {
                "energy": -1.0,
                "converged": True,
                "s_squared": 0.75,
                "diagnostics": {
                    "final_delta_e": -1e-6,
                    "energy_tail": [-1.1, -1.0],
                    "final_d_basin": {"Fe1": "dxy"},
                    "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
                    "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dxy",
                },
            },
        )()

    monkeypatch.setattr(
        "apex_filter.steps_reference_uhf._SessionManager",
        FakeSessionManager,
    )
    monkeypatch.setattr(
        "apex_filter.steps_reference_uhf._converge_reference_uhf",
        fake_converge,
    )
    monkeypatch.setattr(
        "apex_filter.steps_reference_uhf._write_selection_artifacts",
        lambda *args, **kwargs: captured.setdefault("selection_kwargs", kwargs),
    )

    step_uhf(
        str(tmp_path / "session"),
        pick="all",
        conv_tol=1e-10,
        max_cycle=500,
        stabilize_cycles=80,
        level_shift=0.5,
        damp=0.3,
        newton_refine=True,
        newton_max_cycle=6,
    )

    assert captured["picked"] == ["L1"]
    assert captured["result_label"] == "L1"
    assert captured["kwargs"] == {
        "conv_tol": 1e-10,
        "max_cycle": 500,
        "stabilize_cycles": 80,
        "level_shift": 0.5,
        "damp": 0.3,
        "newton_refine": True,
        "newton_max_cycle": 6,
    }
    assert captured["summary"] == [
        {
            "label": "L1",
            "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dxy",
            "energy": -1.0,
            "converged": True,
            "s_squared": 0.75,
            "family": "BS1",
            "last_delta_e": -1e-6,
            "energy_tail": [-1.1, -1.0],
            "two_s": None,
            "two_sz_fe1": None,
            "two_sz_fe2": None,
            "final_d_basin": {"Fe1": "dxy"},
            "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
            "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dxy",
        }
    ]
    assert captured["selection_kwargs"]["keep_default"] == "1"


def test_step_uhf_writes_post_scf_observables_when_inputs_available(monkeypatch, tmp_path):
    captured = {}
    session_dir = tmp_path / "session"
    results_dir = session_dir / "step3_uhf" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    class FakeSessionManager:
        def __init__(self, session_dir):
            self.session_dir = session_dir

        @property
        def method_controls_path(self):
            return str(Path(self.session_dir) / "method_controls.yaml")

        def require_previous(self, *args, **kwargs):
            return None

        def resolve_method_controls(self, section, defaults, cli_overrides):
            merged = dict(defaults)
            merged.update(cli_overrides)
            return merged

        def load_load_state(self):
            return {
                "cas": CAS(n_electrons=2, n_orbitals=2),
                "cluster_info": ClusterInfo(),
                "fcidump_data": object(),
                "fcidump_path": str(tmp_path / "case" / "outputs" / "fcidump" / "FCIDUMP.test"),
                "config_path": str(tmp_path / "filter_settings.yaml"),
            }

        def load_enumeration(self):
            spin_isomer = type("Iso", (), {"family": "BS1"})()
            cfg = type("Cfg", (), {"label": "L1", "spin_isomer": spin_isomer})()
            return {"configs": [cfg]}

        def _build_step_settings_payload(self, source_settings, *, theory: str, **overrides):
            return build_base_settings_payload(
                source_settings,
                control_source=self.method_controls_path,
                theory=theory,
                **overrides,
            )

        def save_step_picked(self, step_name, labels):
            assert step_name == "step3_uhf"
            captured["picked"] = labels

        def save_uhf_result(self, label, result, **kwargs):
            captured["result_label"] = label

        def save_step_summary(self, step_name, filename, results, mark_completed=True):
            assert step_name == "step3_uhf"
            assert filename == "uhf_summary.json"
            captured["summary"] = results

        def _rebuild_uhf_summary(self, configs, current_results=None):
            return current_results or []

    def fake_converge(cas, cfg, fcid, ci, **kwargs):
        return type(
            "Res",
            (),
            {
                "energy": -1.0,
                "converged": True,
                "s_squared": 0.75,
                "diagnostics": {
                    "final_delta_e": -1e-6,
                    "energy_tail": [-1.1, -1.0],
                    "final_d_basin": {"Fe1": "dxy"},
                    "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
                    "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dxy",
                },
            },
        )()

    def fake_observables(**kwargs):
        captured["observable_kwargs"] = kwargs
        return {
            "two_s": 2.5,
            "two_sz_by_metal_label": {"Fe1": -4.2, "Fe2": 4.2},
        }

    monkeypatch.setattr("apex_filter.steps_reference_uhf._SessionManager", FakeSessionManager)
    monkeypatch.setattr("apex_filter.steps_reference_uhf._converge_reference_uhf", fake_converge)
    monkeypatch.setattr("apex_filter.steps_reference_uhf._analyze_step3_uhf_observables", fake_observables)
    monkeypatch.setattr("apex_filter.steps_reference_uhf._write_selection_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "apex_filter.steps_reference_uhf._build_case_observable_inputs",
        lambda state, cfg: {
            "xyz_path": str(tmp_path / "fe2s2.xyz"),
            "cluster_info_path": str(tmp_path / "cluster.yaml"),
            "cas_settings_path": str(tmp_path / "cas.yaml"),
            "cas_data_h5_path": str(tmp_path / "cas_data.h5"),
            "label": cfg.label,
            "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
        },
    )

    step_uhf(str(session_dir), pick="all")

    assert captured["summary"][0]["two_s"] == pytest.approx(2.5)
    assert captured["summary"][0]["two_sz_fe1"] == pytest.approx(-4.2)
    assert captured["summary"][0]["two_sz_fe2"] == pytest.approx(4.2)
    assert captured["observable_kwargs"]["step3_h5_path"].endswith("L1_uhf.h5")
    assert "chan_benchmark_json" not in captured["observable_kwargs"]
    assert (results_dir / "L1_post_scf_observables.json").exists()


def test_session_internal_rebuild_uhf_summary_keeps_all_saved_npzs(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    iso1 = type("Iso", (), {"family": "BS1_1"})()
    iso2 = type("Iso", (), {"family": "BS1_2"})()
    cfg1 = type("Cfg", (), {"label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1", "spin_isomer": iso1})()
    cfg2 = type("Cfg", (), {"label": "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:d1", "spin_isomer": iso2})()

    class Res:
        def __init__(self, energy, converged, s2, sig):
            self.energy = energy
            self.converged = converged
            self.s_squared = s2
            self.mo_coeff = (None, None)
            self.mo_occ = (None, None)
            self.mo_energy = (None, None)
            self.dm = (None, None)
            self.diagnostics = {
                "final_delta_e": -1e-9,
                "final_state_signature": sig,
                "final_d_basin": {"Fe1": "dz^2"} if "Fe1" in sig else {"Fe2": "dz^2"},
                "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
            }

    sm.save_uhf_result(cfg1.label, Res(-1.0, True, 0.75, "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"))
    sm.save_uhf_result(cfg2.label, Res(-2.0, True, 0.80, "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:dz^2"))

    rebuilt = sm._rebuild_uhf_summary([cfg1, cfg2], current_results=[
        {
            "label": cfg1.label,
            "display_label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
            "energy": -1.0,
            "converged": True,
            "s_squared": 0.75,
            "family": "BS1_1",
            "last_delta_e": -1e-9,
            "energy_tail": [],
            "final_d_basin": {"Fe1": "dz^2"},
            "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
            "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
        }
    ])

    assert [row["label"] for row in rebuilt] == [cfg2.label, cfg1.label]
    assert rebuilt[0]["display_label"] == "Fe1↑Fe2↓|Fe1(III)+Fe2(II)|Fe2:dz^2"
    assert rebuilt[1]["display_label"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"


def test_session_internal_rebuild_uhf_summary_can_read_h5_only(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    iso = type("Iso", (), {"family": "BS1"})()
    cfg = type("Cfg", (), {"label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1", "spin_isomer": iso})()

    class Res:
        def __init__(self):
            self.energy = -1.0
            self.converged = True
            self.s_squared = 0.75
            self.mo_coeff = (np.eye(2), np.eye(2))
            self.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
            self.mo_energy = (np.array([-1.0, 0.5]), np.array([-1.0, 0.5]))
            self.dm = (np.eye(2), np.eye(2))
            self.diagnostics = {
                "final_delta_e": -1e-9,
                "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
                "final_d_basin": {"Fe1": "dz^2"},
                "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
            }

    sm.save_uhf_result(cfg.label, Res())
    results_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    os.remove(os.path.join(results_dir, f"{_sanitize_label(cfg.label)}_uhf.npz"))

    rebuilt = sm._rebuild_uhf_summary([cfg], current_results=[])
    assert len(rebuilt) == 1
    assert rebuilt[0]["label"] == cfg.label
    assert rebuilt[0]["final_state_signature"] == "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2"


def test_session_internal_rebuild_uhf_summary_reads_post_scf_sidecar(tmp_path):
    sm = SessionManager(str(tmp_path / "session"))
    sm.create()

    iso = type("Iso", (), {"family": "BS1"})()
    cfg = type("Cfg", (), {"label": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:d1", "spin_isomer": iso})()

    class Res:
        def __init__(self):
            self.energy = -1.0
            self.converged = True
            self.s_squared = 0.75
            self.mo_coeff = (np.eye(2), np.eye(2))
            self.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
            self.mo_energy = (np.array([-1.0, 0.5]), np.array([-1.0, 0.5]))
            self.dm = (np.eye(2), np.eye(2))
            self.diagnostics = {
                "final_delta_e": -1e-9,
                "final_state_signature": "Fe1↓Fe2↑|Fe1(II)+Fe2(III)|Fe1:dz^2",
                "final_d_basin": {"Fe1": "dz^2"},
                "final_site_spin_proxy": {"Fe1": -3.0, "Fe2": 4.0},
            }

    sm.save_uhf_result(cfg.label, Res())
    results_dir = os.path.join(sm.session_dir, "step3_uhf", "results")
    safe = _sanitize_label(cfg.label)
    with open(os.path.join(results_dir, f"{safe}_post_scf_observables.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "two_s": 2.5,
                "two_sz_by_metal_label": {"Fe1": -4.2, "Fe2": 4.2},
            },
            f,
        )

    rebuilt = sm._rebuild_uhf_summary([cfg], current_results=[])
    assert rebuilt[0]["two_s"] == pytest.approx(2.5)
    assert rebuilt[0]["two_sz_fe1"] == pytest.approx(-4.2)
    assert rebuilt[0]["two_sz_fe2"] == pytest.approx(4.2)
