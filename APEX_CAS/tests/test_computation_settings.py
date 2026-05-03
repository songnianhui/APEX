"""Regression tests for shared settings and preset utilities.

Comprehensive coverage for presets, basis file loading, overrides,
and basis-dict construction.
"""

import os
import tempfile
import unittest

import yaml

from apex_cas import ComputationSettings
from shared.setting_utils import (
    DEFAULT_PRESET,
    FAST_PRESET,
    PRESETS,
    apply_overrides,
    build_basis_dict,
    load_basis_file,
    settings_from_preset,
)
from apex_cas import ClusterInfo, MetalCenter, BridgingAtom


class TestPresets(unittest.TestCase):
    def test_default_preset_baseline(self):
        s = PRESETS["default"]
        self.assertEqual(s.scf_method, "uks")
        self.assertEqual(s.xc_functional, "B3LYP")
        self.assertEqual(s.relativistic, "sf-x2c")
        self.assertEqual(s.solvation_model, "ddcosmo")
        self.assertEqual(s.solvation_epsilon, 4.0)
        self.assertEqual(s.conv_tol, 1e-8)
        self.assertEqual(s.max_cycle, 200)

    def test_fast_preset(self):
        s = PRESETS["fast"]
        self.assertEqual(s.scf_method, "uks")
        self.assertEqual(s.basis_set_default, "def2-SVP")
        self.assertEqual(s.relativistic, "none")
        self.assertEqual(s.solvation_model, "none")
        self.assertEqual(s.conv_tol, 1e-6)
        self.assertEqual(s.max_cycle, 100)

    def test_default_preset_mixed_basis(self):
        s = DEFAULT_PRESET
        self.assertEqual(s.get_basis("Fe"), "def2-TZVP")
        self.assertEqual(s.get_basis("Mo"), "def2-TZVP")
        self.assertEqual(s.get_basis("S"), "def2-TZVP")
        self.assertEqual(s.get_basis("C"), "def2-SVP")
        self.assertEqual(s.get_basis("H"), "def2-SVP")
        self.assertEqual(s.get_basis("O"), "def2-SVP")
        self.assertEqual(s.get_basis("N"), "def2-SVP")

    def test_unknown_element_uses_default(self):
        s = DEFAULT_PRESET
        self.assertEqual(s.get_basis("Xe"), "def2-TZVP")  # falls back to default

    def test_two_presets_available(self):
        self.assertIn("default", PRESETS)
        self.assertIn("fast", PRESETS)
        self.assertEqual(len(PRESETS), 2)


class TestLoadBasisFile(unittest.TestCase):
    def test_load_valid_file(self):
        data = {"Fe": "def2-QZVP", "S": "def2-TZVP", "H": "def2-SVP"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            result = load_basis_file(f.name)
        os.unlink(f.name)
        self.assertEqual(result, data)

    def test_invalid_element_key(self):
        data = {"invalidkey": "def2-SVP"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_basis_file(f.name)
        os.unlink(f.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(None, f)
            f.flush()
            result = load_basis_file(f.name)
        os.unlink(f.name)
        self.assertEqual(result, {})

    def test_non_string_value_raises(self):
        data = {"Fe": 123}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_basis_file(f.name)
        os.unlink(f.name)

    def test_lowercase_element_raises(self):
        data = {"fe": "def2-SVP"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_basis_file(f.name)
        os.unlink(f.name)

    def test_two_char_element_valid(self):
        data = {"Fe": "def2-SVP"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            result = load_basis_file(f.name)
        os.unlink(f.name)
        self.assertEqual(result["Fe"], "def2-SVP")


class TestApplyOverrides(unittest.TestCase):
    def test_simple_override(self):
        s = apply_overrides(DEFAULT_PRESET, scf_method="uhf")
        self.assertEqual(s.scf_method, "uhf")
        # Original should be unchanged
        self.assertEqual(DEFAULT_PRESET.scf_method, "uks")

    def test_basis_merge(self):
        s = apply_overrides(DEFAULT_PRESET,
                            basis_set_per_element={"Fe": "def2-QZVP"})
        self.assertEqual(s.get_basis("Fe"), "def2-QZVP")
        # Other elements should still have Chan 2019 values
        self.assertEqual(s.get_basis("S"), "def2-TZVP")
        self.assertEqual(s.get_basis("C"), "def2-SVP")

    def test_basis_file_merge(self):
        data = {"Fe": "def2-QZVP", "Mo": "def2-QZVP"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            s = apply_overrides(DEFAULT_PRESET, basis_set_file=f.name)
        os.unlink(f.name)
        self.assertEqual(s.get_basis("Fe"), "def2-QZVP")
        self.assertEqual(s.get_basis("Mo"), "def2-QZVP")
        self.assertEqual(s.get_basis("S"), "def2-TZVP")

    def test_multiple_overrides(self):
        s = apply_overrides(DEFAULT_PRESET,
                            scf_method="uhf",
                            conv_tol=1e-6,
                            max_cycle=50)
        self.assertEqual(s.scf_method, "uhf")
        self.assertEqual(s.conv_tol, 1e-6)
        self.assertEqual(s.max_cycle, 50)
        self.assertEqual(s.xc_functional, "B3LYP")  # unchanged

    def test_override_does_not_mutate_original(self):
        original_tol = DEFAULT_PRESET.conv_tol
        updated = apply_overrides(DEFAULT_PRESET, conv_tol=1e-4)
        self.assertEqual(updated.conv_tol, 1e-4)
        self.assertEqual(DEFAULT_PRESET.conv_tol, original_tol)


class TestSettingsFromPreset(unittest.TestCase):
    def test_default_preset(self):
        s = settings_from_preset("default")
        self.assertEqual(s.xc_functional, "B3LYP")

    def test_fast_preset_with_override(self):
        s = settings_from_preset("fast", scf_method="uhf")
        self.assertEqual(s.scf_method, "uhf")
        self.assertEqual(s.basis_set_default, "def2-SVP")

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            settings_from_preset("nonexistent")

    def test_preset_with_basis_override(self):
        s = settings_from_preset("default",
                                  basis_set_per_element={"Fe": "def2-QZVP"})
        self.assertEqual(s.get_basis("Fe"), "def2-QZVP")


class TestBuildBasisDict(unittest.TestCase):
    def _make_cluster(self):
        metals = [
            MetalCenter(element="Fe", index=0, position=[0, 0, 0], label="Fe1"),
            MetalCenter(element="Fe", index=1, position=[2, 0, 0], label="Fe2"),
        ]
        bridges = [
            BridgingAtom(element="S", index=2, position=[1, 1, 0]),
        ]
        return ClusterInfo(
            metals=metals,
            bridging_atoms=bridges,
            all_elements=["Fe", "Fe", "S"],
        )

    def test_build_basis_dict(self):
        ci = self._make_cluster()
        result = build_basis_dict(ci, DEFAULT_PRESET)
        self.assertEqual(result["Fe"], "def2-TZVP")
        self.assertEqual(result["S"], "def2-TZVP")

    def test_build_basis_dict_with_terminal_ligands(self):
        metals = [
            MetalCenter(element="Fe", index=0, position=[0, 0, 0], label="Fe1"),
        ]
        bridges = [
            BridgingAtom(element="S", index=1, position=[1, 0, 0]),
        ]
        from apex_cas import TerminalLigand
        ligands = [
            TerminalLigand(name="thiolate", atom_indices=[2, 3], donor_atom_index=2),
        ]
        ci = ClusterInfo(
            metals=metals,
            bridging_atoms=bridges,
            terminal_ligands=ligands,
            all_elements=["Fe", "S", "C", "H"],
        )
        result = build_basis_dict(ci, DEFAULT_PRESET)
        self.assertIn("Fe", result)
        self.assertIn("S", result)

    def test_build_basis_dict_sorted_keys(self):
        ci = self._make_cluster()
        result = build_basis_dict(ci, DEFAULT_PRESET)
        self.assertEqual(list(result.keys()), sorted(result.keys()))

    def test_build_basis_dict_fast_preset(self):
        ci = self._make_cluster()
        result = build_basis_dict(ci, FAST_PRESET)
        self.assertEqual(result["Fe"], "def2-SVP")
        self.assertEqual(result["S"], "def2-SVP")


class TestGetBasisMethod(unittest.TestCase):
    """Regression coverage for `ComputationSettings.get_basis(...)`."""

    def test_per_element_takes_priority(self):
        s = ComputationSettings(
            basis_set_default="def2-SVP",
            basis_set_per_element={"Fe": "def2-TZVP"},
        )
        self.assertEqual(s.get_basis("Fe"), "def2-TZVP")
        self.assertEqual(s.get_basis("C"), "def2-SVP")

    def test_empty_per_element(self):
        s = ComputationSettings(
            basis_set_default="def2-SVP",
            basis_set_per_element={},
        )
        self.assertEqual(s.get_basis("Fe"), "def2-SVP")

