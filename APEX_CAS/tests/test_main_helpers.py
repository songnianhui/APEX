"""Tests for lightweight workflow helpers in apex_cas.main."""

import os
import tempfile
import unittest
from types import SimpleNamespace

from apex_cas import CAS, ComputationSettings
from apex_cas.CAS_tester import _resolve_dmrgci_twodot_to_onedot
from apex_cas.main import (
    _build_localization_params,
    _resolve_fcidump_selection_path,
    _resolve_fcidump_spin_projection,
)


class TestMainHelpers(unittest.TestCase):
    def test_build_localization_params_non_pm_returns_none(self):
        settings = ComputationSettings()
        self.assertIsNone(_build_localization_params(settings, "boys"))

    def test_build_localization_params_pm_uses_settings(self):
        settings = ComputationSettings()
        settings.pm_pop_method = "mulliken"
        settings.pm_conv_tol = 1e-8
        settings.pm_max_cycle = 123
        settings.pm_exponent = 4
        settings.pm_init_guess = "cholesky"

        params = _build_localization_params(settings, "pm")

        self.assertEqual(params["pop_method"], "mulliken")
        self.assertEqual(params["conv_tol"], 1e-8)
        self.assertEqual(params["max_cycle"], 123)
        self.assertEqual(params["exponent"], 4)
        self.assertEqual(params["init_guess"], "cholesky")

    def test_resolve_fcidump_spin_projection_defaults_to_saved_target_spin(self):
        args = SimpleNamespace(spin_projection=None)
        cas = CAS()
        cas.target_spin = 1.5

        self.assertEqual(_resolve_fcidump_spin_projection(args, cas), 1.5)

    def test_resolve_fcidump_spin_projection_falls_back_to_zero(self):
        args = SimpleNamespace(spin_projection=None)
        cas = CAS()
        cas.target_spin = 0.0

        self.assertEqual(_resolve_fcidump_spin_projection(args, cas), 0.0)

    def test_resolve_fcidump_selection_path_prefers_default_case_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            orbitals_dir = os.path.join(output_dir, "orbitals")
            os.makedirs(orbitals_dir, exist_ok=True)
            selection_path = os.path.join(orbitals_dir, "demo_selection.txt")
            with open(selection_path, "w") as f:
                f.write("n-electrons: 2\nn-orbital: 1\n0\n")

            args = SimpleNamespace(active_file=None)
            resolved = _resolve_fcidump_selection_path(args, tmpdir, output_dir, "demo")

        self.assertTrue(resolved.endswith("demo_selection.txt"))

    def test_resolve_dmrgci_twodot_to_onedot_keeps_valid_default(self):
        self.assertEqual(_resolve_dmrgci_twodot_to_onedot(26, 30), 26)

    def test_resolve_dmrgci_twodot_to_onedot_clamps_invalid_equal_max_iter(self):
        self.assertEqual(_resolve_dmrgci_twodot_to_onedot(30, 30), 29)

    def test_resolve_dmrgci_twodot_to_onedot_preserves_zero(self):
        self.assertEqual(_resolve_dmrgci_twodot_to_onedot(0, 30), 0)


if __name__ == "__main__":
    unittest.main()
