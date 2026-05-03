"""Tests for high-level entrypoints in CAS_builder."""

import unittest
from unittest.mock import patch

from apex_cas import ComputationSettings
from apex_cas.CAS_builder import (
    _run_high_spin_scf,
    build_cas_from_mean_field,
    _build_cas_from_cluster,
)
from shared.models import AVASConfig


class TestComputedBuilderEntrypoints(unittest.TestCase):
    def test__build_cas_from_cluster_composes_scf_and_builder_steps(self):
        settings = ComputationSettings()
        with patch(
            "apex_cas.CAS_builder.run_scf_initialization",
            return_value=("mol", "mf", "chk"),
        ) as mocked_scf, patch(
            "apex_cas.CAS_builder.build_cas_from_mean_field",
            return_value="cas",
        ) as mocked_build:
            result = _build_cas_from_cluster(
                "cluster",
                settings,
                cpt_cas_type="uno",
                localization_method="boys",
                projection_threshold=0.4,
                save_dir="tmp",
            )

        self.assertEqual(result, ("cas", "mol", "mf", "chk"))
        mocked_scf.assert_called_once_with("cluster", settings, save_dir="tmp")
        mocked_build.assert_called_once()

    def test_build_cas_from_mean_field_builds_dispatch_inputs(self):
        settings = ComputationSettings()
        settings.scf_method = "uks"
        settings.xc_functional = "BP86"
        settings.pm_pop_method = "mulliken"
        settings.pm_conv_tol = 1e-8
        settings.pm_max_cycle = 77
        settings.pm_exponent = 4
        settings.pm_init_guess = "cholesky"

        with patch(
            "apex_cas.CAS_builder._dispatch_computed_cas_builder",
            return_value="cas",
        ) as mocked:
            result = build_cas_from_mean_field(
                "mol",
                "mf",
                "cluster",
                computation_settings=settings,
                cpt_cas_type="luo",
                localization_method="pm",
                projection_threshold=0.55,
                avas_config=None,
            )

        self.assertEqual(result, "cas")
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["cpt_cas_type"], "luo")
        self.assertEqual(kwargs["source_prefix"], "UKS-BP86")
        self.assertEqual(kwargs["projection_threshold"], 0.55)
        self.assertEqual(kwargs["loc_params"]["pop_method"], "mulliken")
        self.assertEqual(kwargs["loc_params"]["init_guess"], "cholesky")

    def test_dispatch_avas_uses_default_config_when_missing(self):
        with patch(
            "apex_cas.CAS_builder._construct_avas",
            return_value=("cas", []),
        ) as mocked:
            from apex_cas.CAS_builder import _dispatch_computed_cas_builder

            result = _dispatch_computed_cas_builder(
                "mol",
                "mf",
                "cluster",
                cpt_cas_type="avas",
                localization_method="boys",
                projection_threshold=0.3,
                source_prefix="UHF",
                loc_params=None,
                avas_config=None,
            )

        self.assertEqual(result, "cas")
        args, _kwargs = mocked.call_args
        self.assertIsInstance(args[3], AVASConfig)

    def test_run_high_spin_scf_raises_when_unconverged_and_not_allowed(self):
        settings = ComputationSettings(
            scf_stage1_rough=False,
            scf_stage3_newton=False,
            scf_allow_unconverged=False,
        )

        class FakeMF:
            converged = False
            e_tot = -1.23
            cycles = 7

            def kernel(self, *args, **kwargs):
                return None

        with patch("apex_cas.CAS_builder._build_mf_object", return_value=FakeMF()):
            with self.assertRaisesRegex(RuntimeError, "scf_allow_unconverged: true"):
                _run_high_spin_scf("mol", settings)

    def test_run_high_spin_scf_allows_unconverged_when_configured(self):
        settings = ComputationSettings(
            scf_stage1_rough=False,
            scf_stage3_newton=False,
            scf_allow_unconverged=True,
        )

        class FakeMF:
            converged = False
            e_tot = -1.23
            cycles = 7

            def kernel(self, *args, **kwargs):
                return None

        with patch("apex_cas.CAS_builder._build_mf_object", return_value=FakeMF()):
            mf = _run_high_spin_scf("mol", settings)

        self.assertFalse(mf.converged)
