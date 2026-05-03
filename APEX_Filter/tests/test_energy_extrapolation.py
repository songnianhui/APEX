"""Tests for energy_extrapolation module.

Test DMRG D-extrapolation, CC composite, FNO extrapolation,
correlation increment ratio, and MP2 space correction with synthetic data.
"""

import unittest

import numpy as np

from shared.models import ExtrapolatedEnergy
from apex_filter.energy_extrapolation import (
    cc_composite_energy,
    correlation_increment_ratio,
    dmrg_d_extrapolation,
    fno_extrapolation,
    mp2_space_correction,
)


class TestDMRGExtrapolation(unittest.TestCase):
    """Test DMRG bond dimension extrapolation."""

    def test_synthetic_converging_data(self):
        """Test with synthetic data that converges to a known value.

        Model: E(D) = -100.0 + 0.5 * exp(-0.1 * (ln D)^2)
        E_inf = -100.0
        """
        true_e_inf = -100.0
        true_A = 0.5
        true_kappa = 0.1

        bond_dims = [500, 1000, 2000, 5000]
        energies = [
            true_e_inf + true_A * np.exp(-true_kappa * np.log(D) ** 2)
            for D in bond_dims
        ]

        result = dmrg_d_extrapolation(bond_dims, energies)

        self.assertIsInstance(result, ExtrapolatedEnergy)
        self.assertEqual(result.method, "DMRG_D_extrapolation")
        # The fit should recover E_inf within reasonable tolerance
        self.assertAlmostEqual(result.energy, true_e_inf, places=2,
                               msg=f"E_inf={result.energy:.6f}, expected {true_e_inf}")
        self.assertLess(result.uncertainty, 1.0,
                        "Uncertainty should be small for clean synthetic data")
        self.assertIn("E_inf", result.fit_params)
        self.assertIn("A", result.fit_params)
        self.assertIn("kappa", result.fit_params)

    def test_perfectly_converged_data(self):
        """When all energies are the same, E_inf should be that value."""
        bond_dims = [1000, 2000, 5000]
        energies = [-200.0, -200.0, -200.0]

        result = dmrg_d_extrapolation(bond_dims, energies)
        self.assertAlmostEqual(result.energy, -200.0, places=4)

    def test_monotonically_converging(self):
        """Energies should be monotonically decreasing for converging DMRG."""
        bond_dims = [500, 1000, 2000, 5000, 10000]
        energies = [-150.5, -150.3, -150.15, -150.05, -150.01]

        result = dmrg_d_extrapolation(bond_dims, energies)
        # E_inf should be below the last energy
        self.assertLessEqual(result.energy, energies[-1] + 0.1)

    def test_single_point(self):
        """Single data point should return with infinite uncertainty."""
        result = dmrg_d_extrapolation([1000], [-100.0])
        self.assertEqual(result.energy, -100.0)
        self.assertEqual(result.uncertainty, float("inf"))

    def test_two_points(self):
        """Two points should work (may use fallback)."""
        result = dmrg_d_extrapolation([500, 5000], [-100.5, -100.0])
        # Should return some result without error
        self.assertIsInstance(result, ExtrapolatedEnergy)
        self.assertNotEqual(result.energy, 0.0)

    def test_result_has_description(self):
        """Result should have a human-readable description."""
        bond_dims = [500, 1000, 5000]
        energies = [-100.5, -100.2, -100.05]
        result = dmrg_d_extrapolation(bond_dims, energies)
        self.assertIsInstance(result.description, str)
        self.assertTrue(len(result.description) > 0)

    def test_fit_params_contain_input_data(self):
        """Fit params should include the original bond_dims and energies."""
        bond_dims = [500, 1000, 5000]
        energies = [-100.5, -100.2, -100.05]
        result = dmrg_d_extrapolation(bond_dims, energies)
        self.assertEqual(result.fit_params["bond_dims"], bond_dims)
        self.assertEqual(result.fit_params["energies"], energies)


class TestCCCompositeEnergy(unittest.TestCase):
    """Test CC composite energy calculation."""

    def test_basic_composite(self):
        """Test the additive composite scheme.

        E = E_CCSDT(full) + [E_CCSDTQ(FNO) - E_CCSDT(FNO)]
        """
        e_ccsdt_full = -100.0
        e_ccsdtq_fno = -100.5
        e_ccsdt_fno = -100.3
        # delta_TQ = -100.5 - (-100.3) = -0.2
        # E_composite = -100.0 + (-0.2) = -100.2

        result = cc_composite_energy(e_ccsdt_full, e_ccsdtq_fno, e_ccsdt_fno)

        self.assertAlmostEqual(result.energy, -100.2, places=10)
        self.assertEqual(result.method, "CC_composite")
        self.assertAlmostEqual(result.fit_params["delta_TQ"], -0.2, places=10)
        self.assertAlmostEqual(result.fit_params["E_CCSDT_full"], -100.0)
        self.assertAlmostEqual(result.fit_params["E_CCSDTQ_FNO"], -100.5)
        self.assertAlmostEqual(result.fit_params["E_CCSDT_FNO"], -100.3)

    def test_no_quadruples_correction(self):
        """When CCSDTQ(FNO) == CCSDT(FNO), result equals CCSDT(full)."""
        e_ccsdt_full = -200.0
        e_fno = -200.1

        result = cc_composite_energy(e_ccsdt_full, e_fno, e_fno)
        self.assertAlmostEqual(result.energy, -200.0, places=10)

    def test_positive_quadruples_correction(self):
        """When quadruples lower energy (more negative)."""
        result = cc_composite_energy(-100.0, -101.0, -100.5)
        # delta_TQ = -101.0 - (-100.5) = -0.5
        # E = -100.0 + (-0.5) = -100.5
        self.assertAlmostEqual(result.energy, -100.5, places=10)

    def test_uncertainty_estimate(self):
        """Uncertainty should be 10% of |delta_TQ|."""
        e_ccsdt_full = -100.0
        e_ccsdtq_fno = -100.5
        e_ccsdt_fno = -100.3
        result = cc_composite_energy(e_ccsdt_full, e_ccsdtq_fno, e_ccsdt_fno)
        delta = abs(e_ccsdtq_fno - e_ccsdt_fno)
        expected_unc = delta * 0.1
        self.assertAlmostEqual(result.uncertainty, expected_unc, places=10)

    def test_description_contains_energy(self):
        """Description should include the energy value."""
        result = cc_composite_energy(-100.0, -100.5, -100.3)
        self.assertIn("-100.2", result.description)


class TestFNOExtrapolation(unittest.TestCase):
    """Test FNO threshold extrapolation."""

    def test_linear_extrapolation(self):
        """Test with linear data: E(t) = -0.5 + 0.1*t.

        E(0) should be -0.5.
        """
        thresholds = [1e-3, 5e-4, 1e-4]
        energies = [-0.5 + 0.1 * t for t in thresholds]

        result = fno_extrapolation(thresholds, energies, degree=1)
        self.assertAlmostEqual(result.energy, -0.5, places=4)
        self.assertEqual(result.method, "FNO_extrapolation")

    def test_quadratic_extrapolation(self):
        """Test with quadratic data: E(t) = -1.0 + 0.2*t - 0.5*t^2.

        E(0) = -1.0.
        """
        thresholds = [1e-2, 5e-3, 1e-3]
        energies = [-1.0 + 0.2 * t - 0.5 * t ** 2 for t in thresholds]

        result = fno_extrapolation(thresholds, energies, degree=2)
        self.assertAlmostEqual(result.energy, -1.0, places=3)

    def test_insufficient_data(self):
        """Fewer points than degree+1 should give large uncertainty."""
        result = fno_extrapolation([1e-3], [-0.5], degree=2)
        self.assertEqual(result.uncertainty, float("inf"))

    def test_two_points_linear(self):
        """Two points with degree 1 should extrapolate correctly.

        Linear data: E(t) = -0.4 + (-100)*t
        E(0) = -0.4
        """
        thresholds = [1e-3, 1e-4]
        energies = [-0.4 - 100 * 1e-3, -0.4 - 100 * 1e-4]
        # = [-0.5, -0.41]
        result = fno_extrapolation(thresholds, energies, degree=1)
        self.assertAlmostEqual(result.energy, -0.4, places=2)

    def test_fit_params_include_coefficients(self):
        """Fit params should contain polynomial coefficients."""
        thresholds = [1e-3, 5e-4, 1e-4]
        energies = [-0.5, -0.52, -0.55]
        result = fno_extrapolation(thresholds, energies, degree=2)
        self.assertIn("coefficients", result.fit_params)
        self.assertEqual(result.fit_params["degree"], 2)


class TestCorrelationIncrementRatio(unittest.TestCase):
    """Test correlation increment ratio method."""

    def test_basic_ratio(self):
        """Test simple ratio transfer.

        E_target(high) = E_target(low) * [E_ref(high) / E_ref(low)]
        """
        e_target_low = -100.0
        e_ref_low = -50.0
        e_ref_high = -55.0
        # ratio = -55 / -50 = 1.1
        # E_target(high) = -100 * 1.1 = -110

        result = correlation_increment_ratio(e_target_low, e_ref_low, e_ref_high)
        self.assertAlmostEqual(result.energy, -110.0, places=6)
        self.assertEqual(result.method, "correlation_increment_ratio")
        self.assertAlmostEqual(result.fit_params["ratio"], 1.1, places=6)

    def test_zero_ref_low(self):
        """Zero reference low energy should give ratio = 1.0."""
        result = correlation_increment_ratio(-100.0, 0.0, -50.0)
        self.assertAlmostEqual(result.energy, -100.0, places=6)

    def test_with_exact_reference(self):
        """With exact reference, uncertainty should be estimated."""
        result = correlation_increment_ratio(
            -100.0, -50.0, -55.0, e_ref_exact=-55.5
        )
        self.assertGreater(result.uncertainty, 0)

    def test_ratio_preserves_sign(self):
        """If both references are positive, the ratio should be positive."""
        result = correlation_increment_ratio(100.0, 50.0, 55.0)
        self.assertAlmostEqual(result.energy, 110.0, places=6)


class TestMP2SpaceCorrection(unittest.TestCase):
    """Test MP2 space correction method."""

    def test_basic_correction(self):
        """Test MP2 space correction formula.

        E(large) = E(small_CAS) + [E_MP2(large) - E_MP2(small)]
        """
        e_small_cas = -100.0
        e_mp2_small = -0.5
        e_mp2_large = -0.8
        # delta = -0.8 - (-0.5) = -0.3
        # E = -100.0 + (-0.3) = -100.3

        result = mp2_space_correction(e_small_cas, e_mp2_small, e_mp2_large)
        self.assertAlmostEqual(result.energy, -100.3, places=10)
        self.assertEqual(result.method, "MP2_space_correction")

    def test_no_correction_needed(self):
        """When MP2(large) == MP2(small), result equals small CAS energy."""
        result = mp2_space_correction(-100.0, -0.5, -0.5)
        self.assertAlmostEqual(result.energy, -100.0, places=10)

    def test_positive_correction(self):
        """When large MP2 is higher (less negative) than small."""
        result = mp2_space_correction(-100.0, -0.8, -0.5)
        # delta = -0.5 - (-0.8) = +0.3
        # E = -100.0 + 0.3 = -99.7
        self.assertAlmostEqual(result.energy, -99.7, places=10)

    def test_uncertainty_is_20_percent_of_delta(self):
        """Uncertainty should be 20% of |delta_MP2|."""
        e_small_cas = -100.0
        e_mp2_small = -0.5
        e_mp2_large = -0.8
        result = mp2_space_correction(e_small_cas, e_mp2_small, e_mp2_large)
        delta = abs(e_mp2_large - e_mp2_small)
        expected_unc = delta * 0.2
        self.assertAlmostEqual(result.uncertainty, expected_unc, places=10)

    def test_fit_params(self):
        """Fit params should contain all input energies and delta."""
        result = mp2_space_correction(-100.0, -0.5, -0.8)
        self.assertAlmostEqual(result.fit_params["E_small_CAS"], -100.0)
        self.assertAlmostEqual(result.fit_params["E_MP2_small"], -0.5)
        self.assertAlmostEqual(result.fit_params["E_MP2_large"], -0.8)
        self.assertAlmostEqual(result.fit_params["delta_MP2"], -0.3)

    def test_description(self):
        """Description should contain the energy value."""
        result = mp2_space_correction(-100.0, -0.5, -0.8)
        self.assertIn("-100.3", result.description)


class TestExtrapolatedEnergyDataclass(unittest.TestCase):
    """Test the ExtrapolatedEnergy dataclass."""

    def test_default_values(self):
        """Test default values of ExtrapolatedEnergy."""
        result = ExtrapolatedEnergy()
        self.assertEqual(result.method, "")
        self.assertEqual(result.energy, 0.0)
        self.assertEqual(result.uncertainty, 0.0)
        self.assertEqual(result.fit_params, {})
        self.assertEqual(result.description, "")

    def test_custom_values(self):
        """Test custom values."""
        result = ExtrapolatedEnergy(
            method="test",
            energy=-100.5,
            uncertainty=0.01,
            fit_params={"a": 1},
            description="test result",
        )
        self.assertEqual(result.method, "test")
        self.assertAlmostEqual(result.energy, -100.5)
        self.assertAlmostEqual(result.uncertainty, 0.01)

