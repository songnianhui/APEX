"""Tests for CAS_quality module.

Comprehensive tests for NOON validation, entropy analysis,
composite quality scoring, and report generation.
"""

import unittest

import numpy as np

from apex_cas import CAS, ActiveSpaceQuality
from apex_cas.CAS_quality import (
    validate_noon,
    analyze_entropy,
    compute_quality_score,
    print_quality_report,
)


class TestValidateNoon(unittest.TestCase):
    """Tests for the validate_noon function."""

    def _make_orbitals(self, noon, labels=None):
        n = len(noon)
        return CAS(
            mo_coeff_alpha=np.eye(n),
            occupations=np.array(noon),
            orbital_labels=labels or [f"orb_{i}" for i in range(n)],
            n_electrons=int(round(sum(noon))),
            n_orbitals=n,
        )

    def test_perfect_active_space(self):
        """All NOON between 0.02 and 1.98 -> quality score 1.0"""
        noon = np.array([1.95, 1.80, 1.50, 1.20, 0.80, 0.50, 0.20, 0.05])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertEqual(q.n_doubly_occupied, 0)
        self.assertEqual(q.n_empty, 0)
        self.assertAlmostEqual(q.quality_score, 1.0)

    def test_doubly_occupied_warning(self):
        """NOON > 1.98 should be flagged as doubly occupied."""
        noon = np.array([1.99, 1.50, 0.50, 0.01])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertGreaterEqual(q.n_doubly_occupied, 1)
        self.assertTrue(any("doubly occupied" in w.lower() for w in q.warnings))

    def test_empty_orbitals_warning(self):
        """NOON < 0.02 should be flagged as empty."""
        noon = np.array([1.50, 0.50, 0.01, 0.005])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertGreaterEqual(q.n_empty, 1)

    def test_missing_orbital_types(self):
        noon = np.array([1.50, 0.50])
        labels = ["Fe1_3d", "Fe2_3d"]
        orb = self._make_orbitals(noon, labels)
        expected = [
            {"atom_label": "Fe1", "element": "Fe", "ao_type": "3d", "priority": "required"},
            {"atom_label": "S1", "element": "S", "ao_type": "3p", "priority": "required"},
        ]
        q = validate_noon(orb, expected)
        self.assertTrue(any("S" in t for t in q.missing_orbital_types))

    def test_quality_score_decreases_with_doubly_occupied(self):
        """Each doubly occupied orbital should reduce quality by 0.1."""
        noon_good = np.array([1.50, 0.50, 1.00])
        noon_bad = np.array([1.99, 0.50, 1.00])
        q_good = validate_noon(self._make_orbitals(noon_good))
        q_bad = validate_noon(self._make_orbitals(noon_bad))
        self.assertGreater(q_good.quality_score, q_bad.quality_score)

    def test_quality_score_decreases_with_empty(self):
        """Each empty orbital should reduce quality by 0.05."""
        noon_good = np.array([1.50, 0.50, 1.00])
        noon_bad = np.array([1.50, 0.01, 1.00])
        q_good = validate_noon(self._make_orbitals(noon_good))
        q_bad = validate_noon(self._make_orbitals(noon_bad))
        self.assertGreater(q_good.quality_score, q_bad.quality_score)

    def test_all_near_one_warning(self):
        """All NOON near 1.0 should trigger a warning."""
        noon = np.array([1.01, 0.99, 1.00, 1.02])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertTrue(any("n=1.0" in w for w in q.warnings))

    def test_no_missing_types_when_none_expected(self):
        """Without expected_types, missing_orbital_types should be empty."""
        noon = np.array([1.50, 0.50])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertEqual(q.missing_orbital_types, [])

    def test_quality_score_clamped_to_zero(self):
        """Quality score should never go below 0."""
        noon = np.array([1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertGreaterEqual(q.quality_score, 0.0)

    def test_quality_score_clamped_to_one(self):
        """Quality score should never exceed 1.0."""
        noon = np.array([1.50, 0.50, 1.00])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertLessEqual(q.quality_score, 1.0)

    def test_narrow_noon_spread(self):
        """All NOON in narrow range near 1.0 should trigger warning."""
        noon = np.array([0.98, 1.02, 1.01, 0.99])
        orb = self._make_orbitals(noon)
        q = validate_noon(orb)
        self.assertTrue(any("insufficient" in w.lower() for w in q.warnings))

    def test_orbital_character_map_populated(self):
        """orbital_character_map should be populated from labels."""
        noon = np.array([1.50, 0.50])
        labels = ["Fe1_3d", "S1_3p"]
        orb = self._make_orbitals(noon, labels)
        q = validate_noon(orb)
        self.assertEqual(q.orbital_character_map[0], "Fe1_3d")
        self.assertEqual(q.orbital_character_map[1], "S1_3p")

    def test_expected_types_all_found(self):
        """When all expected types are in labels, missing should be empty."""
        noon = np.array([1.50, 0.50])
        labels = ["Fe1_3d", "S1_3p"]
        orb = self._make_orbitals(noon, labels)
        expected = [
            {"atom_label": "Fe1", "element": "Fe", "ao_type": "3d"},
        ]
        q = validate_noon(orb, expected)
        self.assertEqual(q.missing_orbital_types, [])

    def test_custom_noon_thresholds(self):
        """Test custom noon_lo and noon_hi thresholds."""
        noon = np.array([1.97, 0.03])
        orb = self._make_orbitals(noon)
        # With default thresholds (0.02, 1.98): not flagged
        q_default = validate_noon(orb)
        self.assertEqual(q_default.n_doubly_occupied, 0)
        self.assertEqual(q_default.n_empty, 0)

        # With tighter thresholds: should be flagged
        q_tight = validate_noon(orb, noon_lo=0.05, noon_hi=1.95)
        self.assertGreaterEqual(q_tight.n_doubly_occupied, 1)
        self.assertGreaterEqual(q_tight.n_empty, 1)

    def test_missing_type_with_element_only(self):
        """Test matching by element + ao_type when atom_label is missing."""
        noon = np.array([1.50, 0.50])
        labels = ["Fe_3d", "S_3p"]
        orb = self._make_orbitals(noon, labels)
        expected = [
            {"element": "Fe", "ao_type": "3d"},
        ]
        q = validate_noon(orb, expected)
        self.assertEqual(q.missing_orbital_types, [])

    def test_quality_penalty_for_missing_types(self):
        """Each missing expected type should reduce quality by 0.1."""
        noon = np.array([1.50])
        labels = ["Fe1_3d"]
        orb = self._make_orbitals(noon, labels)

        expected = [
            {"atom_label": "Fe1", "element": "Fe", "ao_type": "3d"},
            {"atom_label": "S1", "element": "S", "ao_type": "3p"},
            {"atom_label": "Mo1", "element": "Mo", "ao_type": "4d"},
        ]
        q = validate_noon(orb, expected)
        # Fe1_3d matches the label, S1_3p and Mo1_4d are missing (2 missing)
        # quality_score = 1.0 - 0.1 * 2 = 0.8
        self.assertAlmostEqual(q.quality_score, 0.8)
        self.assertEqual(len(q.missing_orbital_types), 2)


class TestAnalyzeEntropy(unittest.TestCase):
    """Tests for the analyze_entropy function."""

    def test_classifications(self):
        entropy = [0.8, 0.3, 0.05]
        labels = ["Fe1_3d", "S1_3p", "core_1"]
        result = analyze_entropy(entropy, labels)
        self.assertEqual(result[0]["classification"], "critical")
        self.assertEqual(result[1]["classification"], "active")
        self.assertEqual(result[2]["classification"], "removable")

    def test_recommendations(self):
        entropy = [0.6, 0.2, 0.02]
        labels = ["a", "b", "c"]
        result = analyze_entropy(entropy, labels)
        self.assertIn("Must keep", result[0]["recommendation"])
        self.assertIn("Important", result[1]["recommendation"])
        self.assertIn("removing", result[2]["recommendation"].lower())

    def test_custom_thresholds(self):
        entropy = [0.4, 0.15, 0.05]
        labels = ["a", "b", "c"]
        result = analyze_entropy(entropy, labels,
                                 threshold_critical=0.3,
                                 threshold_removable=0.1)
        self.assertEqual(result[0]["classification"], "critical")
        self.assertEqual(result[1]["classification"], "active")
        self.assertEqual(result[2]["classification"], "removable")

    def test_missing_labels_get_default(self):
        """When labels list is shorter than entropy, default labels are used."""
        entropy = [0.5, 0.3]
        labels = ["Fe1_3d"]
        result = analyze_entropy(entropy, labels)
        self.assertEqual(result[0]["label"], "Fe1_3d")
        self.assertEqual(result[1]["label"], "orb_1")

    def test_result_structure(self):
        entropy = [0.5]
        labels = ["test"]
        result = analyze_entropy(entropy, labels)
        self.assertEqual(len(result), 1)
        self.assertIn("orbital_idx", result[0])
        self.assertIn("label", result[0])
        self.assertIn("entropy", result[0])
        self.assertIn("classification", result[0])
        self.assertIn("recommendation", result[0])

    def test_boundary_values(self):
        """Test exact boundary values for classification thresholds."""
        entropy = [0.5, 0.1]
        labels = ["a", "b"]
        result = analyze_entropy(entropy, labels)
        self.assertEqual(result[0]["classification"], "critical")
        self.assertEqual(result[1]["classification"], "active")

    def test_all_critical(self):
        entropy = [0.8, 0.9, 1.0]
        labels = ["a", "b", "c"]
        result = analyze_entropy(entropy, labels)
        self.assertTrue(all(r["classification"] == "critical" for r in result))


class TestComputeQualityScore(unittest.TestCase):
    """Tests for the compute_quality_score function."""

    def test_score_range(self):
        noon = np.array([1.5, 0.5, 0.01])
        char_map = {0: "Fe1_3d", 1: "S1_3p", 2: "virt"}
        expected = [{"element": "Fe", "ao_type": "3d", "priority": "required"}]
        score = compute_quality_score(noon, char_map, expected)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_noon_perfect_coverage(self):
        """Perfect NOON + perfect coverage should give high score."""
        noon = np.array([1.5, 0.5, 1.0, 1.2])
        char_map = {0: "Fe1_3d", 1: "S1_3p", 2: "Fe2_3d", 3: "S2_3p"}
        expected = [
            {"atom_label": "Fe1", "element": "Fe", "ao_type": "3d"},
            {"atom_label": "S1", "element": "S", "ao_type": "3p"},
        ]
        score = compute_quality_score(noon, char_map, expected)
        self.assertGreater(score, 0.8)

    def test_no_expected_types(self):
        """Without expected types, coverage quality should be neutral (1.0)."""
        noon = np.array([1.5, 0.5])
        char_map = {0: "Fe1_3d", 1: "S1_3p"}
        score = compute_quality_score(noon, char_map, [])
        self.assertGreater(score, 0.0)

    def test_with_entropy(self):
        """Entropy values should be included in the composite score."""
        noon = np.array([1.5, 0.5])
        char_map = {0: "Fe1_3d", 1: "S1_3p"}
        expected = [{"element": "Fe", "ao_type": "3d"}]
        score_no_entropy = compute_quality_score(noon, char_map, expected)
        score_with_entropy = compute_quality_score(
            noon, char_map, expected, entropy_values=[0.5, 0.3]
        )
        # Both should be valid scores
        self.assertGreaterEqual(score_no_entropy, 0.0)
        self.assertLessEqual(score_no_entropy, 1.0)
        self.assertGreaterEqual(score_with_entropy, 0.0)
        self.assertLessEqual(score_with_entropy, 1.0)

    def test_empty_noon(self):
        """Empty NOON array should return a valid score."""
        noon = np.array([])
        char_map = {}
        expected = []
        score = compute_quality_score(noon, char_map, expected)
        self.assertGreaterEqual(score, 0.0)


class TestPrintQualityReport(unittest.TestCase):
    """Tests for the print_quality_report function."""

    def test_report_string(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.5, 0.5]),
            n_doubly_occupied=0,
            n_empty=0,
            quality_score=0.9,
            warnings=[],
        )
        report = print_quality_report(q)
        self.assertIn("Active Space Quality Report", report)
        self.assertIn("0.9", report)

    def test_report_with_warnings(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.99, 0.01]),
            n_doubly_occupied=1,
            n_empty=1,
            quality_score=0.5,
            warnings=["1 orbital(s) with n > 1.98", "1 orbital(s) with n < 0.02"],
        )
        report = print_quality_report(q)
        self.assertIn("WARNING", report)
        self.assertIn("1", report)

    def test_report_good_rating(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.5, 0.5]),
            n_doubly_occupied=0,
            n_empty=0,
            quality_score=0.9,
            warnings=[],
        )
        report = print_quality_report(q)
        self.assertIn("GOOD", report)

    def test_report_warning_rating(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.5]),
            n_doubly_occupied=0,
            n_empty=0,
            quality_score=0.6,
            warnings=[],
        )
        report = print_quality_report(q)
        self.assertIn("WARNING", report)

    def test_report_poor_rating(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.99]),
            n_doubly_occupied=1,
            n_empty=0,
            quality_score=0.3,
            warnings=["doubly occupied"],
        )
        report = print_quality_report(q)
        self.assertIn("POOR", report)

    def test_report_no_noon_data(self):
        q = ActiveSpaceQuality(
            noon_values=None,
            n_doubly_occupied=0,
            n_empty=0,
            quality_score=0.5,
            warnings=[],
        )
        report = print_quality_report(q)
        self.assertIn("no NOON data", report)

    def test_report_missing_types(self):
        q = ActiveSpaceQuality(
            noon_values=np.array([1.5]),
            n_doubly_occupied=0,
            n_empty=0,
            quality_score=0.8,
            missing_orbital_types=["S 3p", "Mo 4d"],
            warnings=[],
        )
        report = print_quality_report(q)
        self.assertIn("S 3p", report)
        self.assertIn("Mo 4d", report)


if __name__ == "__main__":
    unittest.main()
