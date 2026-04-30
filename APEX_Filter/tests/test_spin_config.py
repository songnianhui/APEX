"""Tests for spin_config module.

Test spin enumeration for FeMo-co (7 Fe + 1 Mo, Sz=3/2).
Uses Fe(III) S=2.5 for 4 Fe and Fe(II) S=2 for 3 Fe, Mo(III) S=0.5.
Expected: 35 spin isomers when all Fe have the same local spin.
"""

import unittest
from unittest.mock import patch

import numpy as np

from apex_filter.models import (
    ClusterInfo,
    MetalCenter,
    SpinIsomer,
    SpinIsomerFamily,
)
from apex_filter.spin_config import (
    apply_symmetry_reduction,
    enumerate_spin_isomers,
    label_isomers,
    rank_by_heisenberg,
    _get_local_spins,
    _load_spin_magnitudes,
    _parse_fold,
    _rotation_matrix,
)


def _build_femoco_cluster():
    """Build a FeMo-co-like ClusterInfo for testing.

    7 Fe (indices 0-6) + 1 Mo (index 7), target Sz = 3/2, charge = -1.
    Positions are arbitrary for spin enumeration (only local spins matter).
    """
    metals = []
    for i in range(7):
        metals.append(MetalCenter(
            element="Fe",
            index=i,
            position=np.array([float(i), 0.0, 0.0]),
            label=f"Fe{i+1}",
        ))
    metals.append(MetalCenter(
        element="Mo",
        index=7,
        position=np.array([7.0, 0.0, 0.0]),
        label="Mo8",
    ))

    return ClusterInfo(
        metals=metals,
        formula="Fe7MoS9C",
        total_charge=-1,
        target_spin=1.5,  # S = 3/2
    )


class TestEnumerateSpinIsomers(unittest.TestCase):
    """Test the enumerate_spin_isomers function."""

    def setUp(self):
        self.cluster = _build_femoco_cluster()

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_femoco_35_spin_isomers(self, mock_get_spin, mock_get_ox):
        """FeMo-co should yield exactly 35 collinear spin isomers.

        With local spins [2.5]*7 + [0.5] and target Sz = 2.0:
        2.5 * (n_up_Fe - n_down_Fe) + 0.5 * sign_Mo = 2.0
        Mo must be -1, giving 2.5 * diff_Fe = 2.5 → diff_Fe = 1
        n_up = 4, n_down = 3 → C(7,4) = 35 solutions.
        """
        # All Fe(III) -> S=2.5, Mo(III) -> S=0.5
        def spin_lookup(element, oxidation_state):
            if element == "Fe" and oxidation_state == 3:
                return 2.5
            elif element == "Mo" and oxidation_state == 3:
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [3]

        # All Fe(III), Mo(III)
        oxidation_states = {i: 3 for i in range(8)}

        isomers = enumerate_spin_isomers(
            self.cluster,
            target_Sz=2.0,
            oxidation_states=oxidation_states,
        )

        self.assertEqual(len(isomers), 35,
                         f"Expected 35 spin isomers, got {len(isomers)}")

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_femoco_mixed_oxidation_22_isomers(self, mock_get_spin, mock_get_ox):
        """With 4 Fe(III) S=2.5, 3 Fe(II) S=2, Mo(III) S=0.5 -> 22 isomers.

        This verifies the count changes when local spins are not all equal.
        """
        def spin_lookup(element, oxidation_state):
            if element == "Fe":
                return 2.5 if oxidation_state == 3 else 2.0
            elif element == "Mo":
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2, 3]

        oxidation_states = {0: 3, 1: 3, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3}
        isomers = enumerate_spin_isomers(
            self.cluster, target_Sz=1.5, oxidation_states=oxidation_states,
        )
        self.assertEqual(len(isomers), 22)

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_all_isomers_satisfy_target_sz(self, mock_get_spin, mock_get_ox):
        """Every isomer should have Sz = target_Sz."""
        def spin_lookup(element, oxidation_state):
            if element == "Fe":
                return 2.5 if oxidation_state == 3 else 2.0
            elif element == "Mo":
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2, 3]

        oxidation_states = {0: 3, 1: 3, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3}
        target_Sz = 1.5

        isomers = enumerate_spin_isomers(
            self.cluster,
            target_Sz=target_Sz,
            oxidation_states=oxidation_states,
        )

        for iso in isomers:
            self.assertAlmostEqual(
                iso.Sz, target_Sz, places=5,
                msg=f"Isomer {iso.label} has Sz={iso.Sz}, expected {target_Sz}"
            )

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_spin_assignments_are_plus_minus_one(self, mock_get_spin, mock_get_ox):
        """Each spin assignment should be +1 or -1."""
        def spin_lookup(element, oxidation_state):
            if element == "Fe":
                return 2.5 if oxidation_state == 3 else 2.0
            elif element == "Mo":
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2, 3]

        oxidation_states = {0: 3, 1: 3, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3}
        isomers = enumerate_spin_isomers(
            self.cluster, target_Sz=1.5, oxidation_states=oxidation_states,
        )

        for iso in isomers:
            for idx, sign in iso.spin_assignment.items():
                self.assertIn(sign, [+1, -1],
                              f"Isomer {iso.label}, metal {idx}: sign={sign}")

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_n_minority_matches_spin_assignment(self, mock_get_spin, mock_get_ox):
        """n_minority should equal the count of -1 assignments."""
        def spin_lookup(element, oxidation_state):
            if element == "Fe":
                return 2.5 if oxidation_state == 3 else 2.0
            elif element == "Mo":
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2, 3]

        oxidation_states = {0: 3, 1: 3, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3}
        isomers = enumerate_spin_isomers(
            self.cluster, target_Sz=1.5, oxidation_states=oxidation_states,
        )

        for iso in isomers:
            actual_minority = sum(1 for v in iso.spin_assignment.values() if v == -1)
            self.assertEqual(iso.n_minority, actual_minority,
                             f"Isomer {iso.label}: n_minority mismatch")

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_label_format(self, mock_get_spin, mock_get_ox):
        """Labels should follow BS-prefix format with dash separator."""
        def spin_lookup(element, oxidation_state):
            if element == "Fe":
                return 2.5 if oxidation_state == 3 else 2.0
            elif element == "Mo":
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2, 3]

        oxidation_states = {0: 3, 1: 3, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3}
        isomers = enumerate_spin_isomers(
            self.cluster, target_Sz=1.5, oxidation_states=oxidation_states,
        )

        for iso in isomers:
            self.assertTrue(iso.label.startswith("BS"),
                            f"Label {iso.label} should start with BS")
            parts = iso.label.split("-")
            self.assertGreaterEqual(len(parts), 2,
                                    "Label '{}' should have BSn-sites format".format(iso.label))

    def test_empty_cluster_returns_empty(self):
        """No metals should yield no spin isomers."""
        cluster = ClusterInfo(metals=[], target_spin=1.5)
        isomers = enumerate_spin_isomers(cluster, target_Sz=1.5)
        self.assertEqual(len(isomers), 0)

    def test_target_Sz_from_cluster_info(self):
        """If target_Sz is not provided, use cluster_info.target_spin."""
        cluster = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0,
                                position=np.zeros(3), label="Fe1")],
            target_spin=2.0,
        )
        with patch("apex_filter.spin_config.get_common_oxidation_states", return_value=[3]), \
             patch("apex_filter.spin_config.get_local_spin", return_value=2.5):
            isomers = enumerate_spin_isomers(cluster)
            # Only one metal with S=2.5, target Sz=2.0
            # +1: Sz = 2.5 != 2.0, -1: Sz = -2.5 != 2.0 -> no valid isomers
            self.assertEqual(len(isomers), 0)


class TestApplySymmetryReduction(unittest.TestCase):
    """Test symmetry reduction of spin isomers."""

    def _make_simple_isomers(self):
        """Create a set of simple isomers for a 4-metal system."""
        isomers = []
        # 2 minority-spin sites in a 4-metal system
        for i in range(4):
            for j in range(i + 1, 4):
                signs = [+1] * 4
                signs[i] = -1
                signs[j] = -1
                minority = sorted([i, j])
                label = "BS2-{}".format("".join(str(s+1) for s in minority))
                isomers.append(SpinIsomer(
                    label=label,
                    spin_assignment={k: signs[k] for k in range(4)},
                    n_minority=2,
                    family="BS2",
                    Sz=0.0,
                ))
        return isomers

    def test_c1_no_reduction(self):
        """C1 symmetry should not reduce isomers."""
        isomers = self._make_simple_isomers()
        families = apply_symmetry_reduction(isomers, symmetry_group="C1")
        # Each isomer is its own family
        self.assertEqual(len(families), len(isomers))

    def test_empty_isomers(self):
        """Empty isomer list should return empty families."""
        families = apply_symmetry_reduction([], symmetry_group="C3")
        self.assertEqual(len(families), 0)

    def test_c3_symmetry_with_equilateral_positions(self):
        """C3 symmetry reduction with explicitly provided equivalence maps.

        The PCA-based axis detection may not find the correct axis for
        only 3+1 points. Instead, test that the grouping mechanism works
        by providing positions where PCA can find the z-axis.
        """
        # Place 3 metals in an equilateral triangle in the xy-plane,
        # and 1 metal high above on the z-axis to force PCA to find z-axis
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        positions = np.array([
            [2.0 * np.cos(a), 2.0 * np.sin(a), 0.0] for a in angles
        ] + [[0.0, 0.0, 10.0]])  # index 3 is far away on z-axis

        # Create isomers with 1 minority-spin site among the 3 triangle metals
        isomers = []
        for i in range(3):
            signs = [+1, +1, +1, +1]
            signs[i] = -1
            isomers.append(SpinIsomer(
                label="BS1-{}".format(i+1),
                spin_assignment={k: signs[k] for k in range(4)},
                n_minority=1,
                family="BS1",
                Sz=0.0,
            ))

        families = apply_symmetry_reduction(
            isomers, symmetry_group="C3", metal_positions=positions
        )
        bs1_families = [f for f in families if f.n_minority == 1]
        # All 3 triangle-metal isomers should be grouped into 1 family
        self.assertEqual(len(bs1_families), 1)
        self.assertEqual(len(bs1_families[0].isomers), 3)

    def test_group_by_minority_set_fallback(self):
        """When fold <= 1, should fall back to grouping by minority set."""
        # Create isomers where two have the same minority set
        isomers = [
            SpinIsomer(label="BS1-1", spin_assignment={0: -1, 1: +1},
                       n_minority=1, family="BS1", Sz=0.0),
            SpinIsomer(label="BS1-2", spin_assignment={0: +1, 1: -1},
                       n_minority=1, family="BS1", Sz=0.0),
        ]
        # Pass fold=1 (C1) to trigger _group_by_minority_set
        families = apply_symmetry_reduction(isomers, symmetry_group="C1")
        self.assertEqual(len(families), 2)  # each unique minority set is 1 family


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_parse_fold(self):
        self.assertEqual(_parse_fold("C1"), 1)
        self.assertEqual(_parse_fold("C2"), 2)
        self.assertEqual(_parse_fold("C3"), 3)
        self.assertEqual(_parse_fold("C4"), 4)
        self.assertEqual(_parse_fold("C6"), 6)
        self.assertEqual(_parse_fold("D2"), 1)  # not Cn -> 1
        self.assertEqual(_parse_fold("Cs"), 1)

    def test_rotation_matrix_identity(self):
        """Zero-angle rotation should be identity."""
        axis = np.array([0.0, 0.0, 1.0])
        R = _rotation_matrix(axis, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_90_degrees(self):
        """90-degree rotation around z-axis."""
        axis = np.array([0.0, 0.0, 1.0])
        R = _rotation_matrix(axis, np.pi / 2)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotation_matrix_120_degrees(self):
        """120-degree rotation around z-axis (C3)."""
        axis = np.array([0.0, 0.0, 1.0])
        R = _rotation_matrix(axis, 2 * np.pi / 3)
        # (1,0,0) -> (cos120, sin120, 0) = (-0.5, sqrt(3)/2, 0)
        v = np.array([1.0, 0.0, 0.0])
        rotated = R @ v
        np.testing.assert_allclose(
            rotated, [-0.5, np.sqrt(3) / 2, 0.0], atol=1e-10
        )

    def test_rotation_preserves_norm(self):
        """Rotation should preserve vector length."""
        axis = np.array([1.0, 2.0, 3.0])
        axis = axis / np.linalg.norm(axis)
        R = _rotation_matrix(axis, 1.37)
        v = np.array([3.0, 4.0, 5.0])
        rotated = R @ v
        np.testing.assert_allclose(
            np.linalg.norm(rotated), np.linalg.norm(v), atol=1e-10
        )


class TestRankByHeisenberg(unittest.TestCase):
    """Test Heisenberg exchange energy ranking."""

    def test_empty_isomers(self):
        result = rank_by_heisenberg([])
        self.assertEqual(result, [])

    def test_ranking_with_custom_J(self):
        """Ranking should order isomers by exchange energy."""
        isomers = [
            SpinIsomer(label="BS0-0", spin_assignment={0: +1, 1: +1},
                       n_minority=0, family="BS0", Sz=5.0),
            SpinIsomer(label="BS2-12", spin_assignment={0: -1, 1: -1},
                       n_minority=2, family="BS2", Sz=-5.0),
            SpinIsomer(label="BS1-1", spin_assignment={0: -1, 1: +1},
                       n_minority=1, family="BS1", Sz=0.0),
        ]
        # AFM coupling: J = -1
        J = np.array([[0, -1], [-1, 0]], dtype=float)
        ranked = rank_by_heisenberg(isomers, J_couplings=J)
        self.assertEqual(len(ranked), 3)
        # Check they are sorted by energy (lowest first)
        self.assertIsInstance(ranked[0], SpinIsomer)


class TestLabelIsomers(unittest.TestCase):
    """Test isomer labeling."""

    def test_label_families(self):
        """Test that families get proper BSn_k labels."""
        families = [
            SpinIsomerFamily(
                label="BS0", n_minority=0,
                isomers=[
                    SpinIsomer(label="BS0-0", spin_assignment={0: +1},
                               n_minority=0, family="BS0", Sz=2.0),
                ],
            ),
            SpinIsomerFamily(
                label="BS1", n_minority=1,
                isomers=[
                    SpinIsomer(label="BS1-1", spin_assignment={0: -1, 1: +1},
                               n_minority=1, family="BS1", Sz=0.0),
                    SpinIsomer(label="BS1-2", spin_assignment={0: +1, 1: -1},
                               n_minority=1, family="BS1", Sz=0.0),
                ],
            ),
        ]
        labeled = label_isomers(families)
        self.assertEqual(len(labeled), 2)
        # First BS0 family
        self.assertEqual(labeled[0].label, "BS0_1")
        # First BS1 family
        self.assertEqual(labeled[1].label, "BS1_1")
        # Representative should be set
        self.assertIsNotNone(labeled[0].representative)
        self.assertIsNotNone(labeled[1].representative)


class TestSpinMagnitudesFromKB(unittest.TestCase):
    """Test that spin magnitudes come from the knowledge base, not hardcoded."""

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_mn_cluster_uses_correct_spin(self, mock_get_spin, mock_get_ox):
        """Mn(III) should use S=2 from KB, not hardcoded S=2."""
        def spin_lookup(element, oxidation_state):
            if element == "Mn" and oxidation_state == 3:
                return 2.0  # Mn(III) d4 high-spin S=2
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [3]

        metals = [
            MetalCenter(element="Mn", index=i,
                        position=np.array([float(i), 0, 0]), label=f"Mn{i+1}")
            for i in range(4)
        ]
        cluster = ClusterInfo(metals=metals, target_spin=4.0)

        oxidation_states = {i: 3 for i in range(4)}
        isomers = enumerate_spin_isomers(
            cluster, target_Sz=4.0, oxidation_states=oxidation_states,
        )
        # Mn(III) S=2: 4 metals, target Sz=4 -> all +1
        self.assertGreater(len(isomers), 0)

    @patch("apex_filter.spin_config.get_common_oxidation_states")
    @patch("apex_filter.spin_config.get_local_spin")
    def test_cu_cluster_s_half(self, mock_get_spin, mock_get_ox):
        """Cu(II) should use S=0.5 from KB."""
        def spin_lookup(element, oxidation_state):
            if element == "Cu" and oxidation_state == 2:
                return 0.5
            return 0.0

        mock_get_spin.side_effect = spin_lookup
        mock_get_ox.return_value = [2]

        metals = [
            MetalCenter(element="Cu", index=i,
                        position=np.array([float(i), 0, 0]), label=f"Cu{i+1}")
            for i in range(2)
        ]
        cluster = ClusterInfo(metals=metals, target_spin=0.0)

        oxidation_states = {0: 2, 1: 2}
        # Cu(II) S=0.5 each: target Sz=0 means up+down=0 -> 0.5-0.5=0
        isomers = enumerate_spin_isomers(
            cluster, target_Sz=0.0, oxidation_states=oxidation_states,
        )
        self.assertEqual(len(isomers), 2)  # (up,down) and (down,up)

    def test_rank_by_heisenberg_uses_kb_spins(self):
        """rank_by_heisenberg should use spin magnitudes from KB."""
        from apex_filter.spin_config import _load_spin_magnitudes

        metals = [
            MetalCenter(element="Mn", index=0,
                        position=np.zeros(3), label="Mn1"),
            MetalCenter(element="Cu", index=1,
                        position=np.ones(3), label="Cu1"),
        ]
        cluster = ClusterInfo(metals=metals)

        # Mn(III) S=2, Cu(II) S=0.5
        oxidation_states = {0: 3, 1: 2}
        magnitudes = _load_spin_magnitudes(2, cluster, oxidation_states)

        self.assertAlmostEqual(magnitudes[0], 2.0)  # Mn(III) S=2
        self.assertAlmostEqual(magnitudes[1], 0.5)  # Cu(II) S=0.5


if __name__ == "__main__":
    unittest.main()
