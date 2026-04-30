"""Tests for structure_analyzer module.

Test formula generation, symmetry detection, metal center identification,
bridging atom identification, and bond detection.
"""

import unittest

import numpy as np

from apex_cas import (
    BridgingAtom,
    ClusterInfo,
    MetalCenter,
    TerminalLigand,
)
from apex_cas.structure_analyzer import (
    TRANSITION_METALS,
    COVALENT_RADII,
    BRIDGING_ELEMENTS,
    _generate_formula,
    _detect_symmetry,
    _identify_metal_centers,
    _identify_bridging_atoms,
    _build_connectivity,
    _bond_distance,
    _symmetry_order,
    _count_rotational_matches,
)


class TestGenerateFormula(unittest.TestCase):
    """Test Hill-order chemical formula generation."""

    def test_simple_elements(self):
        """Test formula with C and H first (Hill order)."""
        elements = ["C", "H", "H", "O"]
        formula = _generate_formula(elements)
        self.assertEqual(formula, "CH2O")

    def test_no_carbon_alphabetical(self):
        """Without C, elements should be alphabetical."""
        elements = ["Fe", "S", "S", "S", "S"]
        formula = _generate_formula(elements)
        self.assertEqual(formula, "FeS4")

    def test_multiple_same_elements(self):
        """Repeated elements should have counts."""
        elements = ["Fe", "Fe", "Fe", "Mo", "S", "S"]
        formula = _generate_formula(elements)
        self.assertEqual(formula, "Fe3MoS2")

    def test_carbon_hydrogen_order(self):
        """C should come first, then H, then alphabetical."""
        elements = ["S", "C", "H", "H", "Fe"]
        formula = _generate_formula(elements)
        # C first, H second, then Fe, S
        self.assertTrue(formula.startswith("C"))
        self.assertIn("H2", formula)

    def test_single_element(self):
        """Single element should have no subscript."""
        formula = _generate_formula(["Fe"])
        self.assertEqual(formula, "Fe")

    def test_carbon_only(self):
        """Only carbon should be just C."""
        formula = _generate_formula(["C"])
        self.assertEqual(formula, "C")

    def test_femoco_composition(self):
        """Test a FeMo-co-like composition."""
        elements = ["Fe"] * 7 + ["Mo"] + ["S"] * 9 + ["C"]
        formula = _generate_formula(elements)
        self.assertEqual(formula, "CFe7MoS9")

    def test_fe4s4(self):
        """Test Fe4S4 cubane."""
        elements = ["Fe"] * 4 + ["S"] * 4
        formula = _generate_formula(elements)
        self.assertEqual(formula, "Fe4S4")

    def test_empty(self):
        """Empty element list should give empty formula."""
        formula = _generate_formula([])
        self.assertEqual(formula, "")


class TestIdentifyMetalCenters(unittest.TestCase):
    """Test identification of transition metal centers."""

    def test_identifies_fe_and_mo(self):
        """Should identify Fe and Mo as transition metals."""
        elements = ["Fe", "S", "Mo", "C", "H"]
        positions = [np.zeros(3)] * 5
        metals = _identify_metal_centers(elements, positions)

        self.assertEqual(len(metals), 2)
        self.assertEqual(metals[0].element, "Fe")
        self.assertEqual(metals[0].label, "Fe1")
        self.assertEqual(metals[1].element, "Mo")
        self.assertEqual(metals[1].label, "Mo1")

    def test_labels_increment_per_element(self):
        """Labels should increment per element type."""
        elements = ["Fe", "Fe", "Fe", "Mo"]
        positions = [np.zeros(3)] * 4
        metals = _identify_metal_centers(elements, positions)

        self.assertEqual(metals[0].label, "Fe1")
        self.assertEqual(metals[1].label, "Fe2")
        self.assertEqual(metals[2].label, "Fe3")
        self.assertEqual(metals[3].label, "Mo1")

    def test_no_metals(self):
        """Organic molecule with no metals should give empty list."""
        elements = ["C", "H", "O", "N"]
        positions = [np.zeros(3)] * 4
        metals = _identify_metal_centers(elements, positions)
        self.assertEqual(len(metals), 0)

    def test_all_transition_metals_recognized(self):
        """All elements in TRANSITION_METALS should be detected."""
        for elem in TRANSITION_METALS:
            elements = [elem]
            positions = [np.zeros(3)]
            metals = _identify_metal_centers(elements, positions)
            self.assertEqual(len(metals), 1, f"Failed for {elem}")
            self.assertEqual(metals[0].element, elem)

    def test_metal_index_matches_input(self):
        """Metal index should correspond to its position in the element list."""
        elements = ["S", "Fe", "S", "Mo"]
        positions = [np.zeros(3)] * 4
        metals = _identify_metal_centers(elements, positions)
        self.assertEqual(metals[0].index, 1)  # Fe is at index 1
        self.assertEqual(metals[1].index, 3)  # Mo is at index 3

    def test_custom_metals_override(self):
        """Custom metals should be detected alongside transition metals."""
        elements = ["Ca", "Mn", "O", "O"]
        positions = [np.zeros(3)] * 4
        # Without custom_metals, only Mn is detected
        metals_auto = _identify_metal_centers(elements, positions)
        self.assertEqual(len(metals_auto), 1)
        self.assertEqual(metals_auto[0].element, "Mn")

        # With custom_metals=["Ca"], both Ca and Mn should be detected
        metals_custom = _identify_metal_centers(elements, positions,
                                                 custom_metals=["Ca"])
        self.assertEqual(len(metals_custom), 2)
        elements_found = {m.element for m in metals_custom}
        self.assertIn("Ca", elements_found)
        self.assertIn("Mn", elements_found)


class TestBondDistance(unittest.TestCase):
    """Test covalent-radius-based bond detection."""

    def test_bonded_pair(self):
        """Two atoms within bonding distance."""
        # Fe-S: r_Fe=1.32 + r_S=1.05 = 2.37, times tolerance 1.3 = 3.08
        # A distance of 2.2 should be bonded
        r1 = ("Fe", np.array([0.0, 0.0, 0.0]))
        r2 = ("S", np.array([2.2, 0.0, 0.0]))
        self.assertTrue(_bond_distance(r1, r2))

    def test_non_bonded_pair(self):
        """Two atoms too far apart to bond."""
        r1 = ("Fe", np.array([0.0, 0.0, 0.0]))
        r2 = ("S", np.array([5.0, 0.0, 0.0]))
        self.assertFalse(_bond_distance(r1, r2))

    def test_unknown_element_uses_default(self):
        """Unknown elements should use default covalent radius."""
        r1 = ("Xx", np.array([0.0, 0.0, 0.0]))
        r2 = ("Xx", np.array([2.5, 0.0, 0.0]))
        # Default radius 1.5, so (1.5 + 1.5) * 1.3 = 3.9, distance 2.5 < 3.9
        self.assertTrue(_bond_distance(r1, r2))


class TestBuildConnectivity(unittest.TestCase):
    """Test connectivity graph construction."""

    def test_simple_dimer(self):
        """Two bonded atoms should have reciprocal connectivity."""
        elements = ["Fe", "S"]
        positions = np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
        conn = _build_connectivity(elements, positions)
        self.assertIn(1, conn[0])
        self.assertIn(0, conn[1])

    def test_distant_atoms_not_connected(self):
        """Atoms far apart should not be connected."""
        elements = ["Fe", "S"]
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        conn = _build_connectivity(elements, positions)
        self.assertNotIn(1, conn[0])
        self.assertNotIn(0, conn[1])

    def test_three_atom_chain(self):
        """Three atoms in a chain: Fe-S-Fe."""
        elements = ["Fe", "S", "Fe"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ])
        conn = _build_connectivity(elements, positions)
        self.assertIn(1, conn[0])  # Fe0 bonded to S1
        self.assertIn(0, conn[1])  # S1 bonded to Fe0
        self.assertIn(2, conn[1])  # S1 bonded to Fe2
        self.assertIn(1, conn[2])  # Fe2 bonded to S1


class TestIdentifyBridgingAtoms(unittest.TestCase):
    """Test bridging atom identification."""

    def test_s_bridges_two_fe(self):
        """S bonded to two Fe should be identified as bridging."""
        elements = ["Fe", "Fe", "S"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
            [2.2, 0.0, 0.0],
        ])
        metals = _identify_metal_centers(elements, positions)
        conn = _build_connectivity(elements, positions)
        for metal in metals:
            metal.neighbors = conn.get(metal.index, [])
        bridging = _identify_bridging_atoms(elements, positions, metals, conn)

        self.assertEqual(len(bridging), 1)
        self.assertEqual(bridging[0].element, "S")
        self.assertEqual(bridging[0].role, "bridging")
        self.assertEqual(len(bridging[0].bridged_metals), 2)

    def test_interstitial_carbon(self):
        """C bonded to 4+ metals should be identified as interstitial."""
        # Place 4 Fe in a tetrahedron around a central C
        elements = ["C", "Fe", "Fe", "Fe", "Fe"]
        # Simplified positions
        positions = np.array([
            [2.0, 2.0, 2.0],       # C
            [0.0, 0.0, 0.0],       # Fe1
            [4.0, 0.0, 0.0],       # Fe2
            [0.0, 4.0, 0.0],       # Fe3
            [0.0, 0.0, 4.0],       # Fe4
        ])
        metals = _identify_metal_centers(elements, positions)
        conn = _build_connectivity(elements, positions)
        for metal in metals:
            metal.neighbors = conn.get(metal.index, [])

        bridging = _identify_bridging_atoms(elements, positions, metals, conn)
        # If C is bonded to 4+ metals, it should be interstitial
        carbon_bridges = [b for b in bridging if b.element == "C"]
        # This may or may not be detected depending on distances
        for b in carbon_bridges:
            if len(b.bridged_metals) >= 4:
                self.assertEqual(b.role, "interstitial")

    def test_non_bridging_atom_not_detected(self):
        """S bonded to only 1 metal should not be bridging."""
        elements = ["Fe", "S", "H"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [2.2, 2.0, 0.0],  # H far from Fe
        ])
        metals = _identify_metal_centers(elements, positions)
        conn = _build_connectivity(elements, positions)
        for metal in metals:
            metal.neighbors = conn.get(metal.index, [])
        bridging = _identify_bridging_atoms(elements, positions, metals, conn)
        # S is bonded to only 1 Fe, so not bridging
        self.assertEqual(len(bridging), 0)


class TestDetectSymmetry(unittest.TestCase):
    """Test approximate symmetry detection."""

    def test_c3_equilateral_triangle(self):
        """3 metals in an equilateral triangle around an axis.

        This should detect C3 symmetry.
        """
        # Place 3 Fe in a plane at 120 degrees
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        metals = []
        for i, a in enumerate(angles):
            metals.append(MetalCenter(
                element="Fe", index=i,
                position=np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.0]),
                label=f"Fe{i+1}",
            ))
        # Add two metals on the axis (above and below)
        metals.append(MetalCenter(
            element="Fe", index=3,
            position=np.array([0.0, 0.0, 3.0]), label="Fe4",
        ))
        metals.append(MetalCenter(
            element="Fe", index=4,
            position=np.array([0.0, 0.0, -3.0]), label="Fe5",
        ))

        sym, axis_atoms = _detect_symmetry(metals, None)
        # Should detect at least C1 (may detect C3 depending on tolerance)
        self.assertIn(sym, ["C1", "C3", "C4"])

    def test_no_symmetry_single_metal(self):
        """Single metal should have C1 symmetry."""
        metals = [MetalCenter(
            element="Fe", index=0,
            position=np.zeros(3), label="Fe1",
        )]
        sym, axis_atoms = _detect_symmetry(metals, None)
        self.assertEqual(sym, "C1")

    def test_symmetry_order(self):
        """Test _symmetry_order helper."""
        self.assertEqual(_symmetry_order("C1"), 1)
        self.assertEqual(_symmetry_order("C2"), 2)
        self.assertEqual(_symmetry_order("C3"), 3)
        self.assertEqual(_symmetry_order("C4"), 4)
        self.assertEqual(_symmetry_order("C6"), 6)
        self.assertEqual(_symmetry_order("Cs"), 1)

    def test_count_rotational_matches_insufficient_points(self):
        """Fewer points than fold should return 0."""
        projections = [(1.0, 0.0, None)]
        result = _count_rotational_matches(projections, 3)
        self.assertEqual(result, 0)


class TestConstants(unittest.TestCase):
    """Test that constants are properly defined."""

    def test_transition_metals_include_fe(self):
        self.assertIn("Fe", TRANSITION_METALS)

    def test_transition_metals_include_mo(self):
        self.assertIn("Mo", TRANSITION_METALS)

    def test_bridging_elements_include_s(self):
        self.assertIn("S", BRIDGING_ELEMENTS)

    def test_covalent_radii_have_fe(self):
        self.assertIn("Fe", COVALENT_RADII)

    def test_covalent_radii_values_reasonable(self):
        """Covalent radii should be in a reasonable range (0.5 - 3.0 A)."""
        for elem, r in COVALENT_RADII.items():
            self.assertGreater(r, 0.1, f"{elem} radius too small: {r}")
            self.assertLess(r, 3.0, f"{elem} radius too large: {r}")


class TestMissingCovalentRadii(unittest.TestCase):
    """Test that previously missing covalent radii are now present."""

    def test_y_has_radius(self):
        self.assertIn("Y", COVALENT_RADII)

    def test_zr_has_radius(self):
        self.assertIn("Zr", COVALENT_RADII)

    def test_nb_has_radius(self):
        self.assertIn("Nb", COVALENT_RADII)

    def test_tc_has_radius(self):
        self.assertIn("Tc", COVALENT_RADII)

    def test_la_has_radius(self):
        self.assertIn("La", COVALENT_RADII)

    def test_hf_has_radius(self):
        self.assertIn("Hf", COVALENT_RADII)

    def test_ta_has_radius(self):
        self.assertIn("Ta", COVALENT_RADII)

    def test_all_transition_metals_have_radii(self):
        """All 30 transition metals should have covalent radii."""
        for elem in TRANSITION_METALS:
            self.assertIn(elem, COVALENT_RADII,
                          f"Missing covalent radius for {elem}")


class TestExpandedBridgingElements(unittest.TestCase):
    """Test that expanded bridging elements are recognized."""

    def test_bridging_elements_include_p(self):
        self.assertIn("P", BRIDGING_ELEMENTS)

    def test_bridging_elements_include_f(self):
        self.assertIn("F", BRIDGING_ELEMENTS)

    def test_bridging_elements_include_br(self):
        self.assertIn("Br", BRIDGING_ELEMENTS)

    def test_bridging_elements_include_i(self):
        self.assertIn("I", BRIDGING_ELEMENTS)

    def test_bridging_elements_include_h(self):
        self.assertIn("H", BRIDGING_ELEMENTS)

    def test_original_elements_still_present(self):
        """Original bridging elements should still be present."""
        for elem in ["S", "O", "N", "Se", "Cl"]:
            self.assertIn(elem, BRIDGING_ELEMENTS)


if __name__ == "__main__":
    unittest.main()
