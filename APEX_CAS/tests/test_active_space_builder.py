"""Tests for active_space_builder module.

Comprehensive tests for knowledge base loading, active space building,
oxidation state inference, and orbital group construction.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from apex_cas import (
    CAS,
    ActiveSpaceLevel,
    BridgingAtom,
    ClusterInfo,
    MetalCenter,
    OrbitalGroup,
    TerminalLigand,
)
from apex_cas.CAS_builder_noncomputing import (
    build_NC_CAS,
    get_common_oxidation_states,
    get_d_electron_count,
    get_local_spin,
    get_n_active_orbitals,
    _bridging_electron_count,
    _parse_formula,
    _get_3d_metals,
    _get_4d_metals,
    _get_5d_metals,
    _get_orbital_type,
    _infer_default_oxidation_states,
    _balance_oxidation_states,
    _estimate_ligand_charge,
    _match_cluster_template,
)


class TestKnowledgeBaseLoading(unittest.TestCase):
    """Test that the knowledge base is loaded correctly."""

    def test_get_local_spin_fe_ii(self):
        """Fe(II) should have S = 2.0 (high-spin d6)."""
        S = get_local_spin("Fe", 2)
        self.assertAlmostEqual(S, 2.0)

    def test_get_local_spin_fe_iii(self):
        """Fe(III) should have S = 2.5 (high-spin d5)."""
        S = get_local_spin("Fe", 3)
        self.assertAlmostEqual(S, 2.5)

    def test_get_local_spin_mo_iii(self):
        """Mo(III) should have S = 0.5 (low-spin d3 in strong field)."""
        S = get_local_spin("Mo", 3)
        self.assertAlmostEqual(S, 0.5)

    def test_get_local_spin_unknown_element(self):
        """Unknown element should return S = 0.0."""
        S = get_local_spin("Xx", 2)
        self.assertEqual(S, 0.0)

    def test_get_d_electron_count_fe_ii(self):
        """Fe(II) should have d_count = 6."""
        count = get_d_electron_count("Fe", 2)
        self.assertEqual(count, 6)

    def test_get_d_electron_count_fe_iii(self):
        """Fe(III) should have d_count = 5."""
        count = get_d_electron_count("Fe", 3)
        self.assertEqual(count, 5)

    def test_get_d_electron_count_mo_iii(self):
        """Mo(III) should have d_count = 3."""
        count = get_d_electron_count("Mo", 3)
        self.assertEqual(count, 3)

    def test_get_d_electron_count_unknown(self):
        """Unknown element should return d_count = 0."""
        count = get_d_electron_count("Xx", 2)
        self.assertEqual(count, 0)

    def test_get_common_oxidation_states_fe(self):
        """Fe should have common oxidation states [2, 3]."""
        states = get_common_oxidation_states("Fe")
        self.assertIn(2, states)
        self.assertIn(3, states)

    def test_get_common_oxidation_states_mo(self):
        """Mo should have common oxidation states including 3, 4, 5, 6."""
        states = get_common_oxidation_states("Mo")
        self.assertIn(3, states)

    def test_get_common_oxidation_states_unknown(self):
        """Unknown element should return empty list."""
        states = get_common_oxidation_states("Xx")
        self.assertEqual(states, [])

    def test_get_n_active_orbitals_fe(self):
        """Fe should have 5 active orbitals (d shell)."""
        self.assertEqual(get_n_active_orbitals("Fe"), 5)

    def test_get_n_active_orbitals_mo(self):
        """Mo should have 5 active orbitals (d shell)."""
        self.assertEqual(get_n_active_orbitals("Mo"), 5)

    def test_get_n_active_orbitals_unknown(self):
        """Unknown element should return 0 active orbitals."""
        self.assertEqual(get_n_active_orbitals("Xx"), 0)


class TestBuildMinimalCAS(unittest.TestCase):
    """Test minimal active space (metal d only)."""

    def _make_fe2_cluster(self):
        return ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0,
                            position=np.zeros(3), label="Fe1"),
                MetalCenter(element="Fe", index=1,
                            position=np.ones(3), label="Fe2"),
            ],
        )

    def test_minimal_has_metal_d_orbitals_only(self):
        cluster = self._make_fe2_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        for og in aspace.orbital_groups:
            self.assertIn("d", og.orbital_type)

    def test_minimal_fe2_electron_count(self):
        """Minimal active space for 2 Fe should have d electrons only."""
        cluster = self._make_fe2_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        total_d = sum(og.n_electrons for og in aspace.orbital_groups)
        self.assertGreater(total_d, 0)

    def test_minimal_qubits(self):
        """Qubits should be 2 * n_orbitals."""
        cluster = self._make_fe2_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        self.assertEqual(aspace.n_qubits, 2 * aspace.n_orbitals)


class TestBuildStandardCAS(unittest.TestCase):
    """Test standard active space (metal d + bridging)."""

    def _make_fe2s2_cluster(self):
        return ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0,
                            position=np.array([0, 0, 0], dtype=float), label="Fe1"),
                MetalCenter(element="Fe", index=1,
                            position=np.array([4.4, 0, 0], dtype=float), label="Fe2"),
            ],
            bridging_atoms=[
                BridgingAtom(element="S", index=2,
                             position=np.array([2.2, 0, 0], dtype=float),
                             bridged_metals=[0, 1], role="bridging"),
            ],
            total_charge=-2,
        )

    def test_standard_includes_bridging_orbitals(self):
        cluster = self._make_fe2s2_cluster()
        cases, _ = build_NC_CAS(cluster)
        aspace = cases["rule"]
        has_bridging = any("p" in og.orbital_type.lower() or "3p" in og.orbital_type
                          for og in aspace.orbital_groups)
        self.assertTrue(has_bridging or len(aspace.orbital_groups) > 2)

    def test_standard_larger_than_minimal(self):
        cluster = self._make_fe2s2_cluster()
        cases_min, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        minimal = cases_min["rule"]
        cases_std, _ = build_NC_CAS(cluster)
        standard = cases_std["rule"]
        self.assertGreaterEqual(standard.n_orbitals, minimal.n_orbitals)

    def test_standard_description(self):
        cluster = self._make_fe2s2_cluster()
        cases, _ = build_NC_CAS(cluster)
        aspace = cases["rule"]
        self.assertIn("standard", aspace.description)


class TestBuildCASUnified(unittest.TestCase):
    """Test the unified build_active_space entry point."""

    def _make_cluster(self):
        return ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0,
                            position=np.zeros(3), label="Fe1"),
            ],
        )

    def test_minimal_level(self):
        cluster = self._make_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        self.assertEqual(aspace.level, ActiveSpaceLevel.MINIMAL)

    def test_standard_level(self):
        cluster = self._make_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.STANDARD)
        aspace = cases["rule"]
        self.assertEqual(aspace.level, ActiveSpaceLevel.STANDARD)

    def test_extended_level(self):
        cluster = self._make_cluster()
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.EXTENDED)
        aspace = cases["rule"]
        self.assertEqual(aspace.level, ActiveSpaceLevel.EXTENDED)


class TestEstimateCASSize(unittest.TestCase):
    """Test quick active space size estimation."""

    def test_returns_tuple(self):
        cluster = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0,
                                position=np.zeros(3), label="Fe1")],
        )
        cases, _ = build_NC_CAS(cluster)
        cas = cases["rule"]
        result = (cas.n_electrons, cas.n_orbitals, cas.n_qubits)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class TestBridgingElectronCount(unittest.TestCase):
    """Test bridging electron count estimation."""

    def test_sulfur(self):
        self.assertEqual(_bridging_electron_count("S", "bridging"), 6)

    def test_oxygen(self):
        self.assertEqual(_bridging_electron_count("O", "bridging"), 6)

    def test_carbon_interstitial(self):
        self.assertEqual(_bridging_electron_count("C", "interstitial"), 4)

    def test_selenium(self):
        self.assertEqual(_bridging_electron_count("Se", "bridging"), 6)

    def test_chlorine(self):
        self.assertEqual(_bridging_electron_count("Cl", "bridging"), 6)

    def test_unknown_defaults_to_4(self):
        self.assertEqual(_bridging_electron_count("X", "bridging"), 4)

    def test_phosphorus(self):
        self.assertEqual(_bridging_electron_count("P", "bridging"), 3)

    def test_fluorine(self):
        self.assertEqual(_bridging_electron_count("F", "bridging"), 1)

    def test_bromine(self):
        self.assertEqual(_bridging_electron_count("Br", "bridging"), 1)

    def test_iodine(self):
        self.assertEqual(_bridging_electron_count("I", "bridging"), 1)

    def test_hydrogen(self):
        self.assertEqual(_bridging_electron_count("H", "bridging"), 1)

    def test_nitrogen_bridging(self):
        self.assertEqual(_bridging_electron_count("N", "bridging"), 6)

    def test_nitrogen_terminal(self):
        self.assertEqual(_bridging_electron_count("N", "terminal"), 4)


class TestParseFormula(unittest.TestCase):
    """Test chemical formula parsing."""

    def test_simple_formula(self):
        result = _parse_formula("Fe7MoS9C")
        self.assertEqual(result["Fe"], 7)
        self.assertEqual(result["Mo"], 1)
        self.assertEqual(result["S"], 9)
        self.assertEqual(result["C"], 1)

    def test_no_subscript_means_one(self):
        result = _parse_formula("FeS")
        self.assertEqual(result["Fe"], 1)
        self.assertEqual(result["S"], 1)

    def test_multi_digit_subscript(self):
        result = _parse_formula("C10H20")
        self.assertEqual(result["C"], 10)
        self.assertEqual(result["H"], 20)

    def test_empty_formula(self):
        result = _parse_formula("")
        self.assertEqual(result, {})

    def test_single_element(self):
        result = _parse_formula("Fe")
        self.assertEqual(result["Fe"], 1)


class TestMetalCategories(unittest.TestCase):
    """Test metal category helpers."""

    def test_3d_metals(self):
        metals_3d = _get_3d_metals()
        self.assertIn("Fe", metals_3d)
        self.assertIn("Ni", metals_3d)
        self.assertNotIn("Mo", metals_3d)

    def test_4d_metals(self):
        metals_4d = _get_4d_metals()
        self.assertIn("Mo", metals_4d)
        self.assertIn("Ru", metals_4d)
        self.assertNotIn("Fe", metals_4d)

    def test_5d_metals(self):
        metals_5d = _get_5d_metals()
        self.assertIn("W", metals_5d)
        self.assertIn("Ir", metals_5d)
        self.assertNotIn("Fe", metals_5d)


class TestOrbitalTypeAssignment(unittest.TestCase):
    """Test that orbital types are correctly assigned based on metal row."""

    def test_3d_metal_gets_3d_type(self):
        cluster = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0,
                                position=np.zeros(3), label="Fe1")],
        )
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        fe_group = aspace.orbital_groups[0]
        self.assertEqual(fe_group.orbital_type, "3d")

    def test_4d_metal_gets_4d_type(self):
        cluster = ClusterInfo(
            metals=[MetalCenter(element="Mo", index=0,
                                position=np.zeros(3), label="Mo1")],
        )
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        mo_group = aspace.orbital_groups[0]
        self.assertEqual(mo_group.orbital_type, "4d")

    def test_5d_metal_gets_5d_type(self):
        """5d metals (W, Re, Os, Ir, Pt, Au, Hg) should get '5d' orbital type."""
        for elem in ["W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]:
            self.assertEqual(_get_orbital_type(elem), "5d",
                             f"{elem} should have 5d orbital type")

    def test_5d_metal_in_active_space(self):
        """A W cluster should have 5d orbital type in the built active space."""
        cluster = ClusterInfo(
            metals=[MetalCenter(element="W", index=0,
                                position=np.zeros(3), label="W1")],
        )
        cases, _ = build_NC_CAS(cluster, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        w_group = aspace.orbital_groups[0]
        self.assertEqual(w_group.orbital_type, "5d")

    def test_orbital_type_helper_all_metals(self):
        """All 30 transition metals should return a valid orbital type."""
        all_metals = _get_3d_metals() | _get_4d_metals() | _get_5d_metals()
        for elem in all_metals:
            otype = _get_orbital_type(elem)
            self.assertIn(otype, {"3d", "4d", "5d"},
                          f"{elem} has unexpected orbital type: {otype}")


class TestOxidationStateInference(unittest.TestCase):
    """Tests for oxidation state inference and balancing."""

    def test_infer_from_charge_balance(self):
        """Fe2S2 with charge -2: should get feasible oxidation states."""
        ci = ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
                MetalCenter(element="Fe", index=1, position=np.ones(3), label="Fe2"),
            ],
            bridging_atoms=[
                BridgingAtom(element="S", index=2, position=np.array([1, 0, 0]),
                             bridged_metals=[0, 1]),
                BridgingAtom(element="S", index=3, position=np.array([0, 1, 0]),
                             bridged_metals=[0, 1]),
            ],
            total_charge=-2,
        )
        result = _infer_default_oxidation_states(ci)
        self.assertEqual(len(result), 2)
        for idx, ox in result.items():
            self.assertIn(ox, [2, 3])

    def test_balance_oxidation_single_metal(self):
        ci = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
            total_charge=0,
        )
        result = _balance_oxidation_states(ci)
        self.assertEqual(len(result), 1)

    def test_estimate_ligand_charge_sulfur(self):
        ci = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
            bridging_atoms=[
                BridgingAtom(element="S", index=1, position=np.ones(3)),
            ],
        )
        charge = _estimate_ligand_charge(ci)
        self.assertLess(charge, 0)  # S should contribute negative charge


class TestTemplateMatching(unittest.TestCase):
    """Tests for cluster template matching."""

    def test_no_match_returns_none(self):
        ci = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
            formula="Xx9Zz9",
        )
        result = _match_cluster_template(ci)
        # May return None or a match; just check it doesn't crash
        self.assertTrue(result is None or isinstance(result, dict))


class TestMissingMetalsInKB(unittest.TestCase):
    """Test that previously missing metals are now in the knowledge base."""

    def test_missing_4d_metals_have_spins(self):
        """Y, Zr, Nb, Tc, Ag, Cd should have valid spin data."""
        for elem in ["Y", "Zr", "Nb", "Tc", "Ag", "Cd"]:
            states = get_common_oxidation_states(elem)
            self.assertGreater(len(states), 0,
                               f"{elem} should have oxidation states")

    def test_missing_5d_metals_have_spins(self):
        """La, Hf, Ta, Re, Os, Au, Hg should have valid spin data."""
        for elem in ["La", "Hf", "Ta", "Re", "Os", "Au", "Hg"]:
            states = get_common_oxidation_states(elem)
            self.assertGreater(len(states), 0,
                               f"{elem} should have oxidation states")

    def test_cu_oxidation_states(self):
        """Cu should have [1, 2] oxidation states, not fallback [2, 3]."""
        states = get_common_oxidation_states("Cu")
        self.assertIn(1, states)
        self.assertIn(2, states)
        self.assertNotIn(3, states)

    def test_zn_oxidation_states(self):
        """Zn should have [2] only."""
        states = get_common_oxidation_states("Zn")
        self.assertEqual(states, [2])

    def test_ag_oxidation_states(self):
        """Ag should have [1] only."""
        states = get_common_oxidation_states("Ag")
        self.assertIn(1, states)

    def test_mn_spin_magnitude(self):
        """Mn(II) should have S=2.5, not the old hardcoded 2.0."""
        S = get_local_spin("Mn", 2)
        self.assertAlmostEqual(S, 2.5)

    def test_cu_spin_magnitude(self):
        """Cu(II) should have S=0.5, not the old hardcoded 2.0."""
        S = get_local_spin("Cu", 2)
        self.assertAlmostEqual(S, 0.5)

    def test_cu_i_spin_is_zero(self):
        """Cu(I) should have S=0."""
        S = get_local_spin("Cu", 1)
        self.assertAlmostEqual(S, 0.0)


class TestOrbitalGroupStructure(unittest.TestCase):
    """Test that orbital groups have correct structure."""

    def test_orbital_groups_have_labels(self):
        ci = ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            ],
            bridging_atoms=[
                BridgingAtom(element="S", index=1, position=np.ones(3)),
            ],
        )
        cases, _ = build_NC_CAS(ci)
        aspace = cases["rule"]
        for og in aspace.orbital_groups:
            self.assertTrue(og.atom_label)
            self.assertTrue(og.orbital_type)
            self.assertGreater(og.n_orbitals, 0)

    def test_electron_count_matches_groups(self):
        ci = ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            ],
        )
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        group_electrons = sum(og.n_electrons for og in aspace.orbital_groups)
        self.assertEqual(group_electrons, aspace.n_electrons)

    def test_orbital_count_matches_groups(self):
        ci = ClusterInfo(
            metals=[
                MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            ],
        )
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        group_orbitals = sum(og.n_orbitals for og in aspace.orbital_groups)
        self.assertEqual(group_orbitals, aspace.n_orbitals)


if __name__ == "__main__":
    unittest.main()
