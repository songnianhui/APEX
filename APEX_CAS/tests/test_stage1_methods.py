"""Tests for Stage 1 methods in active_space_builder.

Comprehensive tests for topology-aware, knowledge-base, combined,
and expected orbital type generation.
"""

import unittest

import numpy as np

from apex_cas import (
    CAS,
    ActiveSpaceLevel,
    BridgingAtom,
    ClusterInfo,
    MetalCenter,
    TerminalLigand,
)
from apex_cas.CAS_builder_noncomputing import (
    _get_expected_orbital_types,
    _build_from_topology,
    _build_from_knowledge_base,
    build_NC_CAS,
)


def _make_fe2s2():
    """Create a simple Fe2S2 cluster for testing."""
    metals = [
        MetalCenter(element="Fe", index=0, position=np.array([0, 0, 0]), label="Fe1",
                     neighbors=[2, 3], coordination=4),
        MetalCenter(element="Fe", index=1, position=np.array([3, 0, 0]), label="Fe2",
                     neighbors=[2, 3], coordination=4),
    ]
    bridges = [
        BridgingAtom(element="S", index=2, position=np.array([1.5, 1.5, 0]),
                      bridged_metals=[0, 1], role="bridging"),
        BridgingAtom(element="S", index=3, position=np.array([1.5, -1.5, 0]),
                      bridged_metals=[0, 1], role="bridging"),
    ]
    return ClusterInfo(
        metals=metals,
        bridging_atoms=bridges,
        all_elements=["Fe", "Fe", "S", "S"],
        all_positions=np.array([[0, 0, 0], [3, 0, 0], [1.5, 1.5, 0], [1.5, -1.5, 0]]),
        formula="Fe2S2",
        total_charge=-2,
        target_spin=0.0,
    )


def _make_fe4s4():
    """Create a Fe4S4 cubane cluster for testing."""
    metals = [
        MetalCenter(element="Fe", index=i,
                     position=np.array([
                         1.0 if i in (0, 1) else -1.0,
                         1.0 if i in (0, 2) else -1.0,
                         1.0 if i in (0, 3) else -1.0,
                     ]),
                     label=f"Fe{i+1}",
                     coordination=4)
        for i in range(4)
    ]
    bridges = [
        BridgingAtom(element="S", index=4+i,
                     position=np.array([
                         1.0 if i in (0, 2) else -1.0,
                         1.0 if i in (0, 1) else -1.0,
                         1.0 if i in (0, 3) else -1.0,
                     ]) * 0.7,
                     bridged_metals=[j for j in range(4) if j != i],
                     role="bridging")
        for i in range(4)
    ]
    all_elements = ["Fe"] * 4 + ["S"] * 4
    all_positions = np.array([m.position for m in metals] + [b.position for b in bridges])
    return ClusterInfo(
        metals=metals,
        bridging_atoms=bridges,
        all_elements=all_elements,
        all_positions=all_positions,
        formula="Fe4S4",
        total_charge=0,
        target_spin=0.0,
    )


def _make_mo_cluster():
    """Create a cluster with a 4d metal (Mo) for testing."""
    metals = [
        MetalCenter(element="Mo", index=0, position=np.array([0, 0, 0]),
                     label="Mo1", coordination=6),
    ]
    bridges = [
        BridgingAtom(element="S", index=1, position=np.array([2.0, 0, 0]),
                      bridged_metals=[0], role="bridging"),
    ]
    return ClusterInfo(
        metals=metals,
        bridging_atoms=bridges,
        all_elements=["Mo", "S"],
        all_positions=np.array([[0, 0, 0], [2.0, 0, 0]]),
        formula="MoS",
        total_charge=-2,
        target_spin=0.5,
    )


# ──────────────────────────────────────────────────────────────────
# TestGetExpectedOrbitalTypes
# ──────────────────────────────────────────────────────────────────

class TestGetExpectedOrbitalTypes(unittest.TestCase):
    def test_standard_level(self):
        ci = _make_fe2s2()
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.STANDARD)
        self.assertIsInstance(types, list)
        self.assertTrue(len(types) > 0)

        # Should have entries for both Fe and both S
        required = [t for t in types if t["priority"] == "required"]
        self.assertTrue(any(t["element"] == "Fe" and t["ao_type"] == "3d" for t in required))
        self.assertTrue(any(t["element"] == "S" for t in required))

    def test_minimal_level_no_bridges(self):
        ci = _make_fe2s2()
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.MINIMAL)
        # Minimal should only have metals
        self.assertTrue(all(t["element"] == "Fe" for t in types))

    def test_extended_level_has_supplementary(self):
        """Extended level may have supplementary terminal ligand types."""
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1",
                         coordination=6),
        ]
        bridges = [
            BridgingAtom(element="S", index=1, position=np.array([2, 0, 0]),
                          bridged_metals=[0], role="bridging"),
        ]
        ligands = [
            TerminalLigand(name="cysteine_thiolate", atom_indices=[2, 3],
                          donor_atom_index=2, charge=-1, metal_index=0),
        ]
        ci = ClusterInfo(
            metals=metals,
            bridging_atoms=bridges,
            terminal_ligands=ligands,
            all_elements=["Fe", "S", "C", "H"],
        )
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.EXTENDED)
        self.assertTrue(any(t["element"] == "Fe" for t in types))
        self.assertTrue(any(t["element"] == "S" for t in types))

    def test_fe4s4_standard(self):
        """Fe4S4 cubane should have 4 Fe and 4 S entries."""
        ci = _make_fe4s4()
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.STANDARD)
        fe_types = [t for t in types if t["element"] == "Fe"]
        s_types = [t for t in types if t["element"] == "S"]
        self.assertEqual(len(fe_types), 4)
        self.assertEqual(len(s_types), 4)

    def test_mo_cluster_uses_4d(self):
        """Mo cluster should use 4d orbital type."""
        ci = _make_mo_cluster()
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.STANDARD)
        mo_types = [t for t in types if t["element"] == "Mo"]
        self.assertTrue(len(mo_types) > 0)
        self.assertEqual(mo_types[0]["ao_type"], "4d")

    def test_types_have_required_keys(self):
        """All type dicts should have atom_label, element, ao_type, n_expected, priority."""
        ci = _make_fe2s2()
        types = _get_expected_orbital_types(ci, ActiveSpaceLevel.STANDARD)
        for t in types:
            self.assertIn("atom_label", t)
            self.assertIn("element", t)
            self.assertIn("ao_type", t)
            self.assertIn("n_expected", t)
            self.assertIn("priority", t)
            self.assertIn(t["priority"], ("required", "supplementary"))


# ──────────────────────────────────────────────────────────────────
# TestBuildTopologyAware
# ──────────────────────────────────────────────────────────────────

class TestBuildTopologyAware(unittest.TestCase):
    def test_returns_tuple(self):
        ci = _make_fe2s2()
        aspace = _build_from_topology(ci, ActiveSpaceLevel.STANDARD)
        self.assertIsInstance(aspace, CAS)
        self.assertGreater(aspace.n_orbitals, 0)
        self.assertGreater(aspace.n_electrons, 0)

    def test_standard_has_more_than_minimal(self):
        ci = _make_fe2s2()
        std = _build_from_topology(ci, ActiveSpaceLevel.STANDARD)
        mini = _build_from_topology(ci, ActiveSpaceLevel.MINIMAL)
        self.assertGreaterEqual(std.n_orbitals, mini.n_orbitals)

    def test_description_includes_topology(self):
        ci = _make_fe2s2()
        aspace = _build_from_topology(ci, ActiveSpaceLevel.STANDARD)
        self.assertIn("topology-aware", aspace.description)

    def test_minimal_description(self):
        ci = _make_fe2s2()
        aspace = _build_from_topology(ci, ActiveSpaceLevel.MINIMAL)
        self.assertIn("minimal", aspace.description)

    def test_fe4s4_topology(self):
        ci = _make_fe4s4()
        aspace = _build_from_topology(ci, ActiveSpaceLevel.STANDARD)
        # Should have 4*5 = 20 Fe d orbitals + bridging
        self.assertGreaterEqual(aspace.n_orbitals, 20)

    def test_mo_cluster_topology(self):
        ci = _make_mo_cluster()
        aspace = _build_from_topology(ci, ActiveSpaceLevel.STANDARD)
        self.assertGreater(aspace.n_orbitals, 0)
        # Mo should have 5 d orbitals
        mo_group = [og for og in aspace.orbital_groups if og.atom_label == "Mo1"]
        self.assertEqual(len(mo_group), 1)
        self.assertEqual(mo_group[0].n_orbitals, 5)


# ──────────────────────────────────────────────────────────────────
# TestBuildFromKnowledgeBase
# ──────────────────────────────────────────────────────────────────

class TestBuildFromKnowledgeBase(unittest.TestCase):
    def test_returns_tuple(self):
        ci = _make_fe2s2()
        aspace = _build_from_knowledge_base(ci, ActiveSpaceLevel.STANDARD)
        self.assertIsInstance(aspace, CAS)

    def test_standard_level(self):
        ci = _make_fe2s2()
        aspace = _build_from_knowledge_base(ci, ActiveSpaceLevel.STANDARD)
        self.assertGreater(aspace.n_orbitals, 0)
        self.assertIn("knowledge-base", aspace.description)

    def test_minimal_level(self):
        ci = _make_fe2s2()
        aspace = _build_from_knowledge_base(ci, ActiveSpaceLevel.MINIMAL)
        # Should only have metal d orbitals
        for og in aspace.orbital_groups:
            self.assertIn("d", og.orbital_type)

    def test_fe4s4_knowledge_base(self):
        ci = _make_fe4s4()
        aspace = _build_from_knowledge_base(ci, ActiveSpaceLevel.STANDARD)
        self.assertGreaterEqual(aspace.n_orbitals, 20)

    def test_with_explicit_template_name(self):
        """Test passing an explicit (but nonexistent) template name."""
        ci = _make_fe2s2()
        aspace = _build_from_knowledge_base(
            ci, ActiveSpaceLevel.STANDARD,
            template_name="nonexistent_template"
        )
        # Should still return a valid active space (template not found falls back)
        self.assertIsInstance(aspace, CAS)

    def test_with_oxidation_states(self):
        ci = _make_fe2s2()
        aspace = _build_from_knowledge_base(
            ci, ActiveSpaceLevel.STANDARD,
            oxidation_states={0: 3, 1: 2}
        )
        self.assertIsInstance(aspace, CAS)
        # Fe(III) has d5 = 5 electrons, Fe(II) has d6 = 6 electrons
        fe_groups = [og for og in aspace.orbital_groups if "Fe" in og.atom_label]
        self.assertEqual(len(fe_groups), 2)


# ──────────────────────────────────────────────────────────────────
# TestBuildCombined
# ──────────────────────────────────────────────────────────────────

class TestBuildCombined(unittest.TestCase):
    def test_union_approach(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci)
        aspace = cases["combined"]
        self.assertIsInstance(aspace, CAS)

        # Combined should be >= any individual method
        self.assertGreaterEqual(aspace.n_orbitals, cases["rule"].n_orbitals)

    def test_combined_description(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci)
        aspace = cases["combined"]
        self.assertIn("combined", aspace.description)
        self.assertIn("rule=", aspace.description)
        self.assertIn("topology=", aspace.description)
        self.assertIn("knowledge_base=", aspace.description)

    def test_combined_has_source_field(self):
        ci = _make_fe2s2()
        _, types = build_NC_CAS(ci)
        # Expected types should have a 'source' field
        for t in types:
            self.assertIn("source", t)
            self.assertIsInstance(t["source"], list)

    def test_combined_expected_types_deduped(self):
        """Expected types should be deduplicated by (atom_label, ao_type)."""
        ci = _make_fe2s2()
        _, types = build_NC_CAS(ci)
        keys = [(t["atom_label"], t["ao_type"]) for t in types]
        self.assertEqual(len(keys), len(set(keys)))

    def test_fe4s4_combined(self):
        ci = _make_fe4s4()
        cases, _ = build_NC_CAS(ci)
        aspace = cases["combined"]
        self.assertGreaterEqual(aspace.n_orbitals, 20)


# ──────────────────────────────────────────────────────────────────
# TestBuildActiveSpaceLevels
# ──────────────────────────────────────────────────────────────────

class TestBuildActiveSpaceLevels(unittest.TestCase):
    """Test all three active space levels with various clusters."""

    def test_minimal_fe2s2(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        self.assertEqual(aspace.level, ActiveSpaceLevel.MINIMAL)
        # Only metal d orbitals: 2 Fe * 5 d = 10 orbitals
        self.assertEqual(aspace.n_orbitals, 10)

    def test_standard_fe2s2(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.STANDARD)
        aspace = cases["rule"]
        self.assertGreaterEqual(aspace.n_orbitals, 10)

    def test_extended_fe2s2(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.EXTENDED)
        aspace = cases["rule"]
        self.assertGreaterEqual(aspace.n_orbitals, 10)

    def test_qubits_always_double(self):
        for factory in [_make_fe2s2, _make_fe4s4, _make_mo_cluster]:
            ci = factory()
            for level in ActiveSpaceLevel:
                cases, _ = build_NC_CAS(ci, level)
                aspace = cases["rule"]
                self.assertEqual(aspace.n_qubits, 2 * aspace.n_orbitals)

    def test_minimal_no_bridging_groups(self):
        ci = _make_fe2s2()
        cases, _ = build_NC_CAS(ci, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        # All groups should be metal d
        for og in aspace.orbital_groups:
            self.assertIn("d", og.orbital_type)

    def test_mo_cluster_4d_type(self):
        ci = _make_mo_cluster()
        cases, _ = build_NC_CAS(ci)
        aspace = cases["rule"]
        mo_groups = [og for og in aspace.orbital_groups if "Mo" in og.atom_label]
        self.assertTrue(len(mo_groups) > 0)
        self.assertEqual(mo_groups[0].orbital_type, "4d")


if __name__ == "__main__":
    unittest.main()
