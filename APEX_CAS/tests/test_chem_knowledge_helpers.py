"""Tests for retained chemistry-knowledge helpers in ``apex_cas``."""

import unittest
import numpy as np

from shared.chem_knowledge import (
    get_common_oxidation_states,
    get_d_electron_count,
    get_local_spin,
    get_n_active_orbitals,
    match_cluster_template,
)
from shared.models import ClusterInfo, MetalCenter


class TestChemKnowledgeExports(unittest.TestCase):
    def test_get_local_spin_fe_ii(self):
        self.assertAlmostEqual(get_local_spin("Fe", 2), 2.0)

    def test_get_local_spin_fe_iii(self):
        self.assertAlmostEqual(get_local_spin("Fe", 3), 2.5)

    def test_get_local_spin_unknown_element(self):
        self.assertEqual(get_local_spin("Xx", 2), 0.0)

    def test_get_d_electron_count_fe_ii(self):
        self.assertEqual(get_d_electron_count("Fe", 2), 6)

    def test_get_d_electron_count_mo_iii(self):
        self.assertEqual(get_d_electron_count("Mo", 3), 3)

    def test_get_d_electron_count_unknown_element(self):
        self.assertEqual(get_d_electron_count("Xx", 2), 0)

    def test_get_common_oxidation_states_fe(self):
        states = get_common_oxidation_states("Fe")
        self.assertIn(2, states)
        self.assertIn(3, states)

    def test_get_common_oxidation_states_unknown_element(self):
        self.assertEqual(get_common_oxidation_states("Xx"), [])

    def test_get_n_active_orbitals_fe(self):
        self.assertEqual(get_n_active_orbitals("Fe"), 5)

    def test_get_n_active_orbitals_unknown_element(self):
        self.assertEqual(get_n_active_orbitals("Xx"), 0)


class TestClusterTemplateMatching(unittest.TestCase):
    def test_match_cluster_template_returns_none_for_unknown_formula(self):
        cluster = ClusterInfo(
            metals=[MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1")],
            formula="Xx9Zz9",
            total_charge=0,
            target_spin=0.0,
        )
        self.assertIsNone(match_cluster_template(cluster))

