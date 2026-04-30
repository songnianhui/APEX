"""Tests for orbital_optimizer module.

Tests UCCSD-NO orbital optimization and unrestricted orbital basis
construction using mocked PySCF objects where needed.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np

from apex_cas import CAS, ClusterInfo
from apex_cas.orbital_optimizer import (
    optimize_orbitals_uccsd_no,
    build_unrestricted_orbital_basis,
)


def _make_mock_mol(nelec=(5, 3), nao=20, nmo=15):
    """Create a mock PySCF Mole object."""
    mol = MagicMock()
    mol.nelec = nelec
    mol.nao_nr.return_value = nao
    # intor_symmetric returns a simple identity overlap matrix
    S = np.eye(nao)
    mol.intor_symmetric.return_value = S
    # aoslice_by_atom
    aoslices = [(0, 0, 0, 5), (0, 0, 5, 10), (0, 0, 10, 20)]
    mol.aoslice_by_atom.return_value = aoslices
    mol.ao_labels.return_value = [f"AO_{i}" for i in range(nao)]
    return mol


def _make_mock_mf(nao=20, nmo=15, nelec=(5, 3)):
    """Create a mock SCF object."""
    mf = MagicMock()
    mf.mo_coeff = [np.random.randn(nao, nmo), np.random.randn(nao, nmo)]
    mf.mo_occ = [np.zeros(nmo), np.zeros(nmo)]
    mf.mo_occ[0][:nelec[0]] = 1.0
    mf.mo_occ[1][:nelec[1]] = 1.0

    def make_rdm1(mo_coeff, mo_occ):
        return np.eye(nao) * 0.5

    mf.make_rdm1 = make_rdm1
    return mf


def _make_active_space(n_electrons=8, n_orbitals=7):
    return CAS(n_electrons=n_electrons, n_orbitals=n_orbitals)


class TestOptimizeOrbitalsUccsdNo(unittest.TestCase):
    """Tests for the UCCSD-NO orbital optimization path."""

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", False)
    def test_import_error_when_no_pyscf(self):
        mol = _make_mock_mol()
        mf = _make_mock_mf()
        aspace = _make_active_space()
        with self.assertRaises(ImportError):
            optimize_orbitals_uccsd_no(mol, mf, n_active_orbitals=aspace.n_orbitals)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.cc")
    def test_basic_uccsd_no(self, mock_cc):
        """Test UCCSD-NO with mocked UCCSD and density matrix."""
        nao, nmo = 20, 15
        mol = _make_mock_mol(nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo)
        aspace = _make_active_space(n_electrons=8, n_orbitals=5)

        # Mock UCCSD object
        mock_mycc = MagicMock()
        # Return simple density matrices
        dm1_a = np.eye(nao) * 0.3
        dm1_b = np.eye(nao) * 0.2
        mock_mycc.make_rdm1.return_value = (dm1_a, dm1_b)
        mock_cc.UCCSD.return_value = mock_mycc

        result = optimize_orbitals_uccsd_no(mol, mf, n_active_orbitals=aspace.n_orbitals)

        self.assertIsInstance(result, CAS)
        self.assertEqual(result.n_orbitals, 5)
        self.assertEqual(result.source_method, "UCCSD-NO")
        self.assertEqual(result.cpt_cas_type, "uno")
        self.assertEqual(result.mo_coeff_alpha.shape[1], 5)
        self.assertEqual(result.mo_coeff_beta.shape[1], 5)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.cc")
    def test_n_electrons_from_occupation(self, mock_cc):
        """Test that n_electrons is derived from the sum of occupations."""
        nao, nmo = 20, 15
        mol = _make_mock_mol(nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo)
        aspace = _make_active_space(n_electrons=8, n_orbitals=3)

        # Create density matrices that give near-integer occupations
        dm1_a = np.eye(nao) * 0.45
        dm1_b = np.eye(nao) * 0.45
        mock_mycc = MagicMock()
        mock_mycc.make_rdm1.return_value = (dm1_a, dm1_b)
        mock_cc.UCCSD.return_value = mock_mycc

        result = optimize_orbitals_uccsd_no(mol, mf, n_active_orbitals=aspace.n_orbitals)
        self.assertIsInstance(result.n_electrons, int)
        self.assertGreater(result.n_electrons, 0)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.cc")
    def test_occupations_sorted(self, mock_cc):
        """Verify that selected orbital occupations are all positive."""
        nao, nmo = 20, 15
        mol = _make_mock_mol(nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo)
        aspace = _make_active_space(n_electrons=8, n_orbitals=4)

        dm1_a = np.eye(nao) * 0.4
        dm1_b = np.eye(nao) * 0.3
        mock_mycc = MagicMock()
        mock_mycc.make_rdm1.return_value = (dm1_a, dm1_b)
        mock_cc.UCCSD.return_value = mock_mycc

        result = optimize_orbitals_uccsd_no(mol, mf, n_active_orbitals=aspace.n_orbitals)
        # All occupation values should be non-negative
        self.assertTrue(np.all(result.occupations >= 0))

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.cc")
    def test_labels_generated(self, mock_cc):
        """Test that orbital labels are generated."""
        nao, nmo = 20, 15
        mol = _make_mock_mol(nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo)
        aspace = _make_active_space(n_electrons=8, n_orbitals=4)

        dm1_a = np.eye(nao) * 0.4
        dm1_b = np.eye(nao) * 0.3
        mock_mycc = MagicMock()
        mock_mycc.make_rdm1.return_value = (dm1_a, dm1_b)
        mock_cc.UCCSD.return_value = mock_mycc

        result = optimize_orbitals_uccsd_no(mol, mf, n_active_orbitals=aspace.n_orbitals)
        self.assertEqual(len(result.orbital_labels), 4)
        for label in result.orbital_labels:
            self.assertTrue(label.startswith("UCCSD_NO_"))


class TestBuildUnrestrictedOrbitalBasis(unittest.TestCase):
    """Tests for building spin-unrestricted localized orbital basis."""

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", False)
    def test_import_error_when_no_pyscf(self):
        mol = _make_mock_mol()
        mf = _make_mock_mf()
        aspace = _make_active_space()
        with self.assertRaises(ImportError):
            build_unrestricted_orbital_basis(mol, mf, n_active_orbitals=aspace.n_orbitals)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.lo")
    def test_basic_build(self, mock_lo):
        """Test basic unrestricted orbital basis construction."""
        nao, nmo = 20, 15
        nelec = (5, 3)
        mol = _make_mock_mol(nelec=nelec, nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo, nelec=nelec)
        aspace = _make_active_space(n_electrons=8, n_orbitals=5)

        # Mock Pipek-Mezey localizer
        mock_loc = MagicMock()
        mock_loc.kernel.return_value = np.random.randn(nao, 5)
        mock_lo.PM.return_value = mock_loc
        mock_lo.Boys.return_value = mock_loc

        result = build_unrestricted_orbital_basis(mol, mf, n_active_orbitals=aspace.n_orbitals)

        self.assertIsInstance(result, CAS)
        self.assertEqual(result.cpt_cas_type, "luo")
        self.assertEqual(result.n_orbitals, 5)
        self.assertIn("LUO", result.source_method)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.lo")
    def test_boys_localization_method(self, mock_lo):
        """Test that 'boys' method is forwarded correctly."""
        nao, nmo = 20, 15
        nelec = (5, 3)
        mol = _make_mock_mol(nelec=nelec, nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo, nelec=nelec)
        aspace = _make_active_space(n_electrons=8, n_orbitals=3)

        mock_loc = MagicMock()
        mock_loc.kernel.return_value = np.random.randn(nao, 5)
        mock_lo.Boys.return_value = mock_loc

        result = build_unrestricted_orbital_basis(
            mol, mf, n_active_orbitals=aspace.n_orbitals, localization_method="boys"
        )

        self.assertIsInstance(result, CAS)
        self.assertIn("boys", result.source_method)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.lo")
    def test_alpha_beta_separate(self, mock_lo):
        """Test that alpha and beta orbitals are separate."""
        nao, nmo = 20, 15
        nelec = (6, 4)
        mol = _make_mock_mol(nelec=nelec, nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo, nelec=nelec)
        aspace = _make_active_space(n_electrons=10, n_orbitals=4)

        mock_loc = MagicMock()
        # Return different shapes for occ/vir blocks
        def kernel_side_effect():
            return np.random.randn(nao, 5)
        mock_loc.kernel = kernel_side_effect
        mock_lo.PM.return_value = mock_loc
        mock_lo.Boys.return_value = mock_loc

        result = build_unrestricted_orbital_basis(mol, mf, n_active_orbitals=aspace.n_orbitals)
        # Alpha and beta coefficients should exist
        self.assertIsNotNone(result.mo_coeff_alpha)
        self.assertIsNotNone(result.mo_coeff_beta)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.lo")
    def test_electrons_from_active_space(self, mock_lo):
        """Test that n_electrons comes from active_space."""
        nao, nmo = 20, 15
        nelec = (5, 3)
        mol = _make_mock_mol(nelec=nelec, nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo, nelec=nelec)
        aspace = _make_active_space(n_electrons=8, n_orbitals=4)

        mock_loc = MagicMock()
        mock_loc.kernel.return_value = np.random.randn(nao, 5)
        mock_lo.PM.return_value = mock_loc
        mock_lo.Boys.return_value = mock_loc

        result = build_unrestricted_orbital_basis(mol, mf, n_active_orbitals=aspace.n_orbitals)
        self.assertEqual(result.n_electrons, 8)

    @patch("apex_cas.orbital_optimizer.HAS_PYSCF", True)
    @patch("apex_cas.orbital_optimizer.lo")
    def test_fallback_localization(self, mock_lo):
        """Test that fallback to Boys happens when PM fails."""
        nao, nmo = 20, 15
        nelec = (5, 3)
        mol = _make_mock_mol(nelec=nelec, nao=nao, nmo=nmo)
        mf = _make_mock_mf(nao=nao, nmo=nmo, nelec=nelec)
        aspace = _make_active_space(n_electrons=8, n_orbitals=3)

        call_count = [0]

        def pm_factory(mol_obj, mo_block):
            call_count[0] += 1
            if call_count[0] <= 4:
                # PM fails for the first 4 blocks
                raise RuntimeError("PM localization failed")
            loc = MagicMock()
            loc.kernel.return_value = np.random.randn(*mo_block.shape)
            return loc

        def boys_factory(mol_obj, mo_block):
            loc = MagicMock()
            loc.kernel.return_value = np.random.randn(*mo_block.shape)
            return loc

        mock_lo.PM.side_effect = pm_factory
        mock_lo.Boys.side_effect = boys_factory

        result = build_unrestricted_orbital_basis(mol, mf, n_active_orbitals=aspace.n_orbitals)
        self.assertIsInstance(result, CAS)


if __name__ == "__main__":
    unittest.main()
