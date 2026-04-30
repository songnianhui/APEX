"""Module 8: Population Analysis

Assign and verify oxidation states from converged wavefunctions.

Methods:
1. Meta-Löwdin population: Count d-electrons per metal using orthogonal AO basis
2. Oxidation state assignment from d-electron count
3. Electron density at nucleus (related to Mössbauer isomer shift)
4. Spin population per site (verify broken-symmetry pattern)
"""

import numpy as np

from .models import ClusterInfo, CalculationResult

try:
    from pyscf import scf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


# ──────────────────────────────────────────────────────────────────
# Population analysis
# ──────────────────────────────────────────────────────────────────

class PopulationAnalyzer:
    """Analyze converged wavefunctions to extract oxidation states."""

    def __init__(self, mol=None, dm=None, cluster_info=None):
        """
        Args:
            mol: PySCF Mole object.
            dm: Density matrix, shape (2, nao, nao) for UHF.
            cluster_info: ClusterInfo for atom identification.
        """
        self.mol = mol
        self.dm = dm
        self.cluster_info = cluster_info

    def meta_lowdin_population(self, mol=None, dm=None):
        """Compute Meta-Löwdin population analysis.

        Returns per-atom electron counts for alpha and beta channels.

        Args:
            mol: PySCF Mole object (uses self.mol if None).
            dm: Density matrix (uses self.dm if None).

        Returns:
            Dict with "alpha", "beta", "total", "spin" per-atom arrays.
        """
        if not HAS_PYSCF:
            raise ImportError("PySCF required for population analysis.")

        mol = mol or self.mol
        dm = dm or self.dm

        if mol is None or dm is None:
            raise ValueError("mol and dm must be provided.")

        if dm.ndim == 3:
            dm_a, dm_b = dm[0], dm[1]
        else:
            dm_a = dm / 2
            dm_b = dm / 2

        # Meta-Löwdin Mulliken population
        pop_a, _ = scf.uhf.mulliken_meta(mol, dm_a, s=mol.intor_symmetric("int1e_ovlp"))
        pop_b, _ = scf.uhf.mulliken_meta(mol, dm_b, s=mol.intor_symmetric("int1e_ovlp"))

        # Sum per atom (not per AO)
        aoslices = mol.aoslice_by_atom()
        n_atoms = len(aoslices)
        alpha_pop = np.zeros(n_atoms)
        beta_pop = np.zeros(n_atoms)

        for atom_idx in range(n_atoms):
            _, _, ao_s, ao_e = aoslices[atom_idx]
            alpha_pop[atom_idx] = np.sum(pop_a[ao_s:ao_e])
            beta_pop[atom_idx] = np.sum(pop_b[ao_s:ao_e])

        return {
            "alpha": alpha_pop,
            "beta": beta_pop,
            "total": alpha_pop + beta_pop,
            "spin": alpha_pop - beta_pop,
        }

    def d_electron_population(self, mol=None, dm=None, cluster_info=None):
        """Count d-electrons on each metal center.

        Uses the meta-Löwdin approach but restricted to d-type AOs.

        Args:
            mol: PySCF Mole object.
            dm: Density matrix.
            cluster_info: ClusterInfo.

        Returns:
            Dict {metal_idx: d_electron_count} mapping metal index
            to the number of d electrons.
        """
        mol = mol or self.mol
        dm = dm or self.dm
        cluster_info = cluster_info or self.cluster_info

        if mol is None or dm is None or cluster_info is None:
            raise ValueError("mol, dm, and cluster_info required.")

        if dm.ndim == 3:
            dm_a, dm_b = dm[0], dm[1]
        else:
            dm_a = dm / 2
            dm_b = dm / 2

        S = mol.intor_symmetric("int1e_ovlp")

        # Get d-orbital AO indices for each metal
        result = {}
        for k, metal in enumerate(cluster_info.metals):
            d_indices = _get_d_ao_indices(mol, metal.index)
            if not d_indices:
                result[k] = 0
                continue

            # Compute d-electron count: sum of DM elements on d-orbitals
            # Tr(D * S) restricted to d-AOs gives electron count
            d_count_alpha = np.sum(dm_a[np.ix_(d_indices, d_indices)] *
                                    S[np.ix_(d_indices, d_indices)])
            d_count_beta = np.sum(dm_b[np.ix_(d_indices, d_indices)] *
                                   S[np.ix_(d_indices, d_indices)])

            result[k] = d_count_alpha + d_count_beta

        return result

    def assign_oxidation_states(self, mol=None, dm=None, cluster_info=None):
        """Assign oxidation states from d-electron counts.

        Args:
            mol: PySCF Mole object.
            dm: Density matrix.
            cluster_info: ClusterInfo.

        Returns:
            Dict {metal_idx: oxidation_state}.
        """
        d_counts = self.d_electron_population(mol, dm, cluster_info)
        cluster_info = cluster_info or self.cluster_info

        result = {}
        for k, metal in enumerate(cluster_info.metals):
            d_count = d_counts.get(k, 0)
            # Determine oxidation state from d-count
            # For Fe: d6 = Fe(II), d5 = Fe(III), d4 = Fe(IV)
            # General: oxidation = neutral_d_count - measured_d_count
            neutral_d = _get_neutral_d_count(metal.element)
            ox = neutral_d - int(round(d_count))
            result[k] = ox

        return result

    def spin_population_per_site(self, mol=None, dm=None, cluster_info=None):
        """Compute local spin moment at each metal site.

        Returns (alpha-beta) electron difference restricted to d-AOs.

        Args:
            mol: PySCF Mole object.
            dm: Density matrix.
            cluster_info: ClusterInfo.

        Returns:
            Dict {metal_idx: local_spin} where local_spin is in units of electrons.
        """
        mol = mol or self.mol
        dm = dm or self.dm
        cluster_info = cluster_info or self.cluster_info

        if dm.ndim == 3:
            dm_a, dm_b = dm[0], dm[1]
        else:
            dm_a = dm / 2
            dm_b = dm / 2

        S = mol.intor_symmetric("int1e_ovlp")

        result = {}
        for k, metal in enumerate(cluster_info.metals):
            d_indices = _get_d_ao_indices(mol, metal.index)
            if not d_indices:
                result[k] = 0.0
                continue

            spin_a = np.sum(dm_a[np.ix_(d_indices, d_indices)] *
                            S[np.ix_(d_indices, d_indices)])
            spin_b = np.sum(dm_b[np.ix_(d_indices, d_indices)] *
                            S[np.ix_(d_indices, d_indices)])

            result[k] = spin_a - spin_b

        return result

    def electron_density_at_nucleus(self, mol=None, dm=None):
        """Compute electron density at each nuclear position.

        ρ(R_A) = sum_{μν} D_{μν} φ_μ(R_A) φ_ν(R_A)

        Related to Mössbauer isomer shift (proportional to ρ(0) for s-electrons).

        Args:
            mol: PySCF Mole object.
            dm: Density matrix.

        Returns:
            Array of electron densities at each nucleus.
        """
        mol = mol or self.mol
        dm = dm or self.dm

        if mol is None or dm is None:
            raise ValueError("mol and dm required.")

        coords = mol.atom_coords()
        n_atoms = len(coords)

        if dm.ndim == 3:
            dm_total = dm[0] + dm[1]
        else:
            dm_total = dm

        # Evaluate AO basis functions at nuclear positions
        ao_values = mol.eval_gto("GTOval_sph", coords)  # (n_atoms, nao)

        # rho(R_A) = sum_{μν} D_μν * phi_μ(R_A) * phi_ν(R_A)
        #           = ao_values @ dm_total @ ao_values.T  (diagonal)
        densities = np.zeros(n_atoms)
        for A in range(n_atoms):
            ao = ao_values[A]  # (nao,)
            densities[A] = ao @ dm_total @ ao

        return densities

    def verify_bs_pattern(self, mol=None, dm=None, cluster_info=None,
                           electronic_config=None):
        """Verify that a broken-symmetry density matrix matches the expected pattern.

        Compares computed spin populations with the spin_assignment from
        the ElectronicConfig.

        Args:
            mol: PySCF Mole object.
            dm: Density matrix.
            cluster_info: ClusterInfo.
            electronic_config: ElectronicConfig with expected spin_assignment.

        Returns:
            Dict with "match" (bool), "expected", "computed", "correlation".
        """
        spin_pop = self.spin_population_per_site(mol, dm, cluster_info)
        cluster_info = cluster_info or self.cluster_info

        expected = {}
        computed = {}
        for k, metal in enumerate(cluster_info.metals):
            spin_dir = electronic_config.spin_assignment.get(k, +1)
            # Expected: majority spin is positive, minority is negative
            expected[k] = spin_dir
            # Computed: positive = majority, negative = minority
            computed[k] = +1 if spin_pop.get(k, 0) >= 0 else -1

        match = all(expected[k] == computed[k] for k in expected)
        n_correct = sum(1 for k in expected if expected[k] == computed[k])

        return {
            "match": match,
            "n_correct": n_correct,
            "n_total": len(expected),
            "expected": expected,
            "computed": computed,
            "spin_populations": spin_pop,
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _get_d_ao_indices(mol, atom_idx):
    """Get d-orbital AO indices for an atom."""
    aoslices = mol.aoslice_by_atom()
    if atom_idx >= len(aoslices):
        return []

    _, _, ao_s, ao_e = aoslices[atom_idx]
    ao_labels = mol.ao_labels()

    d_indices = []
    for i in range(ao_s, min(ao_e, len(ao_labels))):
        label = ao_labels[i]
        parts = label.split()
        if len(parts) >= 3 and 'd' in parts[-1].lower():
            d_indices.append(i)

    return d_indices


def _get_neutral_d_count(element):
    """Get the d-electron count for the neutral atom."""
    # Simplified: from the electron configuration
    neutral_d = {
        "Sc": 1, "Ti": 2, "V": 3, "Cr": 5, "Mn": 5, "Fe": 6,
        "Co": 7, "Ni": 8, "Cu": 10, "Zn": 10,
        "Y": 1, "Zr": 2, "Nb": 4, "Mo": 5, "Tc": 5, "Ru": 7,
        "Rh": 8, "Pd": 10, "Ag": 10, "Cd": 10,
        "La": 1, "Hf": 2, "Ta": 3, "W": 4, "Re": 5, "Os": 6,
        "Ir": 7, "Pt": 9, "Au": 10, "Hg": 10,
    }
    return neutral_d.get(element, 5)  # default to half-filling
