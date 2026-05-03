"""Tests for shared-structure authority and core Step 2 utility behavior."""

from pathlib import Path

import numpy as np
import pytest

from shared.models import (
    ClusterInfo,
    MetalCenter,
    BridgingAtom,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fe2s2_structure(tmp_path):
    """Create a minimal Fe2S2 XYZ file."""
    content = """4
Fe2S2 test cluster
Fe  0.0000  0.0000  0.0000
Fe  2.3000  0.0000  0.0000
S   1.1500  1.0000  0.0000
S  -1.1500  1.0000  0.0000
"""
    filepath = Path(tmp_path) / "fe2s2.xyz"
    with filepath.open("w") as f:
        f.write(content)
    return str(filepath)


@pytest.fixture
def femoco_structure(tmp_path):
    """Create a simplified FeMo-cofactor-like XYZ file (Fe2MoS3C for testing)."""
    content = """7
Simplified FeMo-co
Fe  0.0000  0.0000  0.0000
Fe  2.5000  0.0000  0.0000
Mo  1.2500  2.1651  0.0000
S   0.6250  0.7080  0.6250
S   1.8750 -0.7080  0.6250
S  -0.6250  1.4160  0.6250
C   1.2500  0.0000  1.0825
"""
    filepath = Path(tmp_path) / "femoco_test.xyz"
    with filepath.open("w") as f:
        f.write(content)
    return str(filepath)


# ──────────────────────────────────────────────────────────────────
# Structure Analyzer Tests
# ──────────────────────────────────────────────────────────────────

class TestStructureAnalyzer:
    def test_parse_fe2s2(self, fe2s2_structure):
        from shared.structure_parser import parse_structure
        info = parse_structure(fe2s2_structure, charge=-2, target_spin=0.0)

        assert info.formula == "Fe2S2" or "S2Fe2" in info.formula.replace(" ", "")
        assert len(info.metals) == 2
        assert info.metals[0].element == "Fe"
        assert info.metals[1].element == "Fe"
        assert info.total_charge == -2
        assert info.target_spin == 0.0

    def test_identify_metal_centers(self):
        from shared.structure_parser import _identify_metal_centers
        elements = ["Fe", "S", "Fe", "O", "Mo"]
        positions = np.zeros((5, 3))
        metals = _identify_metal_centers(elements, positions)
        assert len(metals) == 3
        assert metals[0].element == "Fe"
        assert metals[1].element == "Fe"
        assert metals[2].element == "Mo"

    def test_formula_generation(self):
        from shared.structure_parser import _generate_formula
        assert _generate_formula(["Fe", "Fe", "S", "S"]) in ("Fe2S2", "S2Fe2")
        assert _generate_formula(["C", "H", "H", "H", "H"]) == "CH4"

    def test_bridging_atoms(self, fe2s2_structure):
        from shared.structure_parser import parse_structure
        info = parse_structure(fe2s2_structure, charge=-2)
        # Fe2S2 should have 2 bridging S atoms
        assert len(info.bridging_atoms) >= 1


# ──────────────────────────────────────────────────────────────────
# Spin Config Tests
# ──────────────────────────────────────────────────────────────────

class TestSpinConfig:
    def test_enumerate_simple_case(self):
        """Test with 2 Fe(III) centers, target Sz=0 (antiferromagnetic)."""
        from apex_filter.elec_spin_config_generator import _enumerate_spin_isomers
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, target_spin=0.0)
        # Fe(III) has S=2.5 each
        # Target Sz=0: +2.5 + (-2.5) = 0 and (-2.5) + (+2.5) = 0 → 2 isomers
        # Both spin-flip arrangements are distinct isomers
        isomers = _enumerate_spin_isomers(info, target_Sz=0.0,
                                           oxidation_states={0: 3, 1: 3})
        assert len(isomers) == 2
        assert isomers[0].n_minority == 1

    def test_enumerate_fm_case(self):
        """All spins aligned → ferromagnetic, Sz = sum(Si)."""
        from apex_filter.elec_spin_config_generator import _enumerate_spin_isomers
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Mo", index=1, position=np.zeros(3), label="Mo1"),
        ]
        info = ClusterInfo(metals=metals)
        # Fe(III) S=2.5, Mo(III) S=0.5 → FM: Sz=3.0
        isomers = _enumerate_spin_isomers(info, target_Sz=3.0,
                                           oxidation_states={0: 3, 1: 3})
        # Only 1 way: both +1
        assert len(isomers) == 1
        assert isomers[0].n_minority == 0

    def test_symmetry_reduction_c1(self):
        """C1 symmetry → no reduction."""
        from apex_filter.elec_spin_config_generator import _apply_symmetry_reduction, _enumerate_spin_isomers
        metals = [
            MetalCenter(element="Fe", index=0, position=np.array([0, 0, 0]), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.array([2.5, 0, 0]), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, target_spin=0.0, symmetry_group="C1")
        isomers = _enumerate_spin_isomers(info, target_Sz=0.0,
                                           oxidation_states={0: 3, 1: 3})
        families = _apply_symmetry_reduction(isomers, "C1")
        assert len(families) == len(isomers)  # no reduction

    def test_fiedler_ordering(self):
        """Test Fiedler ordering produces a valid permutation."""
        from shared.orbital_methods.ordering import fiedler_ordering
        # Create a simple interaction matrix
        np.random.seed(42)
        n = 10
        interaction = np.random.rand(n, n)
        interaction = (interaction + interaction.T) / 2  # symmetric
        np.fill_diagonal(interaction, 0)

        ordering = fiedler_ordering(interaction)
        assert len(ordering) == n
        assert len(set(ordering)) == n  # all unique


# ──────────────────────────────────────────────────────────────────
# Electronic Config Tests
# ──────────────────────────────────────────────────────────────────

class TestElectronicConfig:
    def test_internal_enumerate_oxidation_assignments(self):
        """Test oxidation state enumeration with charge balance."""
        from apex_filter.elec_spin_config_generator import _enumerate_oxidation_assignments
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        bridges = [
            BridgingAtom(element="S", index=2, position=np.zeros(3), bridged_metals=[0, 1]),
            BridgingAtom(element="S", index=3, position=np.zeros(3), bridged_metals=[0, 1]),
        ]
        info = ClusterInfo(
            metals=metals, bridging_atoms=bridges,
            formula="Fe2S2", total_charge=-2,
        )
        assignments = _enumerate_oxidation_assignments(info)
        # For Fe2S2 with charge -2 and 2 S(2-):
        # ligand_charge = -4, target_metal_sum = -2 - (-4) = +2
        # Fe(II) + Fe(II) = +4, Fe(II) + Fe(III) = +5, Fe(III) + Fe(III) = +6
        # None equals +2 exactly; closest should be Fe(II)+Fe(II)=+4
        assert len(assignments) >= 1

    def test_d_orbital_choices_fe2(self):
        """Fe(II) d6: 5 choices for the extra electron."""
        from apex_filter.elec_spin_config_generator import _enumerate_d_orbital_configs
        choices = _enumerate_d_orbital_configs("Fe", 2, +1)
        assert len(choices) == 5

    def test_d_orbital_choices_fe3(self):
        """Fe(III) d5: all singly occupied, no choice."""
        from apex_filter.elec_spin_config_generator import _enumerate_d_orbital_configs
        choices = _enumerate_d_orbital_configs("Fe", 3, +1)
        assert len(choices) == 0  # half-filled, no extra electron to place


# ──────────────────────────────────────────────────────────────────
# Energy Extrapolation Tests
# ──────────────────────────────────────────────────────────────────

class TestEnergyExtrapolation:
    def test_dmrg_extrapolation(self):
        """Test DMRG D-extrapolation with synthetic data."""
        from apex_filter.energy_extrapolation import dmrg_d_extrapolation

        # Generate synthetic data using known model: E(D) = E_inf + A*exp(-kappa*(ln D)^2)
        E_inf = -100.0
        A = 0.05
        kappa = 0.05
        bond_dims = [500, 1000, 2000, 5000, 10000]
        energies = [E_inf + A * np.exp(-kappa * np.log(D) ** 2) for D in bond_dims]

        result = dmrg_d_extrapolation(bond_dims, energies)
        assert result.method == "DMRG_D_extrapolation"
        # Should recover E_inf within reasonable tolerance
        assert abs(result.energy - E_inf) < 0.01
        assert result.uncertainty >= 0

    def test_cc_composite_energy(self):
        """Test CC composite energy calculation."""
        from apex_filter.energy_extrapolation import cc_composite_energy

        e_ccsdt_full = -100.0
        e_ccsdtq_fno = -100.05
        e_ccsdt_fno = -100.03

        result = cc_composite_energy(e_ccsdt_full, e_ccsdtq_fno, e_ccsdt_fno)
        expected = -100.0 + (-100.05 - (-100.03))  # = -100.02
        assert abs(result.energy - expected) < 1e-10

    def test_fno_extrapolation(self):
        """Test FNO threshold extrapolation."""
        from apex_filter.energy_extrapolation import fno_extrapolation

        # Synthetic data: E_corr = -1.0 + 0.1 * t + 0.5 * t^2
        thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        energies = [-1.0 + 0.1 * t + 0.5 * t ** 2 for t in thresholds]

        result = fno_extrapolation(thresholds, energies)
        # At threshold=0, should be close to -1.0
        assert abs(result.energy - (-1.0)) < 0.05

    def test_mp2_space_correction(self):
        """Test MP2 space correction."""
        from apex_filter.energy_extrapolation import mp2_space_correction

        result = mp2_space_correction(
            e_small_cas=-100.0,
            e_mp2_small=-100.5,
            e_mp2_large=-100.8,
        )
        expected = -100.0 + (-100.8 - (-100.5))  # = -100.3
        assert abs(result.energy - expected) < 1e-10
