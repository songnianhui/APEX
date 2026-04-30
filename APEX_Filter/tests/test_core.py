"""Tests for structure analyzer,active space builder, spin config, electronic config, and energy extrapolation.

Run with: pytest tests/
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from apex_filter.models import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ElectronicConfig,
    MetalCenter,
    BridgingAtom,
    OxidationAssignment,
    SpinIsomer,
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
    filepath = os.path.join(tmp_path, "fe2s2.xyz")
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


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
    filepath = os.path.join(tmp_path, "femoco_test.xyz")
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


# ──────────────────────────────────────────────────────────────────
# Structure Analyzer Tests
# ──────────────────────────────────────────────────────────────────

class TestStructureAnalyzer:
    def test_parse_fe2s2(self, fe2s2_structure):
        from apex_cas.structure_analyzer import parse_structure
        info = parse_structure(fe2s2_structure, charge=-2, target_spin=0.0)

        assert info.formula == "Fe2S2" or "S2Fe2" in info.formula.replace(" ", "")
        assert len(info.metals) == 2
        assert info.metals[0].element == "Fe"
        assert info.metals[1].element == "Fe"
        assert info.total_charge == -2
        assert info.target_spin == 0.0

    def test_identify_metal_centers(self):
        from apex_cas.structure_analyzer import _identify_metal_centers
        elements = ["Fe", "S", "Fe", "O", "Mo"]
        positions = np.zeros((5, 3))
        metals = _identify_metal_centers(elements, positions)
        assert len(metals) == 3
        assert metals[0].element == "Fe"
        assert metals[1].element == "Fe"
        assert metals[2].element == "Mo"

    def test_formula_generation(self):
        from apex_cas.structure_analyzer import _generate_formula
        assert _generate_formula(["Fe", "Fe", "S", "S"]) in ("Fe2S2", "S2Fe2")
        assert _generate_formula(["C", "H", "H", "H", "H"]) == "CH4"

    def test_bridging_atoms(self, fe2s2_structure):
        from apex_cas.structure_analyzer import parse_structure
        info = parse_structure(fe2s2_structure, charge=-2)
        # Fe2S2 should have 2 bridging S atoms
        assert len(info.bridging_atoms) >= 1


# ──────────────────────────────────────────────────────────────────
# Active Space Builder Tests
# ──────────────────────────────────────────────────────────────────

class TestCASBuilder:
    def test_build_minimal_active_space(self):
        from apex_cas.CAS_builder_noncomputing import build_NC_CAS
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, formula="Fe2S2")
        cases, _ = build_NC_CAS(info, ActiveSpaceLevel.MINIMAL)
        aspace = cases["rule"]
        # 2 Fe × 5 d-orbitals = 10 orbitals minimum
        assert aspace.n_orbitals >= 10
        assert aspace.n_electrons > 0
        assert aspace.level == ActiveSpaceLevel.MINIMAL

    def test_build_standard_active_space(self):
        from apex_cas.CAS_builder_noncomputing import build_NC_CAS
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        bridges = [
            BridgingAtom(element="S", index=2, position=np.zeros(3),
                          bridged_metals=[0, 1], role="bridging"),
            BridgingAtom(element="S", index=3, position=np.zeros(3),
                          bridged_metals=[0, 1], role="bridging"),
        ]
        info = ClusterInfo(
            metals=metals, bridging_atoms=bridges,
            formula="Fe2S2", total_charge=-2,
        )
        cases, _ = build_NC_CAS(info)
        aspace = cases["rule"]
        # Should include metal d + bridging S p
        assert aspace.n_orbitals >= 10  # at least metal d orbitals
        assert aspace.level == ActiveSpaceLevel.STANDARD

    def test_get_local_spin(self):
        from apex_cas.CAS_builder_noncomputing import get_local_spin
        assert get_local_spin("Fe", 2) == 2      # Fe(II) d6, S=2
        assert get_local_spin("Fe", 3) == 2.5    # Fe(III) d5, S=5/2
        assert get_local_spin("Mo", 3) == 0.5    # Mo(III) d3, S=1/2

    def test_get_d_electron_count(self):
        from apex_cas.CAS_builder_noncomputing import get_d_electron_count
        assert get_d_electron_count("Fe", 2) == 6   # Fe(II) d6
        assert get_d_electron_count("Fe", 3) == 5   # Fe(III) d5
        assert get_d_electron_count("Mo", 3) == 3   # Mo(III) d3

    def test_estimate_active_space_size(self):
        from apex_cas.CAS_builder_noncomputing import build_NC_CAS
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, formula="Fe2")
        cases, _ = build_NC_CAS(info)
        cas = cases["rule"]
        n_e, n_o, n_q = cas.n_electrons, cas.n_orbitals, cas.n_qubits
        assert n_e > 0
        assert n_o > 0
        assert n_q == 2 * n_o


# ──────────────────────────────────────────────────────────────────
# Spin Config Tests
# ──────────────────────────────────────────────────────────────────

class TestSpinConfig:
    def test_enumerate_simple_case(self):
        """Test with 2 Fe(III) centers, target Sz=0 (antiferromagnetic)."""
        from apex_filter.spin_config import enumerate_spin_isomers
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.zeros(3), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, target_spin=0.0)
        # Fe(III) has S=2.5 each
        # Target Sz=0: +2.5 + (-2.5) = 0 and (-2.5) + (+2.5) = 0 → 2 isomers
        # Both spin-flip arrangements are distinct isomers
        isomers = enumerate_spin_isomers(info, target_Sz=0.0,
                                          oxidation_states={0: 3, 1: 3})
        assert len(isomers) == 2
        assert isomers[0].n_minority == 1

    def test_enumerate_fm_case(self):
        """All spins aligned → ferromagnetic, Sz = sum(Si)."""
        from apex_filter.spin_config import enumerate_spin_isomers
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Mo", index=1, position=np.zeros(3), label="Mo1"),
        ]
        info = ClusterInfo(metals=metals)
        # Fe(III) S=2.5, Mo(III) S=0.5 → FM: Sz=3.0
        isomers = enumerate_spin_isomers(info, target_Sz=3.0,
                                          oxidation_states={0: 3, 1: 3})
        # Only 1 way: both +1
        assert len(isomers) == 1
        assert isomers[0].n_minority == 0

    def test_symmetry_reduction_c1(self):
        """C1 symmetry → no reduction."""
        from apex_filter.spin_config import enumerate_spin_isomers, apply_symmetry_reduction
        metals = [
            MetalCenter(element="Fe", index=0, position=np.array([0, 0, 0]), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.array([2.5, 0, 0]), label="Fe2"),
        ]
        info = ClusterInfo(metals=metals, target_spin=0.0, symmetry_group="C1")
        isomers = enumerate_spin_isomers(info, target_Sz=0.0,
                                          oxidation_states={0: 3, 1: 3})
        families = apply_symmetry_reduction(isomers, "C1")
        assert len(families) == len(isomers)  # no reduction

    def test_fiedler_ordering(self):
        """Test Fiedler ordering produces a valid permutation."""
        from apex_filter.orbital_ordering import fiedler_ordering
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
    def test_enumerate_oxidation_assignments(self):
        """Test oxidation state enumeration with charge balance."""
        from apex_filter.electronic_config import enumerate_oxidation_assignments
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
        assignments = enumerate_oxidation_assignments(info)
        # For Fe2S2 with charge -2 and 2 S(2-):
        # ligand_charge = -4, target_metal_sum = -2 - (-4) = +2
        # Fe(II) + Fe(II) = +4, Fe(II) + Fe(III) = +5, Fe(III) + Fe(III) = +6
        # None equals +2 exactly; closest should be Fe(II)+Fe(II)=+4
        assert len(assignments) >= 1

    def test_d_orbital_choices_fe2(self):
        """Fe(II) d6: 5 choices for the extra electron."""
        from apex_filter.electronic_config import enumerate_d_orbital_configs
        choices = enumerate_d_orbital_configs("Fe", 2, +1)
        assert len(choices) == 5

    def test_d_orbital_choices_fe3(self):
        """Fe(III) d5: all singly occupied, no choice."""
        from apex_filter.electronic_config import enumerate_d_orbital_configs
        choices = enumerate_d_orbital_configs("Fe", 3, +1)
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


# ──────────────────────────────────────────────────────────────────
# Input Generator Tests
# ──────────────────────────────────────────────────────────────────

class TestInputGenerator:
    def test_generate_pyscf_uhf(self):
        from apex_filter.input_generator import generate_input

        iso = SpinIsomer(
            label="BS1-1",
            spin_assignment={0: +1, 1: -1},
            n_minority=1,
            family="BS1",
            Sz=0.0,
        )
        ox = OxidationAssignment(assignments={0: 3, 1: 3}, description="Fe(III)+Fe(III)")
        config = ElectronicConfig(
            spin_isomer=iso,
            oxidation=ox,
            spin_assignment={0: +1, 1: -1},
            minority_spin_sites=[1],
        )
        aspace = CAS(n_electrons=10, n_orbitals=10)
        metals = [
            MetalCenter(element="Fe", index=0, position=np.zeros(3), label="Fe1"),
            MetalCenter(element="Fe", index=1, position=np.array([2.5, 0, 0]), label="Fe2"),
        ]
        info = ClusterInfo(
            metals=metals,
            all_elements=["Fe", "Fe"],
            all_positions=np.array([[0, 0, 0], [2.5, 0, 0]]),
            formula="Fe2",
            total_charge=0,
        )

        content = generate_input(config, aspace, info, code="pyscf", method="uhf")
        assert "UHF" in content
        assert "from pyscf" in content

    def test_list_templates(self):
        from apex_filter.input_generator import list_available_templates
        templates = list_available_templates()
        assert isinstance(templates, list)

    def test_batch_submission(self):
        from apex_filter.input_generator import generate_batch_submission
        script = generate_batch_submission(
            ["calc1.py", "calc2.py"],
            scheduler="slurm",
            job_name="test",
        )
        assert "#SBATCH" in script
        assert "calc1.py" in script


# ──────────────────────────────────────────────────────────────────
# Filtering Tests
# ──────────────────────────────────────────────────────────────────

class TestFiltering:
    def test_design_funnel(self):
        from apex_filter.filtering import design_filtering_funnel
        aspace = CAS(n_electrons=113, n_orbitals=76)
        plan = design_filtering_funnel(78750, aspace, n_spin_isomers=35)
        assert plan.total_configs == 78750
        assert len(plan.levels) >= 2

    def test_select_lowest_energy(self):
        from apex_filter.filtering import select_lowest_energy
        from apex_filter.models import CalculationResult

        results = [
            CalculationResult(energy=-100.0, converged=True),
            CalculationResult(energy=-99.5, converged=True),
            CalculationResult(energy=-100.5, converged=True),
            CalculationResult(energy=-99.0, converged=False),
        ]
        selected = select_lowest_energy(results, 2)
        assert len(selected) == 2
        assert selected[0].energy == -100.5
        assert selected[1].energy == -100.0
