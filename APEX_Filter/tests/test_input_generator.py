"""Tests for input_generator module.

Test PySCF UHF input generation with mock config.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from apex_filter.models import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ElectronicConfig,
    MetalCenter,
    OxidationAssignment,
    SpinIsomer,
)
from apex_filter.input_generator import (
    generate_input,
    generate_batch,
    generate_batch_submission,
    list_available_templates,
    _build_context,
    _get_template_name,
    _get_file_extension,
    _gen_pyscf_uhf,
    _gen_pyscf_ccsd,
    _gen_pyscf_casscf,
    _gen_orca_bsdft,
    _gen_gaussian_bsdft,
)


def _make_mock_config():
    """Create a mock ElectronicConfig for testing."""
    isomer = SpinIsomer(
        label="BS1-1",
        spin_assignment={0: -1, 1: +1},
        n_minority=1,
        family="BS1",
        Sz=0.5,
    )
    oxidation = OxidationAssignment(
        assignments={0: 2, 1: 3},
        description="Fe(II)+Fe(III)",
    )
    return ElectronicConfig(
        spin_isomer=isomer,
        oxidation=oxidation,
        d_orbital_assignments={0: 2},
        minority_spin_sites=[0],
        spin_assignment={0: -1, 1: +1},
        config_id=42,
        label="BS1-1|Fe(II)+Fe(III)|d0:2",
    )


def _make_mock_cluster():
    """Create a minimal ClusterInfo for testing."""
    return ClusterInfo(
        metals=[
            MetalCenter(element="Fe", index=0,
                        position=np.array([0.0, 0.0, 0.0]), label="Fe1"),
            MetalCenter(element="Fe", index=1,
                        position=np.array([2.5, 0.0, 0.0]), label="Fe2"),
        ],
        all_elements=["Fe", "Fe"],
        all_positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
        total_charge=-2,
        target_spin=0.5,
    )


def _make_mock_active_space():
    """Create a minimal CAS for testing."""
    return CAS(
        n_electrons=11,
        n_orbitals=10,
        level=ActiveSpaceLevel.MINIMAL,
        description="(11e, 10o) minimal",
    )


class TestBuildContext(unittest.TestCase):
    """Test the _build_context helper."""

    def test_context_contains_required_keys(self):
        """Context should have all required template variables."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        ctx = _build_context(config, aspace, cluster, "uhf")

        self.assertIn("geometry", ctx)
        self.assertIn("charge", ctx)
        self.assertIn("spin", ctx)
        self.assertIn("n_electrons", ctx)
        self.assertIn("n_orbitals", ctx)
        self.assertIn("minority_sites", ctx)
        self.assertIn("spin_assignment", ctx)
        self.assertIn("d_orbital_assignments", ctx)
        self.assertIn("basis_set", ctx)
        self.assertIn("label", ctx)

    def test_spin_multiplicity(self):
        """Spin should be 2*S (number of unpaired electrons)."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()
        ctx = _build_context(config, aspace, cluster, "uhf")
        self.assertEqual(ctx["spin"], 1)  # 2 * 0.5 = 1

    def test_charge_from_cluster(self):
        """Charge should come from cluster_info."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()
        ctx = _build_context(config, aspace, cluster, "uhf")
        self.assertEqual(ctx["charge"], -2)

    def test_basis_set_default(self):
        """Default basis set should be cc-pVDZ."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()
        ctx = _build_context(config, aspace, cluster, "uhf")
        self.assertEqual(ctx["basis_set"], "cc-pVDZ")

    def test_basis_set_override(self):
        """Basis set should be overridable via kwargs."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()
        ctx = _build_context(config, aspace, cluster, "uhf",
                             basis_set="def2-TZVP")
        self.assertEqual(ctx["basis_set"], "def2-TZVP")

    def test_kwargs_passed_to_context(self):
        """Extra kwargs should be passed to context."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()
        ctx = _build_context(config, aspace, cluster, "uhf",
                             custom_param=42)
        self.assertEqual(ctx["custom_param"], 42)


class TestGenPySCFUHF(unittest.TestCase):
    """Test PySCF UHF input generation."""

    def test_generates_python_script(self):
        """Output should be a valid Python script."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)

        self.assertIn("#!/usr/bin/env python3", output)
        self.assertIn("from pyscf import gto, scf", output)
        self.assertIn("scf.UHF", output)
        self.assertIn("mf.kernel", output)

    def test_includes_geometry(self):
        """Output should contain the molecular geometry."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)
        self.assertIn("Fe", output)
        self.assertIn("atom", output)

    def test_includes_charge_and_spin(self):
        """Output should contain correct charge and spin."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)
        self.assertIn("charge=-2", output)
        self.assertIn("spin=1", output)

    def test_includes_spin_flip_for_minority(self):
        """Output should include spin flip for minority-spin sites."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)
        # minority_spin_sites = [0], so should have spin flip code
        self.assertIn("Flip", output)
        self.assertIn("dm_a", output)
        self.assertIn("dm_b", output)

    def test_includes_config_label(self):
        """Output should contain the configuration label (sanitized)."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)
        # Label is sanitized: pipes become underscores
        safe_label = config.label.replace("|", "_")
        self.assertIn(safe_label, output)

    def test_no_minority_sites(self):
        """Without minority sites, no spin flip code should be generated."""
        isomer = SpinIsomer(
            label="BS0-0",
            spin_assignment={0: +1, 1: +1},
            n_minority=0, family="BS0", Sz=5.0,
        )
        config = ElectronicConfig(
            spin_isomer=isomer,
            minority_spin_sites=[],
            spin_assignment={0: +1, 1: +1},
            label="BS0-0|test",
        )
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_uhf(config, aspace, cluster)
        self.assertNotIn("Flip Fe1", output)


class TestGenPySCFCCSD(unittest.TestCase):
    """Test PySCF CCSD input generation."""

    def test_generates_ccsd_script(self):
        """Output should be a CCSD script."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_ccsd(config, aspace, cluster)
        self.assertIn("cc.UCCSD", output)
        self.assertIn("ccsd_t", output)
        self.assertIn("UCCSD Energy", output)


class TestGenPySCFCASSCF(unittest.TestCase):
    """Test PySCF CASSCF input generation."""

    def test_generates_casscf_script(self):
        """Output should be a CASSCF script."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_pyscf_casscf(config, aspace, cluster)
        self.assertIn("mcscf.UCASSCF", output)
        self.assertIn("CASSCF Energy", output)
        # Should contain active space size
        self.assertIn(str(aspace.n_orbitals), output)
        self.assertIn(str(aspace.n_electrons), output)


class TestGenOrcaBSDFT(unittest.TestCase):
    """Test ORCA BS-DFT input generation."""

    def test_generates_orca_input(self):
        """Output should be an ORCA input."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_orca_bsdft(config, aspace, cluster)
        self.assertIn("B3LYP", output)
        self.assertIn("TightSCF", output)
        self.assertIn("FlipSpin", output)


class TestGenGaussianBSDFT(unittest.TestCase):
    """Test Gaussian BS-DFT input generation."""

    def test_generates_gaussian_input(self):
        """Output should be a Gaussian input."""
        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = _gen_gaussian_bsdft(config, aspace, cluster)
        self.assertIn("Guess=Mix", output)
        self.assertIn("SCF=QC", output)


class TestGetTemplateName(unittest.TestCase):
    """Test template name resolution."""

    def test_pyscf_uhf(self):
        self.assertEqual(
            _get_template_name("pyscf", "uhf"), "pyscf_uhf.py.j2"
        )

    def test_pyscf_ccsd(self):
        self.assertEqual(
            _get_template_name("pyscf", "ccsd"), "pyscf_ccsd.py.j2"
        )

    def test_block2_dmrg(self):
        self.assertEqual(
            _get_template_name("block2", "dmrg"), "block2_dmrg.py.j2"
        )

    def test_unknown_code_method(self):
        """Unknown code/method should generate a plausible template name."""
        name = _get_template_name("mycode", "mymethod")
        self.assertEqual(name, "mycode_mymethod.j2")

    def test_case_insensitive(self):
        """Template lookup should be case-insensitive."""
        self.assertEqual(
            _get_template_name("PySCF", "UHF"), "pyscf_uhf.py.j2"
        )


class TestGetFileExtension(unittest.TestCase):
    """Test file extension mapping."""

    def test_pyscf(self):
        self.assertEqual(_get_file_extension("pyscf", "uhf"), ".py")

    def test_orca(self):
        self.assertEqual(_get_file_extension("orca", "bsdft"), ".inp")

    def test_gaussian(self):
        self.assertEqual(_get_file_extension("gaussian", "bsdft"), ".gjf")

    def test_molpro(self):
        self.assertEqual(_get_file_extension("molpro", "caspt2"), ".inp")

    def test_bagel(self):
        self.assertEqual(_get_file_extension("bagel", "caspt2"), ".json")

    def test_unknown(self):
        self.assertEqual(_get_file_extension("unknown", "test"), ".inp")


class TestGenerateInput(unittest.TestCase):
    """Test the unified generate_input interface."""

    @patch("apex_filter.input_generator._get_jinja_env")
    def test_pyscf_uhf_builtin(self, mock_jinja_env):
        """Should generate PySCF UHF input via built-in generator fallback."""
        import jinja2
        # Make Jinja2 raise TemplateNotFound to trigger builtin fallback
        mock_env = MagicMock()
        mock_env.get_template.side_effect = jinja2.TemplateNotFound("pyscf_uhf.py.j2")
        mock_jinja_env.return_value = mock_env

        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = generate_input(config, aspace, cluster, code="pyscf", method="uhf")
        self.assertIn("scf.UHF", output)
        self.assertIn("mf.kernel", output)

    @patch("apex_filter.input_generator._get_jinja_env")
    def test_generic_fallback(self, mock_jinja_env):
        """Unknown code/method should generate a generic comment."""
        import jinja2
        mock_env = MagicMock()
        mock_env.get_template.side_effect = jinja2.TemplateNotFound("unknown.j2")
        mock_jinja_env.return_value = mock_env

        config = _make_mock_config()
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        output = generate_input(config, aspace, cluster,
                                code="unknown_code", method="unknown_method")
        self.assertIn("Auto-generated", output)


class TestGenerateBatch(unittest.TestCase):
    """Test batch input generation."""

    @patch("apex_filter.input_generator._get_jinja_env")
    def test_batch_generates_multiple_files(self, mock_jinja_env):
        """Should generate one file per config."""
        import jinja2
        mock_env = MagicMock()
        mock_env.get_template.side_effect = jinja2.TemplateNotFound("pyscf_uhf.py.j2")
        mock_jinja_env.return_value = mock_env

        config1 = _make_mock_config()
        config2 = ElectronicConfig(
            spin_isomer=SpinIsomer(
                label="BS0-0", spin_assignment={0: +1, 1: +1},
                n_minority=0, family="BS0", Sz=5.0,
            ),
            minority_spin_sites=[],
            spin_assignment={0: +1, 1: +1},
            config_id=1, label="BS0-0|test",
        )
        cluster = _make_mock_cluster()
        aspace = _make_mock_active_space()

        results = generate_batch(
            [config1, config2], aspace, cluster,
            code="pyscf", method="uhf",
        )

        self.assertEqual(len(results), 2)
        for filename, content in results:
            self.assertIsInstance(filename, str)
            self.assertIsInstance(content, str)
            self.assertTrue(filename.endswith(".py"))


class TestGenerateBatchSubmission(unittest.TestCase):
    """Test batch submission script generation."""

    def test_slurm_script(self):
        """Should generate a SLURM script."""
        script = generate_batch_submission(
            ["calc1.py", "calc2.py"],
            scheduler="slurm",
            job_name="test_job",
        )
        self.assertIn("#SBATCH", script)
        self.assertIn("test_job", script)
        self.assertIn("python calc1.py", script)
        self.assertIn("python calc2.py", script)

    def test_pbs_script(self):
        """Should generate a PBS script."""
        script = generate_batch_submission(
            ["calc1.py"],
            scheduler="pbs",
            job_name="pbs_test",
        )
        self.assertIn("#PBS", script)
        self.assertIn("pbs_test", script)

    def test_simple_script(self):
        """Should generate a simple shell script for unknown scheduler."""
        script = generate_batch_submission(
            ["calc1.py"],
            scheduler="none",
        )
        self.assertIn("#!/bin/bash", script)

    def test_slurm_with_mem(self):
        """SLURM script should include memory if specified."""
        script = generate_batch_submission(
            ["calc1.py"],
            scheduler="slurm",
            mem="128GB",
        )
        self.assertIn("--mem=128GB", script)


class TestListAvailableTemplates(unittest.TestCase):
    """Test template listing."""

    def test_returns_list(self):
        """Should return a list (may be empty if no templates)."""
        templates = list_available_templates()
        self.assertIsInstance(templates, list)


if __name__ == "__main__":
    unittest.main()
