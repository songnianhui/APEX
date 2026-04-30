"""Tests for result_parser module.

Test parsing of generated `.npz` result files and conversion to
`CalculationResult`.
"""

import os

import numpy as np
import pytest

from apex_filter.models import CalculationResult, ElectronicConfig, OxidationAssignment, SpinIsomer
from apex_filter.result_parser import parse_npz_result, to_calculation_result


def _make_config(label, family="BS1", config_id=0):
    """Create a minimal ElectronicConfig for testing."""
    isomer = SpinIsomer(
        label=f"{family}-1",
        family=family,
        spin_assignment={0: 1},
        n_minority=0,
    )
    return ElectronicConfig(
        spin_isomer=isomer,
        oxidation=OxidationAssignment(assignments={0: 3}),
        config_id=config_id,
        label=label,
        spin_assignment={0: 1},
        minority_spin_sites=[],
    )


class TestParseNpzResult:
    """Tests for `parse_npz_result`."""

    def test_uhf_npz(self, tmp_path):
        """Parse a UHF-style `.npz` file."""
        npz_path = os.path.join(tmp_path, "test_uhf.npz")
        np.savez(
            npz_path,
            energy=-100.5,
            converged=True,
            spin_sq=6.0,
            mo_coeff_a=np.eye(3),
            mo_coeff_b=np.eye(3),
        )
        result = parse_npz_result(npz_path)
        assert result["method"] == "UHF"
        assert result["energy"] == pytest.approx(-100.5)
        assert result["converged"] is True
        assert result["s_squared"] == pytest.approx(6.0)

    def test_ccsd_npz(self, tmp_path):
        """Parse a CCSD-style `.npz` file."""
        npz_path = os.path.join(tmp_path, "test_ccsd.npz")
        np.savez(
            npz_path,
            ccsd_total=-102.3,
            ccsd_corr=-1.8,
            ccsd_converged=True,
            spin_sq=6.0,
        )
        result = parse_npz_result(npz_path)
        assert result["method"] == "UCCSD"
        assert result["energy"] == pytest.approx(-102.3)
        assert result["correlation_energy"] == pytest.approx(-1.8)
        assert result["converged"] is True

    def test_ccsdt_npz(self, tmp_path):
        """Parse a CCSD(T)-style `.npz` file with triples correction."""
        npz_path = os.path.join(tmp_path, "test_ccsdt.npz")
        np.savez(
            npz_path,
            ccsd_total=-102.3,
            ccsd_corr=-1.8,
            ccsd_converged=True,
            et_correction=-0.05,
            ccsd_t_total=-102.35,
        )
        result = parse_npz_result(npz_path)
        assert result["method"] == "UCCSD(T)"
        assert result["energy"] == pytest.approx(-102.35)
        assert result["energy_ccsd"] == pytest.approx(-102.3)
        assert result["energy_t"] == pytest.approx(-102.35)
        assert result["e_t_correction"] == pytest.approx(-0.05)

    def test_uccsdt_npz(self, tmp_path):
        """Parse a true CCSDT-shaped `.npz` payload."""
        npz_path = os.path.join(tmp_path, "test_uccsdt.npz")
        np.savez(
            npz_path,
            ccsdt_total=-102.41,
            ccsdt_corr=-1.91,
            ccsdt_converged=True,
            t1_norm=0.031,
            spin_sq=3.75,
        )
        result = parse_npz_result(npz_path)
        assert result["method"] == "UCCSDT"
        assert result["energy"] == pytest.approx(-102.41)
        assert result["correlation_energy"] == pytest.approx(-1.91)
        assert result["converged"] is True
        assert result["t1_norm"] == pytest.approx(0.031)
        assert result["s_squared"] == pytest.approx(3.75)

    def test_hast_scaffold_npz_stays_distinct(self, tmp_path):
        """Legacy/experimental HAST payloads should not masquerade as CCSDTQ."""
        npz_path = os.path.join(tmp_path, "test_hast.npz")
        np.savez(
            npz_path,
            hast_total=-102.36,
            hast_corr=-1.86,
            hast_converged=True,
            hast_method_level="ccsdt_scaffold",
        )
        result = parse_npz_result(npz_path)
        assert result["method"] == "HAST-UCC"
        assert result["energy"] == pytest.approx(-102.36)
        assert result["correlation_energy"] == pytest.approx(-1.86)
        assert result["converged"] is True
        assert result["nominal_method"] == "ccsdt_scaffold"

    def test_missing_file(self):
        """Raise `FileNotFoundError` for a missing `.npz`."""
        with pytest.raises(FileNotFoundError):
            parse_npz_result("/nonexistent/path.npz")

    def test_empty_energy_defaults_to_uhf(self, tmp_path):
        """Unrecognized payload should fall back to a UHF-shaped result."""
        npz_path = os.path.join(tmp_path, "test_empty.npz")
        np.savez(npz_path, some_data=np.zeros(3))
        result = parse_npz_result(npz_path)
        assert result["method"] == "UHF"
        assert result["energy"] == 0.0


class TestToCalculationResult:
    """Tests converting parsed dicts to `CalculationResult`."""

    def test_uhf_to_result(self, tmp_path):
        """Parsed UHF data should become a typed result object."""
        npz_path = os.path.join(tmp_path, "cfg1_uhf.npz")
        np.savez(npz_path, energy=-99.0, converged=True, spin_sq=4.0)
        parsed = parse_npz_result(npz_path)
        cfg = _make_config("cfg1")
        result = to_calculation_result(parsed, config=cfg)
        assert isinstance(result, CalculationResult)
        assert result.method == "UHF"
        assert result.energy == pytest.approx(-99.0)
        assert result.converged is True
        assert result.config is cfg
