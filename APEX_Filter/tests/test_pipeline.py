"""Tests for pipeline module and parse_npz_result.

Run with: pytest tests/test_pipeline.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from apex_filter.models import (
    CAS,
    CalculationResult,
    ClusterInfo,
    ElectronicConfig,
    FilteringLevel,
    FilteringPlan,
    MetalCenter,
    OxidationAssignment,
    SpinIsomer,
)
from apex_filter.result_parser import parse_npz_result, to_calculation_result
from apex_filter.filtering import (
    select_from_uhf,
    select_from_ccsd,
    select_lowest_energy,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

def _make_config(label, family="BS1", config_id=0):
    """Create a minimal ElectronicConfig for testing."""
    iso = SpinIsomer(label=f"{family}-1", family=family,
                     spin_assignment={0: 1}, n_minority=0)
    return ElectronicConfig(
        spin_isomer=iso,
        oxidation=OxidationAssignment(assignments={0: 3}),
        config_id=config_id,
        label=label,
        spin_assignment={0: 1},
        minority_spin_sites=[],
    )


def _make_cluster():
    """Create a minimal ClusterInfo for testing."""
    metals = [
        MetalCenter(element="Fe", index=0,
                    position=np.array([0.0, 0.0, 0.0]), label="Fe1"),
        MetalCenter(element="Fe", index=1,
                    position=np.array([2.3, 0.0, 0.0]), label="Fe2"),
    ]
    return ClusterInfo(
        metals=metals,
        all_elements=["Fe", "Fe"],
        all_positions=np.array([[0, 0, 0], [2.3, 0, 0]]),
        formula="Fe2",
        total_charge=0,
        target_spin=0.0,
    )


def _make_active_space():
    return CAS(n_electrons=10, n_orbitals=8)


# ──────────────────────────────────────────────────────────────────
# Tests for parse_npz_result
# ──────────────────────────────────────────────────────────────────

class TestParseNpzResult:
    """Tests for result_parser.parse_npz_result."""

    def test_uhf_npz(self, tmp_path):
        """Parse a UHF-style .npz file."""
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
        """Parse a CCSD-style .npz file."""
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
        """Parse a CCSD(T)-style .npz file with (T) correction."""
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
        assert result["energy_t"] == pytest.approx(-102.35)
        assert result["e_t_correction"] == pytest.approx(-0.05)

    def test_missing_file(self):
        """Raise FileNotFoundError for missing .npz."""
        with pytest.raises(FileNotFoundError):
            parse_npz_result("/nonexistent/path.npz")

    def test_empty_energy(self, tmp_path):
        """Parse an .npz with no recognized keys -- defaults to UHF."""
        npz_path = os.path.join(tmp_path, "test_empty.npz")
        np.savez(npz_path, some_data=np.zeros(3))
        result = parse_npz_result(npz_path)
        assert result["method"] == "UHF"
        assert result["energy"] == 0.0


# ──────────────────────────────────────────────────────────────────
# Tests for to_calculation_result with npz data
# ──────────────────────────────────────────────────────────────────

class TestToCalculationResult:
    """Tests converting parsed npz results to CalculationResult."""

    def test_uhf_to_result(self, tmp_path):
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


# ──────────────────────────────────────────────────────────────────
# Tests for filtering selection functions
# ──────────────────────────────────────────────────────────────────

class TestSelectionFunctions:
    """Verify select_from_uhf/ccsd/lowest_energy work correctly."""

    def _make_result(self, label, energy, converged=True, family="BS1"):
        cfg = _make_config(label, family=family)
        return CalculationResult(
            config=cfg, method="UHF", energy=energy,
            converged=converged,
        )

    def test_select_from_uhf_keeps_per_family(self):
        results = [
            self._make_result("c1", -100.0, family="BS1"),
            self._make_result("c2", -99.0, family="BS1"),
            self._make_result("c3", -98.0, family="BS2"),
            self._make_result("c4", -97.0, family="BS2"),
        ]
        selected = select_from_uhf(results, n_per_isomer=1)
        assert len(selected) == 2
        assert selected[0].energy == -100.0
        assert selected[1].energy == -98.0

    def test_select_from_uhf_total_cap(self):
        results = [
            self._make_result(f"c{i}", -100.0 + i, family="BS1")
            for i in range(10)
        ]
        selected = select_from_uhf(results, n_per_isomer=10, n_total=3)
        assert len(selected) == 3

    def test_select_lowest_energy(self):
        results = [
            self._make_result("c1", -95.0),
            self._make_result("c2", -100.0),
            self._make_result("c3", -98.0),
            self._make_result("c4", -97.0, converged=False),
        ]
        selected = select_lowest_energy(results, n_keep=2)
        assert len(selected) == 2
        assert selected[0].energy == -100.0
        assert selected[1].energy == -98.0

    def test_select_skips_unconverged(self):
        results = [
            self._make_result("c1", -100.0, converged=True),
            self._make_result("c2", -105.0, converged=False),
        ]
        selected = select_lowest_energy(results, n_keep=2)
        assert len(selected) == 1
        assert selected[0].energy == -100.0


# ──────────────────────────────────────────────────────────────────
# Tests for pipeline.run_pipeline with mocked execution
# ──────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """End-to-end tests with mocked subprocess execution."""

    def test_pipeline_with_mock_results(self, tmp_path, monkeypatch):
        """Run pipeline through two levels with pre-seeded .npz files."""
        from apex_filter.pipeline import run_pipeline

        configs = [
            _make_config(f"cfg_{i}", family="BS1", config_id=i)
            for i in range(6)
        ]
        cluster = _make_cluster()
        active_space = _make_active_space()
        plan = FilteringPlan(
            levels=[
                FilteringLevel(
                    method="UHF", n_input=6, n_output=3,
                    selection_criterion="energy", n_per_isomer=3,
                ),
            ],
            total_configs=6,
            active_space=active_space,
        )

        # Instead of executing, mock _execute_scripts and _parse_results
        # so that we inject fake .npz results.
        call_count = {"n": 0}

        def fake_execute(scripts, workdir):
            # Write fake .npz files for each script
            for script in scripts:
                # Extract label from script name (format: label_uhf.py)
                label = script.replace("_uhf.py", "")
                npz_path = os.path.join(workdir, f"{label}_uhf.npz")
                energy = -100.0 - call_count["n"] * 0.1
                np.savez(
                    npz_path,
                    energy=energy,
                    converged=True,
                    spin_sq=4.0,
                )
                call_count["n"] += 1
            return [(s, True, None) for s in scripts]

        import apex_filter.pipeline as pl
        monkeypatch.setattr(pl, "_execute_scripts", fake_execute)

        results = run_pipeline(
            configs, active_space, cluster, plan,
            workdir=str(tmp_path / "pipeline"),
            n_final=3,
        )

        assert len(results) <= 3
        assert all(isinstance(r, CalculationResult) for r in results)
        # Verify the summary file was written
        summary_path = os.path.join(
            str(tmp_path / "pipeline"), "pipeline_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as fh:
            summary = json.load(fh)
        assert len(summary) <= 3

    def test_pipeline_multi_level(self, tmp_path, monkeypatch):
        """Two-level pipeline: UHF -> CCSD with mocked execution."""
        from apex_filter.pipeline import run_pipeline

        configs = [
            _make_config(f"cfg_{i}", family="BS1", config_id=i)
            for i in range(4)
        ]
        cluster = _make_cluster()
        active_space = _make_active_space()
        plan = FilteringPlan(
            levels=[
                FilteringLevel(
                    method="UHF", n_input=4, n_output=2,
                    selection_criterion="energy", n_per_isomer=2,
                ),
                FilteringLevel(
                    method="UCCSD", n_input=2, n_output=1,
                    selection_criterion="energy", n_per_isomer=1,
                ),
            ],
            total_configs=4,
            active_space=active_space,
        )

        call_log = []

        def fake_execute(scripts, workdir):
            is_uhf = any("_uhf.py" in s for s in scripts)
            is_ccsd = any("_ccsd.py" in s for s in scripts)

            for script in scripts:
                if is_uhf:
                    label = script.replace("_uhf.py", "")
                    npz_path = os.path.join(workdir, f"{label}_uhf.npz")
                    idx = int(label.split("_")[-1])
                    np.savez(npz_path, energy=-100.0 - idx * 0.5,
                             converged=True, spin_sq=4.0)
                    call_log.append(("UHF", label))
                elif is_ccsd:
                    label = script.replace("_ccsd.py", "")
                    npz_path = os.path.join(workdir, f"{label}_ccsd_results.npz")
                    idx = int(label.split("_")[-1])
                    np.savez(
                        npz_path,
                        ccsd_total=-102.0 - idx * 0.3,
                        ccsd_corr=-2.0,
                        ccsd_converged=True,
                        spin_sq=4.0,
                    )
                    call_log.append(("CCSD", label))
            return [(s, True, None) for s in scripts]

        import apex_filter.pipeline as pl
        monkeypatch.setattr(pl, "_execute_scripts", fake_execute)

        results = run_pipeline(
            configs, active_space, cluster, plan,
            workdir=str(tmp_path / "pipeline2"),
            n_final=1,
        )

        # Should have called UHF and CCSD
        uhf_calls = [c for c in call_log if c[0] == "UHF"]
        ccsd_calls = [c for c in call_log if c[0] == "CCSD"]
        assert len(uhf_calls) > 0
        assert len(ccsd_calls) > 0

        # Final result is the best CCSD
        assert len(results) == 1
        assert results[0].method == "UCCSD"

    def test_pipeline_no_results_stops(self, tmp_path, monkeypatch):
        """Pipeline stops gracefully when no results are obtained."""
        from apex_filter.pipeline import run_pipeline

        configs = [_make_config("cfg_0")]
        cluster = _make_cluster()
        active_space = _make_active_space()
        plan = FilteringPlan(
            levels=[
                FilteringLevel(
                    method="UHF", n_input=1, n_output=1,
                    selection_criterion="energy", n_per_isomer=1,
                ),
            ],
            total_configs=1,
            active_space=active_space,
        )

        def fake_execute_no_output(scripts, workdir):
            # Don't write any .npz files
            return [(s, False, None) for s in scripts]

        import apex_filter.pipeline as pl
        monkeypatch.setattr(pl, "_execute_scripts", fake_execute_no_output)

        results = run_pipeline(
            configs, active_space, cluster, plan,
            workdir=str(tmp_path / "pipeline_empty"),
            n_final=5,
        )

        # No results should be returned
        assert len(results) == 0
