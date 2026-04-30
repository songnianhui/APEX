"""Regression tests for DMRG subprocess orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
from apex_filter.reference_dmrg import run_reference_dmrg


def test_run_reference_dmrg_invokes_worker_and_parses_json(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, capture_output, text, check):
        calls["cmd"] = cmd
        result_json = cmd[cmd.index("--result-json") + 1]
        with open(result_json, "w") as f:
            json.dump(
                {
                    "method": "DMRG",
                    "energy": -1.23,
                    "correlation_energy": -0.45,
                    "converged": True,
                    "s_squared": 0.75,
                    "uhf_energy": -0.78,
                    "backend": "pyblock2_sz",
                    "basis_mode": "step7_paired",
                    "bond_dim": 500,
                    "n_sweeps": 8,
                    "schedule_mode": "workflow",
                    "bond_dims": [50, 100, 200, 500],
                    "noises": [1e-4, 5e-5, 1e-5, 1e-6],
                    "thresholds": [1e-6, 1e-7, 1e-8, 1e-8],
                    "wall_time_s": 12.34,
                },
                f,
            )
        return SimpleNamespace(returncode=0, stdout="dmrg stdout", stderr="")

    monkeypatch.setattr("apex_filter.reference_dmrg.subprocess.run", fake_run)

    scratch = tmp_path / "scratch"
    result = run_reference_dmrg(
        object(),
        str(tmp_path / "ref_uhf.npz"),
        str(tmp_path / "basis.npz"),
        fcidump_path=str(tmp_path / "FCIDUMP.test"),
        backend="pyblock2_sz",
        basis_mode="step7_paired",
        bond_dim=500,
        n_sweeps=8,
        convergence_tol=1e-8,
        schedule_mode="workflow",
        n_threads=4,
        stack_mem=2 * 1024**3,
        scratch=str(scratch),
        log_path=str(tmp_path / "dmrg.log"),
    )

    assert result.energy == -1.23
    assert result.converged is True
    assert result.wall_time_s == 12.34
    assert result.log_path == str((tmp_path / "dmrg.log").resolve())
    assert (tmp_path / "dmrg.log").read_text() == "dmrg stdout\n"
    assert "--fcidump-path" in calls["cmd"]
    assert "--dmrg-basis-npz-path" in calls["cmd"]
    assert "--backend" in calls["cmd"]
    assert "--basis-mode" in calls["cmd"]
    assert "--schedule-mode" in calls["cmd"]
    assert "workflow" in calls["cmd"]
    assert "apex_filter.reference_dmrg_worker" in calls["cmd"]


def test_run_reference_dmrg_rebuilds_empty_schedule_lists(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, check):
        result_json = cmd[cmd.index("--result-json") + 1]
        with open(result_json, "w") as f:
            json.dump(
                {
                    "method": "DMRG",
                    "energy": -1.23,
                    "correlation_energy": -0.45,
                    "converged": True,
                    "s_squared": None,
                    "uhf_energy": -0.78,
                    "backend": "pyscf_dmrgci_sz",
                    "basis_mode": "original_identity",
                    "bond_dim": 2000,
                    "n_sweeps": 30,
                    "schedule_mode": "benchmark",
                    "bond_dims": [],
                    "noises": [],
                    "thresholds": [],
                    "wall_time_s": 12.34,
                },
                f,
            )
        return SimpleNamespace(returncode=0, stdout="dmrg stdout", stderr="")

    monkeypatch.setattr("apex_filter.reference_dmrg.subprocess.run", fake_run)

    result = run_reference_dmrg(
        object(),
        str(tmp_path / "ref_uhf.npz"),
        str(tmp_path / "basis.npz"),
        fcidump_path=str(tmp_path / "FCIDUMP.test"),
        backend="pyscf_dmrgci_sz",
        basis_mode="original_identity",
        bond_dim=2000,
        n_sweeps=30,
        convergence_tol=1e-8,
        schedule_mode="benchmark",
        n_threads=4,
        stack_mem=2 * 1024**3,
        scratch=str(tmp_path / "scratch"),
        log_path=str(tmp_path / "dmrg.log"),
    )

    assert len(result.bond_dims) == 30
    assert len(result.noises) == 30
    assert len(result.thresholds) == 30
    assert result.bond_dims[:10] == [200, 200, 200, 200, 400, 400, 400, 400, 2000, 2000]
    assert result.noises[-12:] == [0.0] * 12
    assert result.thresholds[-12:] == [1e-9] * 12


def test_save_reference_dmrg_result_writes_h5(tmp_path):
    from apex_filter.reference_dmrg import ReferenceDMRGResult, save_reference_dmrg_result

    out_npz = tmp_path / "toy_dmrg.npz"
    result = ReferenceDMRGResult(
        method="DMRG",
        energy=-1.23,
        correlation_energy=-0.45,
        converged=True,
        s_squared=None,
        uhf_energy=-0.78,
        backend="pyscf_dmrgci_sz",
        basis_mode="original_identity",
        bond_dim=500,
        n_sweeps=30,
        schedule_mode="benchmark",
        bond_dims=[200, 400, 500],
        noises=[1e-4, 1e-5, 0.0],
        thresholds=[1e-4, 1e-6, 1e-9],
        wall_time_s=12.34,
        log_path=str(tmp_path / "toy_dmrg.log"),
        reference_state_path=str(tmp_path / "ref_uhf.npz"),
        basis_state_path=str(tmp_path / "basis.npz"),
        scratch_dir=str(tmp_path / "scratch"),
    )
    save_reference_dmrg_result(result, str(out_npz))
    out_h5 = out_npz.with_suffix(".h5")
    assert out_h5.exists()
    with h5py.File(out_h5, "r") as f:
        assert "metadata" in f
        assert "schedule" in f
        assert "dmrg_diagnostics" in f
        assert f["metadata"].attrs["artifact_type"] == "apex_filter_step8_dmrg_state"


def test_run_reference_dmrg_recovers_last_energy_from_worker_output(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, check):
        result_json = cmd[cmd.index("--result-json") + 1]
        open(result_json, "w").close()
        return SimpleNamespace(
            returncode=1,
            stdout=(
                "Sweep =    0 | Direction =  forward | Bond dimension =   50 | Noise =  1.00e-04 | Dav threshold =  1.00e-04\n"
                "Time elapsed =      0.646 | E =   -3338.0789296230 | DW = 3.11722e-05\n"
                "Sweep =    1 | Direction = backward | Bond dimension =   50 | Noise =  1.00e-04 | Dav threshold =  1.00e-04\n"
                "Time elapsed =      1.042 | E =   -3338.2036268292 | DE = -1.25e-01 | DW = 3.90233e-04\n"
                "ATTENTION: DMRG is not converged to desired tolerance of 1.00000e-06\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("apex_filter.reference_dmrg.subprocess.run", fake_run)

    uhf_npz = tmp_path / "ref_uhf.npz"
    import numpy as np
    np.savez(uhf_npz, energy=-3338.1)

    result = run_reference_dmrg(
        object(),
        str(uhf_npz),
        str(tmp_path / "basis.npz"),
        fcidump_path=str(tmp_path / "FCIDUMP.test"),
        backend="pyblock2_sz",
        basis_mode="step7_paired",
        bond_dim=500,
        n_sweeps=8,
        convergence_tol=1e-8,
        schedule_mode="workflow",
        n_threads=4,
        stack_mem=2 * 1024**3,
        scratch=str(tmp_path / "scratch"),
        log_path=str(tmp_path / "dmrg_last.log"),
    )

    assert result.energy == -3338.2036268292
    assert result.converged is False
    assert result.backend == "pyblock2_sz"
    assert result.basis_mode == "step7_paired"
    assert abs(result.correlation_energy - (-0.10362682919972552)) < 1e-12
    assert (tmp_path / "dmrg_last.log").read_text().startswith("Sweep =")
