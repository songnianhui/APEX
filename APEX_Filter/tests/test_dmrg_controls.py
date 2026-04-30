"""Unit tests for shared DMRG schedule controls."""

from shared.dmrg_controls import build_dmrg_sweep_schedule, compress_dmrg_schedule_for_dmrgci


def test_workflow_schedule_matches_legacy_shape():
    bond_dims, noises, thresholds = build_dmrg_sweep_schedule(
        mode="workflow",
        bond_dim=500,
        convergence_tol=1e-8,
        n_sweeps=8,
    )
    assert bond_dims == [50, 50, 100, 100, 200, 200, 500, 500]
    assert noises == [1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5, 1e-6, 1e-6]
    assert thresholds == [1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8]


def test_benchmark_schedule_matches_dmrgci_default_pattern():
    bond_dims, noises, thresholds = build_dmrg_sweep_schedule(
        mode="benchmark",
        bond_dim=500,
        convergence_tol=1e-8,
        n_sweeps=30,
    )
    assert len(bond_dims) == 30
    assert bond_dims[:10] == [200, 200, 200, 200, 400, 400, 400, 400, 500, 500]
    assert noises[:10] == [1e-4] * 10
    assert thresholds[:10] == [1e-4] * 10
    assert thresholds[10:12] == [1e-5, 1e-5]
    assert thresholds[12:14] == [1e-6, 1e-6]
    assert thresholds[14:16] == [1e-7, 1e-7]
    assert thresholds[16:18] == [1e-8, 1e-8]
    assert thresholds[18:] == [1e-9] * 12
    assert noises[18:] == [0.0] * 12


def test_compress_dmrg_schedule_for_dmrgci():
    bond_dims, noises, thresholds = build_dmrg_sweep_schedule(
        mode="benchmark",
        bond_dim=500,
        convergence_tol=1e-8,
        n_sweeps=30,
    )
    sweeps, max_ms, tols, sched_noises = compress_dmrg_schedule_for_dmrgci(
        bond_dims, noises, thresholds
    )
    assert sweeps == [0, 4, 8, 10, 12, 14, 16, 18]
    assert max_ms == [200, 400, 500, 500, 500, 500, 500, 500]
    assert tols == [1e-4, 1e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    assert sched_noises == [1e-4, 1e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
