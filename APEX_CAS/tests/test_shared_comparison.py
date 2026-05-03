"""Direct regression coverage for shared comparison helpers."""

from __future__ import annotations

import json

import numpy as np
from pyscf.tools import fcidump as fcidump_mod

from shared.comparison import (
    compare_artifacts,
    compare_basis_states,
    compare_density_matrices,
    compare_fcidumps,
    compare_energy_triplet,
    compare_matrix_entries,
    compare_matrix_spectra,
    compare_two_particle_density_tensors,
    compare_two_sz_with_benchmark,
    find_chan_benchmark_row,
)


def test_find_chan_benchmark_row_returns_matching_theory(tmp_path):
    benchmark = tmp_path / "chan.json"
    benchmark.write_text(
        json.dumps(
            {
                "table5_ucc_series": [
                    {"theory": "UHF", "two_sz_fe1": 3.2, "two_sz_fe2": -3.2},
                    {"theory": "CCSD", "two_sz_fe1": 3.1, "two_sz_fe2": -3.1},
                ]
            }
        )
    )

    row = find_chan_benchmark_row(str(benchmark), "CCSD")

    assert row is not None
    assert row["theory"] == "CCSD"
    assert row["two_sz_fe1"] == 3.1


def test_compare_two_sz_with_benchmark_prefers_global_sign_flip_when_closer(tmp_path):
    benchmark = tmp_path / "chan.json"
    benchmark.write_text(
        json.dumps(
            {
                "table5_ucc_series": [
                    {"theory": "UHF", "two_sz_fe1": -3.2, "two_sz_fe2": 3.2},
                ]
            }
        )
    )

    result = compare_two_sz_with_benchmark(
        {"Fe1": 3.19, "Fe2": -3.21},
        str(benchmark),
        theory="UHF",
    )

    assert result["reference"] == {"Fe1": -3.2, "Fe2": 3.2}
    assert result["best_alignment"] == "global_sign_flip"
    assert abs(result["best_delta"]["Fe1"] - 0.01) < 1e-12
    assert abs(result["best_delta"]["Fe2"] - 0.01) < 1e-12


def test_compare_fcidumps_reports_match_for_identical_integrals(tmp_path):
    h1 = np.array([[1.0, 0.1], [0.1, 0.8]])
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = 0.5
    h2[1, 1, 1, 1] = 0.4
    h2[0, 1, 0, 1] = 0.1
    h2[1, 0, 1, 0] = 0.1

    ref_path = tmp_path / "ref.FCIDUMP"
    new_path = tmp_path / "new.FCIDUMP"
    fcidump_mod.from_integrals(str(ref_path), h1, h2, 2, 2, nuc=-10.0, ms=0)
    fcidump_mod.from_integrals(str(new_path), h1, h2, 2, 2, nuc=-10.0, ms=0)

    result = compare_fcidumps(str(ref_path), str(new_path))

    assert result["match"] is True
    assert result["n_eigval_mismatch"] == 0
    assert abs(result["eigval_frobenius"]) < 1e-12
    assert abs(result["ecore_diff"]) < 1e-12


def test_compare_fcidumps_detects_h1_and_ecore_differences(tmp_path):
    h1_ref = np.array([[1.0, 0.0], [0.0, 0.8]])
    h1_new = np.array([[1.3, 0.0], [0.0, 0.8]])
    h2 = np.zeros((2, 2, 2, 2))

    ref_path = tmp_path / "ref.FCIDUMP"
    new_path = tmp_path / "new.FCIDUMP"
    fcidump_mod.from_integrals(str(ref_path), h1_ref, h2, 2, 2, nuc=-10.0, ms=0)
    fcidump_mod.from_integrals(str(new_path), h1_new, h2, 2, 2, nuc=-9.0, ms=0)

    result = compare_fcidumps(str(ref_path), str(new_path), eigval_tol=1e-6, ecore_tol=1e-6)

    assert result["match"] is False
    assert result["n_eigval_mismatch"] >= 1
    assert abs(result["ecore_diff"] - 1.0) < 1e-12


def test_compare_energy_triplet_reports_componentwise_deltas():
    result = compare_energy_triplet(
        computed_active=-1.5,
        computed_core=-10.0,
        computed_total=-11.5,
        reference_active=-1.6,
        reference_core=-9.8,
        reference_total=-11.4,
    )

    assert result["computed"]["E_total"] == -11.5
    assert result["reference"]["E_core"] == -9.8
    assert abs(result["delta"]["E_active"] - 0.1) < 1e-12
    assert abs(result["delta"]["E_core"] + 0.2) < 1e-12
    assert abs(result["delta"]["E_total"] + 0.1) < 1e-12


def test_compare_matrix_entries_reports_raw_matrix_distance():
    ref = np.array([[1.0, 0.2], [0.2, 0.8]])
    new = np.array([[1.1, 0.1], [0.1, 0.8]])

    result = compare_matrix_entries(ref, new)

    assert result["shape"] == (2, 2)
    assert abs(result["trace_diff"] - 0.1) < 1e-12
    assert abs(result["max_abs"] - 0.1) < 1e-12
    assert result["frobenius"] > 0.0


def test_compare_matrix_spectra_is_invariant_under_similarity_rotation():
    ref = np.diag([2.0, 1.0])
    rot = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    new = rot @ ref @ rot.T

    result = compare_matrix_spectra(ref, new, hermitian=True, descending=True)

    assert result["shape"] == (2, 2)
    assert result["eigenvalue_max_abs"] < 1e-12
    assert result["eigenvalue_frobenius"] < 1e-12
    assert np.allclose(result["eigenvalues_ref"], [2.0, 1.0])
    assert np.allclose(result["eigenvalues_new"], [2.0, 1.0])


def test_compare_density_matrices_flags_basis_rotation_like_difference():
    ref = np.diag([1.9, 0.1])
    rot = np.array(
        [
            [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
            [np.sin(np.pi / 6), np.cos(np.pi / 6)],
        ]
    )
    new = rot @ ref @ rot.T

    result = compare_density_matrices(ref, new)

    assert result["elementwise"]["max_abs"] > 1e-2
    assert result["spectrum"]["eigenvalue_max_abs"] < 1e-12
    assert result["basis_rotation_likely"] is True
    assert abs(result["trace_diff"]) < 1e-12
    assert abs(result["trace_square_diff"]) < 1e-12


def test_compare_basis_states_detects_same_subspace_with_reordered_representation():
    ref_alpha = np.eye(3)
    ref_beta = np.eye(3)
    rot = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    new_alpha = ref_alpha @ rot
    new_beta = ref_beta @ rot

    result = compare_basis_states(
        {
            "active_coeff_alpha": ref_alpha,
            "active_coeff_beta": ref_beta,
            "alpha_no_occupations": np.array([1.9, 1.0, 0.1]),
            "beta_no_occupations": np.array([1.8, 1.1, 0.1]),
            "ordering": np.array([0, 1, 2]),
            "pairs": np.array([[0, 0], [1, 1], [2, 2]]),
        },
        {
            "active_coeff_alpha": new_alpha,
            "active_coeff_beta": new_beta,
            "alpha_no_occupations": np.array([1.9, 1.0, 0.1]),
            "beta_no_occupations": np.array([1.8, 1.1, 0.1]),
            "ordering": np.array([1, 0, 2]),
            "pairs": np.array([[1, 1], [0, 0], [2, 2]]),
        },
    )

    assert result["same_active_subspace_likely"] is True
    assert result["representation_difference_likely"] is True
    assert result["alpha_subspace"]["subspace_sigma_min"] > 0.99
    assert result["beta_subspace"]["subspace_sigma_min"] > 0.99
    assert result["ordering"]["exact_equal"] is False
    assert result["pairs"]["exact_equal"] is False


def test_compare_two_particle_density_tensors_flags_pair_space_rotation_like_difference():
    ref = np.zeros((2, 2, 2, 2))
    ref[0, 0, 0, 0] = 1.2
    ref[1, 1, 1, 1] = 0.8
    ref[0, 1, 0, 1] = 0.2
    ref[1, 0, 1, 0] = 0.2
    ref[0, 1, 1, 0] = 0.1
    ref[1, 0, 0, 1] = 0.1

    rot = np.array(
        [
            [np.cos(np.pi / 5), -np.sin(np.pi / 5)],
            [np.sin(np.pi / 5), np.cos(np.pi / 5)],
        ]
    )
    new = np.einsum("ap,bq,cr,ds,pqrs->abcd", rot, rot, rot, rot, ref)

    result = compare_two_particle_density_tensors(ref, new)

    assert result["elementwise"]["max_abs"] > 1e-2
    assert result["pair_matrix"]["spectrum"]["eigenvalue_max_abs"] < 1e-10
    assert result["basis_rotation_likely"] is True
    assert abs(result["diag_trace_diff"]) < 1e-10


def test_compare_two_particle_density_tensors_supports_rank5_block_layout():
    block = np.zeros((2, 2, 2, 2))
    block[0, 0, 0, 0] = 1.0
    block[1, 1, 1, 1] = 0.8
    block[0, 1, 0, 1] = 0.2
    blocks_ref = np.stack([block, block * 0.5, block * 0.25], axis=0)

    rot = np.array(
        [
            [np.cos(np.pi / 7), -np.sin(np.pi / 7)],
            [np.sin(np.pi / 7), np.cos(np.pi / 7)],
        ]
    )
    blocks_new = np.stack(
        [
            np.einsum("ap,bq,cr,ds,pqrs->abcd", rot, rot, rot, rot, blocks_ref[idx])
            for idx in range(blocks_ref.shape[0])
        ],
        axis=0,
    )

    result = compare_two_particle_density_tensors(blocks_ref, blocks_new)

    assert result["shape"] == (3, 2, 2, 2, 2)
    assert result["n_components"] == 3
    assert result["basis_rotation_likely"] is True
    assert all(
        component["pair_matrix"]["spectrum"]["eigenvalue_max_abs"] < 1e-10
        for component in result["components"]
    )


def test_compare_artifacts_infers_fcidump_kind(tmp_path):
    h1 = np.array([[1.0, 0.1], [0.1, 0.8]])
    h2 = np.zeros((2, 2, 2, 2))
    ref_path = tmp_path / "ref.FCIDUMP"
    new_path = tmp_path / "new.FCIDUMP"
    fcidump_mod.from_integrals(str(ref_path), h1, h2, 2, 2, nuc=-10.0, ms=0)
    fcidump_mod.from_integrals(str(new_path), h1, h2, 2, 2, nuc=-10.0, ms=0)

    result = compare_artifacts(str(ref_path), str(new_path))

    assert result["kind"] == "fcidump"
    assert result["match"] is True


def test_compare_artifacts_supports_json_payloads(tmp_path):
    ref = tmp_path / "ref.json"
    new = tmp_path / "new.json"
    ref.write_text(json.dumps({"energy": -1.0, "flags": {"converged": True}}))
    new.write_text(json.dumps({"energy": -1.1, "flags": {"converged": True}}))

    result = compare_artifacts(str(ref), str(new))

    assert result["kind"] == "json"
    assert result["paths_ref"] == result["paths_new"]
    assert abs(result["numeric_summary"]["max_abs"] - 0.1) < 1e-12
