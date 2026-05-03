"""Shared comparison authority for benchmark and regression validation.

Workflow-facing modules may keep thin aliases where benchmark tooling expects
historical import paths, but the comparison primitives themselves should live
here so that benchmark checks, artifact comparisons, and future regression
tooling share one implementation.
"""

from __future__ import annotations

import json
from pathlib import Path as _Path

import h5py
import numpy as np
from pyscf.tools import fcidump as fcidump_mod
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment


def compare_matrix_entries(matrix_ref: np.ndarray, matrix_new: np.ndarray) -> dict:
    """Compare two same-shaped matrices at the raw element level."""
    ref = np.asarray(matrix_ref, dtype=float)
    new = np.asarray(matrix_new, dtype=float)
    if ref.shape != new.shape:
        raise ValueError(
            f"Matrix shapes differ: reference {ref.shape}, new {new.shape}"
        )

    delta = new - ref
    return {
        "shape": ref.shape,
        "frobenius": float(np.linalg.norm(delta)),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.max(np.abs(delta))),
        "trace_ref": float(np.trace(ref)),
        "trace_new": float(np.trace(new)),
        "trace_diff": float(np.trace(new) - np.trace(ref)),
    }


def compare_matrix_spectra(
    matrix_ref: np.ndarray,
    matrix_new: np.ndarray,
    *,
    hermitian: bool = True,
    descending: bool = False,
) -> dict:
    """Compare matrices through their eigenspectra.

    This is useful when elementwise differences may be dominated by basis
    rotations but invariant spectral content is the physically relevant signal.
    """
    ref = np.asarray(matrix_ref, dtype=float)
    new = np.asarray(matrix_new, dtype=float)
    if ref.shape != new.shape:
        raise ValueError(
            f"Matrix shapes differ: reference {ref.shape}, new {new.shape}"
        )
    if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError(f"Expected square matrices, got {ref.shape}")

    eig_fn = np.linalg.eigvalsh if hermitian else np.linalg.eigvals
    eig_ref = np.asarray(eig_fn(ref), dtype=float)
    eig_new = np.asarray(eig_fn(new), dtype=float)
    sort_order = -1 if descending else 1
    eig_ref_sorted = np.sort(eig_ref)[::sort_order]
    eig_new_sorted = np.sort(eig_new)[::sort_order]
    eig_delta = eig_new_sorted - eig_ref_sorted

    return {
        "shape": ref.shape,
        "hermitian": bool(hermitian),
        "eigenvalues_ref": eig_ref_sorted,
        "eigenvalues_new": eig_new_sorted,
        "eigenvalue_diff": eig_delta,
        "eigenvalue_frobenius": float(np.linalg.norm(eig_delta)),
        "eigenvalue_rms": float(np.sqrt(np.mean(eig_delta**2))),
        "eigenvalue_max_abs": float(np.max(np.abs(eig_delta))),
    }


def compare_density_matrices(dm_ref: np.ndarray, dm_new: np.ndarray) -> dict:
    """Compare one-particle density matrices with basis-robust diagnostics.

    For density matrices, the raw matrix elements can change substantially under
    orbital rotations even when physically relevant quantities are nearly
    unchanged. This helper reports both the raw elementwise difference and the
    more intrinsic spectrum/invariant differences.
    """
    ref = np.asarray(dm_ref, dtype=float)
    new = np.asarray(dm_new, dtype=float)
    elementwise = compare_matrix_entries(ref, new)
    spectrum = compare_matrix_spectra(ref, new, hermitian=True, descending=True)

    trace_sq_ref = float(np.trace(ref @ ref))
    trace_sq_new = float(np.trace(new @ new))
    trace_sq_diff = float(trace_sq_new - trace_sq_ref)
    spectral_small_vs_elementwise_large = (
        spectrum["eigenvalue_max_abs"] < 1e-4 and elementwise["max_abs"] > 1e-2
    )

    return {
        "shape": ref.shape,
        "elementwise": elementwise,
        "spectrum": spectrum,
        "natural_occupations_ref": spectrum["eigenvalues_ref"],
        "natural_occupations_new": spectrum["eigenvalues_new"],
        "trace_ref": elementwise["trace_ref"],
        "trace_new": elementwise["trace_new"],
        "trace_diff": elementwise["trace_diff"],
        "trace_square_ref": trace_sq_ref,
        "trace_square_new": trace_sq_new,
        "trace_square_diff": trace_sq_diff,
        "basis_rotation_likely": bool(spectral_small_vs_elementwise_large),
    }


def _best_column_matching(overlap: np.ndarray) -> dict:
    """Return the maximum-overlap bipartite matching for a coefficient overlap map."""
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError(f"Expected square overlap map, got {overlap.shape}")
    rows, cols = _linear_sum_assignment(-np.abs(overlap))
    matched = np.abs(overlap[rows, cols])
    return {
        "row_indices": rows.tolist(),
        "col_indices": cols.tolist(),
        "matched_abs_overlaps": matched.tolist(),
        "mean_abs_overlap": float(np.mean(matched)),
        "min_abs_overlap": float(np.min(matched)),
        "max_abs_overlap": float(np.max(matched)),
    }


def _compare_orbital_subspaces(coeff_ref: np.ndarray, coeff_new: np.ndarray) -> dict:
    """Compare two coefficient matrices as orbital subspaces and bases."""
    ref = np.asarray(coeff_ref, dtype=float)
    new = np.asarray(coeff_new, dtype=float)
    if ref.shape != new.shape:
        raise ValueError(
            f"Coefficient shapes differ: reference {ref.shape}, new {new.shape}"
        )
    if ref.ndim != 2:
        raise ValueError(f"Expected rank-2 coefficient matrices, got {ref.shape}")

    overlap = ref.T @ new
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    gram_ref = ref.T @ ref
    gram_new = new.T @ new
    gram_delta = gram_new - gram_ref
    elementwise = compare_matrix_entries(ref, new)
    matching = _best_column_matching(overlap)

    return {
        "shape": ref.shape,
        "elementwise": elementwise,
        "overlap_frobenius": float(np.linalg.norm(overlap)),
        "overlap_max_abs": float(np.max(np.abs(overlap))),
        "subspace_singular_values": singular_values,
        "subspace_sigma_min": float(np.min(singular_values)),
        "subspace_sigma_max": float(np.max(singular_values)),
        "gram_frobenius": float(np.linalg.norm(gram_delta)),
        "gram_max_abs": float(np.max(np.abs(gram_delta))),
        "best_matching": matching,
        "subspace_match_likely": bool(np.min(singular_values) > 0.99),
        "basis_reordered_or_rotated_likely": bool(
            np.min(singular_values) > 0.99 and matching["mean_abs_overlap"] < 0.999
        ),
    }


def compare_basis_states(
    basis_state_ref: dict[str, np.ndarray],
    basis_state_new: dict[str, np.ndarray],
) -> dict:
    """Compare Step-7/8 basis-state artifacts in a basis-aware way.

    The main question here is whether two basis-state artifacts span the same
    active orbital subspace even if the localized basis, ordering, or pairing
    representation has changed.
    """
    required = {"active_coeff_alpha", "active_coeff_beta"}
    missing_ref = required - set(basis_state_ref)
    missing_new = required - set(basis_state_new)
    if missing_ref:
        raise ValueError(f"Reference basis state missing required keys: {sorted(missing_ref)}")
    if missing_new:
        raise ValueError(f"New basis state missing required keys: {sorted(missing_new)}")

    result = {
        "alpha_subspace": _compare_orbital_subspaces(
            basis_state_ref["active_coeff_alpha"],
            basis_state_new["active_coeff_alpha"],
        ),
        "beta_subspace": _compare_orbital_subspaces(
            basis_state_ref["active_coeff_beta"],
            basis_state_new["active_coeff_beta"],
        ),
    }

    for occ_key in ("alpha_no_occupations", "beta_no_occupations"):
        if occ_key in basis_state_ref and occ_key in basis_state_new:
            occ_ref = np.asarray(basis_state_ref[occ_key], dtype=float)
            occ_new = np.asarray(basis_state_new[occ_key], dtype=float)
            if occ_ref.shape != occ_new.shape:
                raise ValueError(
                    f"{occ_key} shapes differ: reference {occ_ref.shape}, new {occ_new.shape}"
                )
            delta = occ_new - occ_ref
            result[occ_key] = {
                "shape": occ_ref.shape,
                "frobenius": float(np.linalg.norm(delta)),
                "rms": float(np.sqrt(np.mean(delta**2))),
                "max_abs": float(np.max(np.abs(delta))),
            }

    for discrete_key in ("ordering", "pairs"):
        if discrete_key in basis_state_ref and discrete_key in basis_state_new:
            ref = np.asarray(basis_state_ref[discrete_key])
            new = np.asarray(basis_state_new[discrete_key])
            if ref.shape != new.shape:
                raise ValueError(
                    f"{discrete_key} shapes differ: reference {ref.shape}, new {new.shape}"
                )
            delta = new.astype(float) - ref.astype(float)
            result[discrete_key] = {
                "shape": ref.shape,
                "exact_equal": bool(np.array_equal(ref, new)),
                "frobenius": float(np.linalg.norm(delta)),
                "max_abs": float(np.max(np.abs(delta))),
            }

    result["same_active_subspace_likely"] = bool(
        result["alpha_subspace"]["subspace_match_likely"]
        and result["beta_subspace"]["subspace_match_likely"]
    )
    result["representation_difference_likely"] = bool(
        result["same_active_subspace_likely"]
        and (
            result["alpha_subspace"]["basis_reordered_or_rotated_likely"]
            or result["beta_subspace"]["basis_reordered_or_rotated_likely"]
            or ("ordering" in result and not result["ordering"]["exact_equal"])
            or ("pairs" in result and not result["pairs"]["exact_equal"])
        )
    )
    return result


def _compare_rank4_two_particle_tensor(ref: np.ndarray, new: np.ndarray) -> dict:
    """Compare a single rank-4 2PDM block."""
    n_orb = ref.shape[0]
    delta = new - ref
    pair_ref = ref.reshape(n_orb * n_orb, n_orb * n_orb)
    pair_new = new.reshape(n_orb * n_orb, n_orb * n_orb)
    pair_elementwise = compare_matrix_entries(pair_ref, pair_new)
    pair_spectrum = compare_matrix_spectra(
        pair_ref,
        pair_new,
        hermitian=True,
        descending=True,
    )

    diag_trace_ref = float(np.einsum("pqpq->", ref))
    diag_trace_new = float(np.einsum("pqpq->", new))
    diag_trace_diff = float(diag_trace_new - diag_trace_ref)
    spectral_small_vs_elementwise_large = (
        pair_spectrum["eigenvalue_max_abs"] < 1e-3 and pair_elementwise["max_abs"] > 1e-2
    )

    return {
        "shape": ref.shape,
        "elementwise": {
            "frobenius": float(np.linalg.norm(delta)),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "max_abs": float(np.max(np.abs(delta))),
        },
        "pair_matrix": {
            "shape": pair_ref.shape,
            "elementwise": pair_elementwise,
            "spectrum": pair_spectrum,
        },
        "pair_trace_ref": pair_elementwise["trace_ref"],
        "pair_trace_new": pair_elementwise["trace_new"],
        "pair_trace_diff": pair_elementwise["trace_diff"],
        "diag_trace_ref": diag_trace_ref,
        "diag_trace_new": diag_trace_new,
        "diag_trace_diff": diag_trace_diff,
        "basis_rotation_likely": bool(spectral_small_vs_elementwise_large),
    }


def compare_two_particle_density_tensors(
    tensor_ref: np.ndarray,
    tensor_new: np.ndarray,
) -> dict:
    """Compare a 2PDM-like tensor with basis-aware invariants.

    Supported retained layouts are:
    - `(n, n, n, n)`
    - `(k, n, n, n, n)` for multi-block 2PDM tensors
    """
    ref = np.asarray(tensor_ref, dtype=float)
    new = np.asarray(tensor_new, dtype=float)
    if ref.shape != new.shape:
        raise ValueError(
            f"Tensor shapes differ: reference {ref.shape}, new {new.shape}"
        )
    if ref.ndim == 4:
        if not (ref.shape[0] == ref.shape[1] == ref.shape[2] == ref.shape[3]):
            raise ValueError(f"Expected square rank-4 tensor, got {ref.shape}")
        return _compare_rank4_two_particle_tensor(ref, new)
    if ref.ndim == 5:
        if not (
            ref.shape[1] == ref.shape[2] == ref.shape[3] == ref.shape[4]
        ):
            raise ValueError(f"Expected square rank-5 tensor blocks, got {ref.shape}")
        components = [
            _compare_rank4_two_particle_tensor(ref[idx], new[idx])
            for idx in range(ref.shape[0])
        ]
        delta = new - ref
        all_basis_rotation_likely = all(
            component["basis_rotation_likely"] for component in components
        )
        return {
            "shape": ref.shape,
            "n_components": int(ref.shape[0]),
            "elementwise": {
                "frobenius": float(np.linalg.norm(delta)),
                "rms": float(np.sqrt(np.mean(delta**2))),
                "max_abs": float(np.max(np.abs(delta))),
            },
            "components": components,
            "basis_rotation_likely": bool(all_basis_rotation_likely),
        }
    raise ValueError(f"Expected rank-4 or rank-5 tensor, got {ref.shape}")


def find_chan_benchmark_row(
    chan_benchmark_json: str,
    theory: str,
    *,
    table_key: str = "table5_ucc_series",
) -> dict | None:
    """Return the benchmark row matching a theory label, if present."""
    data = json.loads(_Path(chan_benchmark_json).read_text())
    for row in data.get(table_key, []):
        if row.get("theory") == theory:
            return row
    return None


def compare_two_sz_with_benchmark(
    computed: dict[str, float],
    chan_benchmark_json: str,
    *,
    theory: str = "UHF",
    table_key: str = "table5_ucc_series",
) -> dict:
    """Compare computed local spins against Chan benchmark values."""
    row = find_chan_benchmark_row(
        chan_benchmark_json,
        theory,
        table_key=table_key,
    )
    if row is None:
        raise ValueError(f"Benchmark row {theory!r} not found in {chan_benchmark_json}")

    reference = {
        "Fe1": float(row["two_sz_fe1"]),
        "Fe2": float(row["two_sz_fe2"]),
    }
    direct = {key: float(computed[key] - reference[key]) for key in reference}
    flipped = {key: float((-computed[key]) - reference[key]) for key in reference}

    direct_error = max(abs(value) for value in direct.values())
    flipped_error = max(abs(value) for value in flipped.values())
    mode = "direct" if direct_error <= flipped_error else "global_sign_flip"
    chosen = direct if mode == "direct" else flipped

    return {
        "reference": reference,
        "direct_delta": direct,
        "global_sign_flip_delta": flipped,
        "best_alignment": mode,
        "best_delta": chosen,
    }


def compare_energy_triplet(
    *,
    computed_active: float,
    computed_core: float,
    computed_total: float,
    reference_active: float,
    reference_core: float,
    reference_total: float,
) -> dict:
    """Compare active/core/total energies against a benchmark triplet."""
    computed = {
        "E_active": float(computed_active),
        "E_core": float(computed_core),
        "E_total": float(computed_total),
    }
    reference = {
        "E_active": float(reference_active),
        "E_core": float(reference_core),
        "E_total": float(reference_total),
    }
    delta = {key: float(computed[key] - reference[key]) for key in computed}
    return {
        "computed": computed,
        "reference": reference,
        "delta": delta,
    }


def compare_fcidumps(
    path_ref: str,
    path_new: str,
    *,
    eigval_tol: float = 0.1,
    ecore_tol: float = 1e-6,
) -> dict:
    """Compare two FCIDUMP files using eigenvalue decomposition."""
    ref = fcidump_mod.read(path_ref, verbose=False)
    new = fcidump_mod.read(path_new, verbose=False)

    h1_ref, h1_new = ref["H1"], new["H1"]
    n_ref, n_new = h1_ref.shape[0], h1_new.shape[0]

    eigvals_ref, eigvecs_ref = np.linalg.eigh(h1_ref)
    eigvals_new, eigvecs_new = np.linalg.eigh(h1_new)
    sort_ref = np.argsort(eigvals_ref)
    sort_new = np.argsort(eigvals_new)
    eigvals_ref_sorted = eigvals_ref[sort_ref]
    eigvals_new_sorted = eigvals_new[sort_new]
    eigvecs_ref_sorted = eigvecs_ref[:, sort_ref]
    eigvecs_new_sorted = eigvecs_new[:, sort_new]

    n_common = min(n_ref, n_new)
    eigval_diff = eigvals_new_sorted[:n_common] - eigvals_ref_sorted[:n_common]
    eigval_frob = float(np.linalg.norm(eigval_diff))
    eigval_max = float(np.max(np.abs(eigval_diff)))
    n_mismatch = int(np.sum(np.abs(eigval_diff) > eigval_tol))

    sorted_pairs = []
    for rank in range(n_common):
        vec_ref = eigvecs_ref_sorted[:, rank]
        vec_new = eigvecs_new_sorted[:, rank]
        top3_ref = np.argsort(np.abs(vec_ref))[-3:][::-1].tolist()
        top3_new = np.argsort(np.abs(vec_new))[-3:][::-1].tolist()
        sorted_pairs.append(
            {
                "rank": rank,
                "eigval_ref": float(eigvals_ref_sorted[rank]),
                "eigval_new": float(eigvals_new_sorted[rank]),
                "eigval_diff": float(eigval_diff[rank]),
                "top3_ref": top3_ref,
                "top3_new": top3_new,
            }
        )

    h1_frob = float(np.linalg.norm(h1_new[:n_common, :n_common] - h1_ref[:n_common, :n_common]))

    h2_rms = 0.0
    h2_max = 0.0
    h2_ref, h2_new = ref["H2"], new["H2"]
    if h2_ref.shape == h2_new.shape:
        delta_h2 = h2_new - h2_ref
        h2_rms = float(np.sqrt(np.mean(delta_h2**2)))
        h2_max = float(np.max(np.abs(delta_h2)))

    ecore_ref = float(ref["ECORE"])
    ecore_new = float(new["ECORE"])
    ecore_diff = ecore_new - ecore_ref
    match = eigval_frob < eigval_tol and abs(ecore_diff) < ecore_tol

    return {
        "norb_ref": n_ref,
        "norb_new": n_new,
        "nelec_ref": int(ref["NELEC"]),
        "nelec_new": int(new["NELEC"]),
        "ms2_ref": int(ref["MS2"]),
        "ms2_new": int(new["MS2"]),
        "eigval_frobenius": eigval_frob,
        "eigval_max": eigval_max,
        "eigval_ref": eigvals_ref_sorted,
        "eigval_new": eigvals_new_sorted,
        "eigval_diff": eigval_diff,
        "n_eigval_mismatch": n_mismatch,
        "sorted_pairs": sorted_pairs,
        "h1e_frobenius": h1_frob,
        "h2e_rms": h2_rms,
        "h2e_max": h2_max,
        "ecore_ref": ecore_ref,
        "ecore_new": ecore_new,
        "ecore_diff": ecore_diff,
        "match": match,
    }


def _compare_vector_entries(vector_ref: np.ndarray, vector_new: np.ndarray) -> dict:
    """Compare two same-shaped 1D numeric arrays."""
    ref = np.asarray(vector_ref, dtype=float)
    new = np.asarray(vector_new, dtype=float)
    if ref.shape != new.shape:
        raise ValueError(
            f"Vector shapes differ: reference {ref.shape}, new {new.shape}"
        )
    delta = new - ref
    return {
        "shape": ref.shape,
        "frobenius": float(np.linalg.norm(delta)),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
    }


def _compare_numeric_array(array_ref: np.ndarray, array_new: np.ndarray) -> dict:
    """Compare two numeric arrays using the best available structural heuristic."""
    ref = np.asarray(array_ref)
    new = np.asarray(array_new)
    if ref.shape != new.shape:
        raise ValueError(
            f"Array shapes differ: reference {ref.shape}, new {new.shape}"
        )

    if ref.ndim == 1:
        return {"kind": "vector", "comparison": _compare_vector_entries(ref, new)}

    if ref.ndim == 2:
        comparison = compare_matrix_entries(ref, new)
        result = {"kind": "matrix", "comparison": comparison}
        if ref.shape[0] == ref.shape[1]:
            is_symmetric = np.allclose(ref, ref.T, atol=1e-8) and np.allclose(
                new, new.T, atol=1e-8
            )
            result["spectrum"] = compare_matrix_spectra(
                ref,
                new,
                hermitian=bool(is_symmetric),
                descending=True,
            )
        return result

    if ref.ndim in (4, 5):
        return {
            "kind": "two_particle_tensor",
            "comparison": compare_two_particle_density_tensors(ref, new),
        }

    delta = new.astype(float) - ref.astype(float)
    return {
        "kind": f"rank{ref.ndim}_array",
        "comparison": {
            "shape": ref.shape,
            "frobenius": float(np.linalg.norm(delta)),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "max_abs": float(np.max(np.abs(delta))),
        },
    }


def _flatten_json_scalars(payload, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    """Flatten a JSON-like object into scalar leaf paths."""
    flat: dict[str, float | int | str | bool | None] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_json_scalars(value, child))
        return flat
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            child = f"{prefix}[{idx}]"
            flat.update(_flatten_json_scalars(value, child))
        return flat
    flat[prefix] = payload
    return flat


def _compare_json_payloads(payload_ref, payload_new) -> dict:
    """Compare two JSON payloads via flattened scalar leaves."""
    flat_ref = _flatten_json_scalars(payload_ref)
    flat_new = _flatten_json_scalars(payload_new)
    keys_ref = set(flat_ref)
    keys_new = set(flat_new)
    common = sorted(keys_ref & keys_new)

    numeric_common = []
    differing_scalars = []
    matching_scalars = 0
    for key in common:
        ref = flat_ref[key]
        new = flat_new[key]
        if isinstance(ref, (int, float, bool)) and isinstance(
            new, (int, float, bool)
        ):
            delta = float(new) - float(ref)
            numeric_common.append((key, float(ref), float(new), delta))
        elif ref == new:
            matching_scalars += 1
        else:
            differing_scalars.append({"path": key, "ref": ref, "new": new})

    numeric_deltas = np.array([row[3] for row in numeric_common], dtype=float)
    numeric_summary = None
    if numeric_common:
        numeric_summary = {
            "count": len(numeric_common),
            "frobenius": float(np.linalg.norm(numeric_deltas)),
            "rms": float(np.sqrt(np.mean(numeric_deltas**2))),
            "max_abs": float(np.max(np.abs(numeric_deltas))),
        }

    return {
        "kind": "json",
        "paths_ref": len(flat_ref),
        "paths_new": len(flat_new),
        "only_in_ref": sorted(keys_ref - keys_new),
        "only_in_new": sorted(keys_new - keys_ref),
        "matching_scalar_count": matching_scalars,
        "differing_scalar_count": len(differing_scalars),
        "differing_scalars_preview": differing_scalars[:20],
        "numeric_summary": numeric_summary,
    }


def _h5_top_groups(h5) -> list[str]:
    return sorted(h5.keys())


def _h5_metadata_attrs(h5) -> dict:
    if "metadata" not in h5:
        return {}
    attrs = {}
    for key in h5["metadata"].attrs:
        value = h5["metadata"].attrs[key]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        attrs[str(key)] = value.item() if hasattr(value, "item") else value
    return attrs


def _infer_h5_kind(h5) -> str:
    attrs = _h5_metadata_attrs(h5)
    artifact_type = str(attrs.get("artifact_type", "")) if attrs else ""
    if artifact_type:
        return artifact_type
    if "dmrg_1rdm" in h5 and "noon" in h5:
        return "apex_cas_testcas_dmrg"
    if "density_matrices" in h5 and "dm_a" in h5["density_matrices"]:
        return "apex_filter_step3_uhf_state"
    if "orbitals" in h5 and "active_coeff_alpha" in h5["orbitals"]:
        return "apex_filter_step7_dmrg_basis"
    if "basis_state" in h5 and "density_matrices" in h5 and "2pdm" in h5["density_matrices"]:
        return "apex_filter_step8_dmrg"
    return "generic_hdf5"


def _compare_step3_uhf_h5(h5_ref, h5_new) -> dict:
    return {
        "kind": "apex_filter_step3_uhf_state",
        "top_groups_ref": _h5_top_groups(h5_ref),
        "top_groups_new": _h5_top_groups(h5_new),
        "metadata_ref": _h5_metadata_attrs(h5_ref),
        "metadata_new": _h5_metadata_attrs(h5_new),
        "density_matrices": {
            "dm_a": compare_density_matrices(
                h5_ref["density_matrices"]["dm_a"][...],
                h5_new["density_matrices"]["dm_a"][...],
            ),
            "dm_b": compare_density_matrices(
                h5_ref["density_matrices"]["dm_b"][...],
                h5_new["density_matrices"]["dm_b"][...],
            ),
        },
        "occupations": {
            "mo_occ_a": _compare_numeric_array(
                h5_ref["orbitals"]["mo_occ_a"][...],
                h5_new["orbitals"]["mo_occ_a"][...],
            ),
            "mo_occ_b": _compare_numeric_array(
                h5_ref["orbitals"]["mo_occ_b"][...],
                h5_new["orbitals"]["mo_occ_b"][...],
            ),
        },
    }


def _compare_step7_basis_h5(h5_ref, h5_new) -> dict:
    orbitals_ref = h5_ref["orbitals"]
    orbitals_new = h5_new["orbitals"]
    return {
        "kind": "apex_filter_step7_dmrg_basis",
        "top_groups_ref": _h5_top_groups(h5_ref),
        "top_groups_new": _h5_top_groups(h5_new),
        "metadata_ref": _h5_metadata_attrs(h5_ref),
        "metadata_new": _h5_metadata_attrs(h5_new),
        "basis_state": compare_basis_states(
            {
                "active_coeff_alpha": orbitals_ref["active_coeff_alpha"][...],
                "active_coeff_beta": orbitals_ref["active_coeff_beta"][...],
                "alpha_no_occupations": orbitals_ref["alpha_no_occupations"][...],
                "beta_no_occupations": orbitals_ref["beta_no_occupations"][...],
                "ordering": orbitals_ref["ordering"][...],
                "pairs": orbitals_ref["pairs"][...],
            },
            {
                "active_coeff_alpha": orbitals_new["active_coeff_alpha"][...],
                "active_coeff_beta": orbitals_new["active_coeff_beta"][...],
                "alpha_no_occupations": orbitals_new["alpha_no_occupations"][...],
                "beta_no_occupations": orbitals_new["beta_no_occupations"][...],
                "ordering": orbitals_new["ordering"][...],
                "pairs": orbitals_new["pairs"][...],
            },
        ),
    }


def _compare_step8_dmrg_h5(h5_ref, h5_new) -> dict:
    basis_ref = h5_ref["basis_state"]["orbitals"]
    basis_new = h5_new["basis_state"]["orbitals"]
    return {
        "kind": "apex_filter_step8_dmrg",
        "top_groups_ref": _h5_top_groups(h5_ref),
        "top_groups_new": _h5_top_groups(h5_new),
        "metadata_ref": _h5_metadata_attrs(h5_ref),
        "metadata_new": _h5_metadata_attrs(h5_new),
        "basis_state": compare_basis_states(
            {
                "active_coeff_alpha": basis_ref["active_coeff_alpha"][...],
                "active_coeff_beta": basis_ref["active_coeff_beta"][...],
                "alpha_no_occupations": basis_ref["alpha_no_occupations"][...],
                "beta_no_occupations": basis_ref["beta_no_occupations"][...],
                "ordering": basis_ref["ordering"][...],
                "pairs": basis_ref["pairs"][...],
            },
            {
                "active_coeff_alpha": basis_new["active_coeff_alpha"][...],
                "active_coeff_beta": basis_new["active_coeff_beta"][...],
                "alpha_no_occupations": basis_new["alpha_no_occupations"][...],
                "beta_no_occupations": basis_new["beta_no_occupations"][...],
                "ordering": basis_new["ordering"][...],
                "pairs": basis_new["pairs"][...],
            },
        ),
        "two_pdm": compare_two_particle_density_tensors(
            h5_ref["density_matrices"]["2pdm"][...],
            h5_new["density_matrices"]["2pdm"][...],
        ),
        "discarded_weights": _compare_numeric_array(
            h5_ref["dmrg_diagnostics"]["discarded_weights"][...],
            h5_new["dmrg_diagnostics"]["discarded_weights"][...],
        ),
    }


def _compare_testcas_dmrg_h5(h5_ref, h5_new) -> dict:
    result = {
        "kind": "apex_cas_testcas_dmrg",
        "top_groups_ref": _h5_top_groups(h5_ref),
        "top_groups_new": _h5_top_groups(h5_new),
        "metadata_ref": _h5_metadata_attrs(h5_ref),
        "metadata_new": _h5_metadata_attrs(h5_new),
        "dmrg_1rdm": compare_density_matrices(
            h5_ref["dmrg_1rdm"][...],
            h5_new["dmrg_1rdm"][...],
        ),
    }
    if "noon" in h5_ref and "noon" in h5_new:
        result["noon"] = _compare_numeric_array(h5_ref["noon"][...], h5_new["noon"][...])
    return result


def _compare_generic_hdf5(h5_ref, h5_new) -> dict:
    groups_ref = _h5_top_groups(h5_ref)
    groups_new = _h5_top_groups(h5_new)
    return {
        "kind": "generic_hdf5",
        "top_groups_ref": groups_ref,
        "top_groups_new": groups_new,
        "only_in_ref": sorted(set(groups_ref) - set(groups_new)),
        "only_in_new": sorted(set(groups_new) - set(groups_ref)),
        "metadata_ref": _h5_metadata_attrs(h5_ref),
        "metadata_new": _h5_metadata_attrs(h5_new),
    }


def _compare_hdf5_artifacts(path_ref: str, path_new: str) -> dict:
    with h5py.File(path_ref, "r") as h5_ref, h5py.File(path_new, "r") as h5_new:
        kind_ref = _infer_h5_kind(h5_ref)
        kind_new = _infer_h5_kind(h5_new)
        if kind_ref != kind_new:
            raise ValueError(
                f"HDF5 artifact kinds differ: reference {kind_ref!r}, new {kind_new!r}"
            )
        if kind_ref == "apex_filter_step3_uhf_state":
            return _compare_step3_uhf_h5(h5_ref, h5_new)
        if kind_ref == "apex_filter_step7_dmrg_basis":
            return _compare_step7_basis_h5(h5_ref, h5_new)
        if kind_ref == "apex_filter_step8_dmrg":
            return _compare_step8_dmrg_h5(h5_ref, h5_new)
        if kind_ref == "apex_cas_testcas_dmrg":
            return _compare_testcas_dmrg_h5(h5_ref, h5_new)
        return _compare_generic_hdf5(h5_ref, h5_new)


def _compare_npz_artifacts(path_ref: str, path_new: str) -> dict:
    with np.load(path_ref, allow_pickle=False) as npz_ref, np.load(
        path_new, allow_pickle=False
    ) as npz_new:
        keys_ref = set(npz_ref.files)
        keys_new = set(npz_new.files)
        common = sorted(keys_ref & keys_new)
        arrays = {}
        for key in common:
            arrays[key] = _compare_numeric_array(npz_ref[key], npz_new[key])
        return {
            "kind": "npz",
            "keys_ref": sorted(keys_ref),
            "keys_new": sorted(keys_new),
            "only_in_ref": sorted(keys_ref - keys_new),
            "only_in_new": sorted(keys_new - keys_ref),
            "arrays": arrays,
        }


def _compare_json_artifacts(path_ref: str, path_new: str) -> dict:
    return _compare_json_payloads(
        json.loads(_Path(path_ref).read_text()),
        json.loads(_Path(path_new).read_text()),
    )


def _is_fcidump_path(path: _Path) -> bool:
    name = path.name
    if name.startswith("FCIDUMP"):
        return True
    if path.suffix == ".FCIDUMP":
        return True
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    except Exception:
        return False
    return bool(first_line and ("&FCI" in first_line[0] or "NORB" in first_line[0]))


def compare_artifacts(path_ref: str, path_new: str) -> dict:
    """Compare two retained artifacts through a type-aware shared entry point."""
    ref = _Path(path_ref)
    new = _Path(path_new)
    if not ref.exists():
        raise FileNotFoundError(f"Reference artifact not found: {ref}")
    if not new.exists():
        raise FileNotFoundError(f"New artifact not found: {new}")

    if _is_fcidump_path(ref) and _is_fcidump_path(new):
        result = compare_fcidumps(str(ref), str(new))
        result["kind"] = "fcidump"
        return result

    suffix_ref = ref.suffix.lower()
    suffix_new = new.suffix.lower()
    if suffix_ref != suffix_new:
        raise ValueError(
            f"Artifact suffixes differ: reference {suffix_ref!r}, new {suffix_new!r}"
        )
    if suffix_ref in {".h5", ".hdf5"}:
        return _compare_hdf5_artifacts(str(ref), str(new))
    if suffix_ref == ".npz":
        return _compare_npz_artifacts(str(ref), str(new))
    if suffix_ref == ".json":
        return _compare_json_artifacts(str(ref), str(new))

    raise ValueError(
        f"Unsupported artifact type for compare_artifacts: {ref.name} vs {new.name}"
    )
