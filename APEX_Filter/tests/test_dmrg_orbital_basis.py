"""Tests for DMRG orbital-basis preparation and ordering."""

import os
from unittest.mock import patch

import numpy as np

from apex_filter.CAS_loader import FCIDUMPData
from apex_filter.dmrg_orbital_basis import (
    build_dmrg_orbital_basis,
    chain_distance_cost,
    compute_dmrg_ordering_matrix,
    compute_overlap_proxy_matrix,
    fiedler_ordering,
    genetic_algorithm_ordering,
    _localize_block,
    pair_alpha_beta_orbitals_with_overlap,
    reorder_cas_orbital_coefficients,
    save_dmrg_orbital_basis,
)
from apex_filter.models import CAS, ActiveSpaceLevel
from apex_filter.reference_uhf import build_fake_mol


def _make_toy_fcidump():
    return FCIDUMPData(
        h1e=np.diag([-1.0, -0.5]),
        h2e=np.zeros((2, 2, 2, 2)),
        ecore=-10.0,
        norb=2,
        nelec=2,
        ms2=0,
    )


def _write_toy_uhf_npz(tmp_path, fcidump_data):
    npz_path = os.path.join(tmp_path, "toy_dmrg_basis_uhf.npz")
    np.savez(
        npz_path,
        energy=-11.0,
        converged=True,
        spin_sq=0.0,
        mo_coeff_a=np.eye(fcidump_data.norb),
        mo_coeff_b=np.eye(fcidump_data.norb),
        mo_occ_a=np.array([1.0, 0.0]),
        mo_occ_b=np.array([1.0, 0.0]),
        mo_energy_a=np.array([-1.0, 0.5]),
        mo_energy_b=np.array([-1.0, 0.5]),
    )
    return npz_path


class _FakeLocalizer:
    instances = []

    def __init__(self, mol, mo_block, pop_method=None):
        self.mo_block = mo_block
        self.pop_method = pop_method
        self.conv_tol = None
        self.conv_tol_grad = None
        self.max_cycle = None
        self.__class__.instances.append(self)

    def kernel(self):
        return self.mo_block


class _FakeCC:
    instances = []

    def __init__(self, mf):
        self.mf = mf
        self.conv_tol = None
        self.max_cycle = None
        self.diis_space = None
        self.direct = None
        self.converged = True
        self.e_corr = -0.1
        self.e_tot = float(mf.e_tot) - 0.1
        self.__class__.instances.append(self)

    def kernel(self):
        return self.e_corr, None, None

    def make_rdm1(self, ao_repr=False):
        dm = np.array([[0.95, 0.0], [0.0, 0.05]])
        return dm, dm


class _FakeMolWithOverlap:
    def __init__(self, s_matrix):
        self._s = np.array(s_matrix, dtype=float)

    def intor_symmetric(self, name):
        assert name == "int1e_ovlp"
        return self._s.copy()


@patch("apex_filter.dmrg_orbital_basis.genetic_algorithm_ordering", return_value=[0, 1])
@patch("apex_filter.dmrg_orbital_basis.compute_dmrg_ordering_matrix", return_value=np.array([[0.0, 1.0], [1.0, 0.0]]))
@patch("apex_filter.dmrg_orbital_basis.pair_alpha_beta_orbitals", return_value=[(0, 0), (1, 1)])
@patch("apex_filter.dmrg_orbital_basis.reorder_beta_to_match_alpha", side_effect=lambda pairs, mo_beta, n: mo_beta)
@patch("pyscf.lo.PM", side_effect=lambda mol, mo_block, pop_method=None: _FakeLocalizer(mol, mo_block, pop_method=pop_method))
@patch("pyscf.cc.UCCSD", side_effect=lambda mf: _FakeCC(mf))
def test_build_dmrg_orbital_basis_pipeline(
    mock_uccsd,
    mock_pm,
    mock_reorder,
    mock_pair,
    mock_matrix,
    mock_ga,
    tmp_path,
):
    _FakeCC.instances.clear()
    _FakeLocalizer.instances.clear()
    fcid = _make_toy_fcidump()
    uhf_npz = _write_toy_uhf_npz(tmp_path, fcid)

    mol = _FakeMolWithOverlap(np.eye(2))
    cas = CAS(
        n_electrons=2,
        n_orbitals=2,
        level=ActiveSpaceLevel.MINIMAL,
        mo_coeff_alpha=np.eye(2),
        mo_coeff_beta=np.eye(2),
    )

    result = build_dmrg_orbital_basis(
        mol,
        cas,
        fcid,
        uhf_npz,
        localization_method="pm",
        cc_conv_tol=1e-10,
        cc_max_cycle=777,
        cc_diis_space=16,
        pm_pop_method="mulliken",
        pm_conv_tol=1e-8,
        pm_conv_tol_grad=1e-4,
        pm_max_cycle=250,
        ga_generations=5,
        ga_population=22,
        ga_mutation_rate=0.25,
        ga_seed=13,
        ordering_matrix_mode="exchange_proxy",
    )

    assert result.mo_coeff_alpha.shape == (2, 2)
    assert result.mo_coeff_beta.shape == (2, 2)
    assert result.nocc_alpha == 1
    assert result.nocc_beta == 1
    assert result.ordering == [0, 1]
    assert result.localization_method == "pm"
    assert result.ordering_matrix_mode == "exchange_proxy"
    assert result.ordering_objective == "distance_weighted"
    assert result.pair_diag_overlap_min == 1.0
    assert result.pair_diag_overlap_mean == 1.0
    assert result.diag_dominant_fraction == 1.0
    assert result.ordering_is_permutation is True
    assert result.ga_cost == 1.0
    assert result.fiedler_cost == 1.0
    assert result.orth_err_alpha < 1e-12
    assert result.orth_err_beta < 1e-12
    cc_obj = _FakeCC.instances[-1]
    assert cc_obj.conv_tol == 1e-10
    assert cc_obj.max_cycle == 777
    assert cc_obj.diis_space == 16
    assert cc_obj.direct is False


@patch("pyscf.lo.PM", side_effect=lambda mol, mo_block, pop_method=None: _FakeLocalizer(mol, mo_block, pop_method=pop_method))
def test_localize_block_applies_pm_controls(mock_pm):
    _FakeLocalizer.instances.clear()
    mol = _FakeMolWithOverlap(np.eye(2))
    mo = np.eye(2)

    loc = _localize_block(
        mol,
        mo,
        method="pm",
        lo_module=__import__("pyscf.lo", fromlist=["PM"]),
        pm_pop_method="mulliken",
        pm_conv_tol=1e-8,
        pm_conv_tol_grad=1e-4,
        pm_max_cycle=250,
        boys_conv_tol=1e-6,
        boys_conv_tol_grad=None,
        boys_max_cycle=100,
    )

    assert np.allclose(loc, mo)
    pm_obj = _FakeLocalizer.instances[-1]
    assert pm_obj.pop_method == "mulliken"
    assert pm_obj.conv_tol == 1e-8
    assert pm_obj.conv_tol_grad == 1e-4
    assert pm_obj.max_cycle == 250


def test_pair_alpha_beta_orbitals_with_overlap_stable_assignment():
    s = np.eye(2)
    mo_alpha = np.eye(2)
    mo_beta = np.array([[0.0, 1.0], [1.0, 0.0]])

    pairs = pair_alpha_beta_orbitals_with_overlap(s, mo_alpha, mo_beta)

    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_genetic_algorithm_ordering_not_worse_than_fiedler():
    interaction = np.array(
        [
            [0.0, 8.0, 1.0, 1.0],
            [8.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 7.0],
            [1.0, 1.0, 7.0, 0.0],
        ]
    )

    fiedler = fiedler_ordering(interaction)
    ga = genetic_algorithm_ordering(
        interaction,
        n_generations=40,
        population_size=20,
        mutation_rate=0.2,
        seed=7,
    )

    assert chain_distance_cost(interaction, ga) <= chain_distance_cost(interaction, fiedler)


def test_compute_overlap_proxy_matrix_is_symmetric_and_zero_diagonal():
    mol = _FakeMolWithOverlap(np.eye(2))
    mo = np.array([[1.0, 0.1], [0.1, 1.0]])

    mat = compute_overlap_proxy_matrix(mol, mo)

    assert mat.shape == (2, 2)
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)
    assert mat[0, 1] > 0.0


def test_compute_dmrg_ordering_matrix_falls_back_for_large_exchange_proxy(caplog):
    mol = _FakeMolWithOverlap(np.eye(3))
    mo = np.eye(3)

    with caplog.at_level("WARNING"):
        mat = compute_dmrg_ordering_matrix(
            mol,
            mo,
            mode="exchange_proxy",
            exchange_proxy_max_orbitals=2,
        )

    assert "falling back to overlap_proxy" in caplog.text
    assert np.allclose(mat, mat.T)


def test_reorder_cas_orbital_coefficients_updates_labels_and_ordering():
    cas = CAS(
        n_electrons=4,
        n_orbitals=3,
        mo_coeff_alpha=np.eye(3),
        mo_coeff_beta=np.eye(3),
        occupations=np.array([1.8, 1.0, 0.2]),
        orbital_labels=["Fe1_3dxy", "S2_3px", "Fe2_3dz2"],
    )

    reordered = reorder_cas_orbital_coefficients(cas, [2, 0, 1])

    assert reordered.orbital_labels == ["Fe2_3dz2", "Fe1_3dxy", "S2_3px"]
    assert np.allclose(reordered.mo_coeff_alpha, np.eye(3)[:, [2, 0, 1]])
    assert np.allclose(reordered.occupations, [0.2, 1.8, 1.0])
    assert np.allclose(reordered.orbital_ordering, [2, 0, 1])


def test_save_dmrg_orbital_basis(tmp_path):
    result = type("R", (), {})()
    result.mo_coeff_alpha = np.eye(2)
    result.mo_coeff_beta = np.eye(2)
    result.active_coeff_alpha = np.eye(2)
    result.active_coeff_beta = np.eye(2)
    result.alpha_no_occupations = np.array([1.0, 0.0])
    result.beta_no_occupations = np.array([1.0, 0.0])
    result.nocc_alpha = 1
    result.nocc_beta = 1
    result.pairs = [(0, 0), (1, 1)]
    result.ordering = [0, 1]
    result.localization_method = "pm"
    result.source_method = "UCCSD-NO/split-localized/paired/GA-ordered"
    result.ordering_matrix_mode = "exchange_proxy"
    result.ordering_objective = "distance_weighted"
    result.pair_diag_overlap_min = 0.9
    result.pair_diag_overlap_mean = 0.95
    result.diag_dominant_fraction = 1.0
    result.orth_err_alpha = 1e-14
    result.orth_err_beta = 2e-14
    result.ordering_is_permutation = True
    result.ga_cost = 1.0
    result.fiedler_cost = 1.2

    out_npz = os.path.join(tmp_path, "basis.npz")
    save_dmrg_orbital_basis(result, out_npz)
    data = np.load(out_npz, allow_pickle=True)
    assert "mo_coeff_alpha" in data.files
    assert "pairs" in data.files
    assert "ordering_matrix_mode" in data.files
    assert "active_coeff_alpha" in data.files
    assert "pair_diag_overlap_min" in data.files
    assert "ga_cost" in data.files
    assert data["nocc_alpha"] == 1
