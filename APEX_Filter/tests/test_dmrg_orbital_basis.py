"""Tests for DMRG orbital-basis preparation, ordering, and internal persistence helpers."""

import json
import os
from unittest.mock import patch

import h5py
import numpy as np

from apex_filter.dmrg_orbital_basis import (
    _build_dmrg_orbital_basis,
    _save_dmrg_orbital_basis,
)
from shared.fcidump_io import FCIDUMPData
from shared.orbital_methods.localization import localize_orbital_block
from shared.orbital_methods.ordering import (
    chain_distance_cost,
    compute_ordering_matrix as compute_dmrg_ordering_matrix,
    compute_overlap_proxy_matrix,
    fiedler_ordering,
    genetic_algorithm_ordering,
)
from shared.orbital_methods.pairing import pair_alpha_beta_orbitals_with_overlap
from shared.models import CAS, ActiveSpaceLevel


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
        self.exponent = None
        self.init_guess = None
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


@patch("apex_filter.dmrg_orbital_basis._genetic_algorithm_ordering", return_value=[0, 1])
@patch("apex_filter.dmrg_orbital_basis._compute_dmrg_ordering_matrix", return_value=np.array([[0.0, 1.0], [1.0, 0.0]]))
@patch("apex_filter.dmrg_orbital_basis._pair_alpha_beta_orbitals", return_value=[(0, 0), (1, 1)])
@patch("apex_filter.dmrg_orbital_basis._reorder_beta_to_match_alpha", side_effect=lambda pairs, mo_beta, n: mo_beta)
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

    result = _build_dmrg_orbital_basis(
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

    loc = localize_orbital_block(
        mol,
        mo,
        method="pm",
        lo_module=__import__("pyscf.lo", fromlist=["PM"]),
        pm_pop_method="mulliken",
        pm_conv_tol=1e-8,
        pm_conv_tol_grad=1e-4,
        pm_max_cycle=250,
        pm_exponent=4,
        pm_init_guess="cholesky",
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
    assert pm_obj.exponent == 4
    assert pm_obj.init_guess == "cholesky"


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


def test_internal_dmrg_basis_writer_saves_npz(tmp_path):
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
    _save_dmrg_orbital_basis(result, out_npz)
    data = np.load(out_npz, allow_pickle=True)
    assert "mo_coeff_alpha" in data.files
    assert "pairs" in data.files
    assert "ordering_matrix_mode" in data.files
    assert "active_coeff_alpha" in data.files
    assert "pair_diag_overlap_min" in data.files
    assert "ga_cost" in data.files
    assert data["nocc_alpha"] == 1


def test_internal_dmrg_basis_writer_copies_reference_state_metadata(tmp_path):
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

    ref_h5 = os.path.join(tmp_path, "ref_uhf.h5")
    with h5py.File(ref_h5, "w") as h5:
        mol = h5.create_group("molecule")
        mol.attrs["charge"] = -2
        mol.create_dataset("atom_positions", data=np.zeros((1, 3)))
        mapping = h5.create_group("active_space_mapping")
        mapping.create_dataset("active_indices", data=np.array([0, 1], dtype=int))

    out_npz = os.path.join(tmp_path, "basis_with_ref.npz")
    _save_dmrg_orbital_basis(
        result,
        out_npz,
        label="BS7|235",
        family="BS7",
        energy=-1.23,
        reference_state_path=ref_h5,
        fcidump_path="/tmp/FCIDUMP.test",
        settings_payload={
            "control_source": "/tmp/method_controls.yaml",
            "theory": "DMRG basis",
            "ordering_matrix_mode": "exchange_proxy",
        },
    )
    out_h5 = out_npz[:-4] + ".h5"
    with h5py.File(out_h5, "r") as h5:
        assert h5["metadata"].attrs["label"] == "BS7|235"
        assert h5["metadata"].attrs["family"] == "BS7"
        assert float(h5["metadata"].attrs["energy"]) == -1.23
        assert h5["metadata"].attrs["reference_state_path"] == ref_h5
        assert h5["metadata"].attrs["source_fcidump_path"] == "/tmp/FCIDUMP.test"
        settings = json.loads(h5["metadata"].attrs["settings_json"])
        assert settings["control_source"] == "/tmp/method_controls.yaml"
        assert settings["theory"] == "DMRG basis"
        assert settings["requested_config"]["theory"] == "DMRG basis"
        assert settings["effective_method"]["theory"] == "DMRG basis"
        assert settings["effective_method"]["source_method"] == "UCCSD-NO/split-localized/paired/GA-ordered"
        assert settings["effective_method"]["ordering_matrix_mode"] == "exchange_proxy"
        for key in ("scf_method", "xc_functional", "relativistic", "solvation_model"):
            assert key not in settings["effective_parameters"]
        assert "molecule" in h5
        assert "active_space_mapping" in h5
        assert "active_indices" in h5["active_space_mapping"]
