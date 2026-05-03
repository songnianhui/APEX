"""Orbital-localization primitives."""

from __future__ import annotations

import numpy as np


def normalize_pm_pop_method(value: str | None) -> str:
    """Normalize Pipek-Mezey population-method spelling for PySCF."""
    if value is None:
        return "mulliken"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "meta_lowdin": "meta_lowdin",
        "metalowdin": "meta_lowdin",
        "mulliken": "mulliken",
        "lowdin": "lowdin",
        "iao": "iao",
        "becke": "becke",
    }
    return aliases.get(normalized, normalized)


def split_localize_unrestricted(
    mol,
    ao_coeff: np.ndarray,
    nocc: int,
    *,
    method: str,
    lo_module,
    pm_pop_method: str,
    pm_conv_tol: float,
    pm_conv_tol_grad: float | None,
    pm_max_cycle: int,
    pm_exponent: int,
    pm_init_guess: str,
    boys_conv_tol: float,
    boys_conv_tol_grad: float | None,
    boys_max_cycle: int,
):
    """Localize occupied and virtual blocks separately for one spin channel."""
    occ_block = ao_coeff[:, :nocc]
    vir_block = ao_coeff[:, nocc:]
    loc_occ = localize_orbital_block(
        mol,
        occ_block,
        method=method,
        lo_module=lo_module,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        pm_exponent=pm_exponent,
        pm_init_guess=pm_init_guess,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )
    loc_vir = localize_orbital_block(
        mol,
        vir_block,
        method=method,
        lo_module=lo_module,
        pm_pop_method=pm_pop_method,
        pm_conv_tol=pm_conv_tol,
        pm_conv_tol_grad=pm_conv_tol_grad,
        pm_max_cycle=pm_max_cycle,
        pm_exponent=pm_exponent,
        pm_init_guess=pm_init_guess,
        boys_conv_tol=boys_conv_tol,
        boys_conv_tol_grad=boys_conv_tol_grad,
        boys_max_cycle=boys_max_cycle,
    )
    return np.hstack([loc_occ, loc_vir])


def localize_orbital_block(
    mol,
    mo_block: np.ndarray,
    *,
    method: str,
    lo_module,
    pm_pop_method: str,
    pm_conv_tol: float,
    pm_conv_tol_grad: float | None,
    pm_max_cycle: int,
    pm_exponent: int,
    pm_init_guess: str,
    boys_conv_tol: float,
    boys_conv_tol_grad: float | None,
    boys_max_cycle: int,
):
    """Localize one orbital block with PM or Boys."""
    if mo_block.shape[1] <= 1:
        return mo_block
    if method == "pm":
        localizer = lo_module.PM(
            mol,
            mo_block,
            pop_method=normalize_pm_pop_method(pm_pop_method),
        )
        localizer.conv_tol = pm_conv_tol
        if pm_conv_tol_grad is not None:
            localizer.conv_tol_grad = pm_conv_tol_grad
        localizer.max_cycle = pm_max_cycle
        localizer.exponent = pm_exponent
        localizer.init_guess = pm_init_guess
        return localizer.kernel()
    if method == "boys":
        localizer = lo_module.Boys(mol, mo_block)
        localizer.conv_tol = boys_conv_tol
        if boys_conv_tol_grad is not None:
            localizer.conv_tol_grad = boys_conv_tol_grad
        localizer.max_cycle = boys_max_cycle
        return localizer.kernel()
    raise ValueError(f"Unsupported localization method: {method}")


def localize_orbitals_with_params(mol, mo_coeff_block, method="boys", loc_params=None):
    """CAS-style localization wrapper using a simple params dict."""
    params = dict(loc_params or {})
    return localize_orbital_block(
        mol,
        mo_coeff_block,
        method=method,
        lo_module=__import__("pyscf.lo", fromlist=["lo"]),
        pm_pop_method=normalize_pm_pop_method(params.get("pop_method", "mulliken")),
        pm_conv_tol=params.get("conv_tol", 1e-6),
        pm_conv_tol_grad=params.get("conv_tol_grad"),
        pm_max_cycle=params.get("max_cycle", 100),
        pm_exponent=params.get("exponent", 2),
        pm_init_guess=params.get("init_guess", "atomic"),
        boys_conv_tol=params.get("conv_tol", 1e-6),
        boys_conv_tol_grad=params.get("conv_tol_grad"),
        boys_max_cycle=params.get("max_cycle", 100),
    )


def build_localization_params_from_settings(settings, localization_method: str):
    """Build localization keyword parameters from a settings object."""
    method = str(localization_method).strip().lower()
    if method == "pm":
        loc_params = {
            "pop_method": normalize_pm_pop_method(settings.pm_pop_method),
            "conv_tol": settings.pm_conv_tol,
            "max_cycle": settings.pm_max_cycle,
            "exponent": settings.pm_exponent,
        }
        if settings.pm_conv_tol_grad is not None:
            loc_params["conv_tol_grad"] = settings.pm_conv_tol_grad
        init_guess_val = settings.pm_init_guess
        if init_guess_val is not None and init_guess_val != "atomic":
            loc_params["init_guess"] = init_guess_val
        return loc_params
    if method == "boys":
        loc_params = {
            "conv_tol": settings.boys_conv_tol,
            "max_cycle": settings.boys_max_cycle,
        }
        if settings.boys_conv_tol_grad is not None:
            loc_params["conv_tol_grad"] = settings.boys_conv_tol_grad
        return loc_params
    return None


def split_localize_by_occupations(
    mol,
    mo_coeff,
    occupations,
    *,
    occ_threshold_core: float = 1.98,
    occ_threshold_virtual: float = 0.02,
    method: str = "boys",
    loc_params=None,
    merge_core_active: bool = False,
):
    """Split-localize orbitals by occupation blocks and return base labels."""
    n_orb = len(occupations)

    core_idx = [i for i in range(n_orb) if occupations[i] > occ_threshold_core]
    active_idx = [
        i
        for i in range(n_orb)
        if occ_threshold_virtual <= occupations[i] <= occ_threshold_core
    ]
    virtual_idx = [i for i in range(n_orb) if occupations[i] < occ_threshold_virtual]

    localized = mo_coeff.copy()
    labels = [""] * n_orb

    if merge_core_active:
        blocks = [
            ("core_active", core_idx + active_idx),
            ("virtual", virtual_idx),
        ]
    else:
        blocks = [
            ("core", core_idx),
            ("active", active_idx),
            ("virtual", virtual_idx),
        ]

    for block_name, block_idx in blocks:
        if len(block_idx) <= 1:
            for i in block_idx:
                labels[i] = f"{block_name}_{i}"
            continue

        mo_block = mo_coeff[:, block_idx]
        loc_block = localize_orbitals_with_params(
            mol, mo_block, method=method, loc_params=loc_params
        )
        localized[:, block_idx] = loc_block

        for k, i in enumerate(block_idx):
            labels[i] = f"{block_name}_{k}"

    return localized, labels
