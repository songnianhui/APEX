"""Module 7: Energy Extrapolation

Estimate exact energies from finite-accuracy DMRG/CC data using the
extrapolation schemes from Zhai et al. 2026.

Methods:
1. DMRG D-extrapolation: E(D) = E_inf + A * exp[-kappa * (ln D)^2]
2. CC composite: E = E_CCSDT(full) + [E_CCSDTQ(FNO) - E_CCSDT(FNO)]
3. FNO threshold extrapolation: polynomial fit of E_corr vs cutoff
4. Correlation increment ratio: transfer ratios from reference compounds
5. MP2 space correction: E(large CAS) ≈ E(small CAS) + delta_MP2
"""

import numpy as np
from scipy.optimize import curve_fit

from .models import ExtrapolatedEnergy


# ──────────────────────────────────────────────────────────────────
# Method 1: DMRG bond dimension extrapolation
# ──────────────────────────────────────────────────────────────────

def dmrg_d_extrapolation(bond_dims: list, energies: list) -> ExtrapolatedEnergy:
    """Extrapolate DMRG energies to infinite bond dimension.

    Fit model: E(D) = E_inf + A * exp(-kappa * (ln D)^2)

    Args:
        bond_dims: List of bond dimensions used (e.g., [500, 1000, 2000, 5000]).
        energies: Corresponding total energies in Hartrees.

    Returns:
        ExtrapolatedEnergy with E_inf and fit parameters.
    """
    D = np.array(bond_dims, dtype=float)
    E = np.array(energies, dtype=float)

    if len(D) < 2:
        return ExtrapolatedEnergy(
            method="DMRG_D_extrapolation",
            energy=E[0] if len(E) > 0 else 0.0,
            uncertainty=float("inf"),
            description="Insufficient data for extrapolation",
        )

    def model(d, e_inf, A, kappa):
        return e_inf + A * np.exp(-kappa * np.log(d) ** 2)

    # Initial guess
    e_inf_guess = E[-1]  # best energy as starting guess
    A_guess = E[0] - E[-1]
    kappa_guess = 0.1

    try:
        popt, pcov = curve_fit(
            model, D, E,
            p0=[e_inf_guess, A_guess, kappa_guess],
            maxfev=10000,
        )
        e_inf = popt[0]
        perr = np.sqrt(np.diag(pcov))
        uncertainty = perr[0] if len(perr) > 0 else 0.0

        return ExtrapolatedEnergy(
            method="DMRG_D_extrapolation",
            energy=e_inf,
            uncertainty=uncertainty,
            fit_params={
                "E_inf": popt[0],
                "A": popt[1],
                "kappa": popt[2],
                "bond_dims": bond_dims,
                "energies": energies,
                "residuals": (E - model(D, *popt)).tolist(),
            },
            description=f"DMRG D→∞ extrapolation: E_inf = {e_inf:.10f} ± {uncertainty:.2e} Ha",
        )
    except Exception as e:
        # Fallback: use last two points for linear extrapolation
        return _fallback_dmrg_extrapolation(D, E, str(e))


def _fallback_dmrg_extrapolation(D, E, error_msg):
    """Linear fallback for DMRG extrapolation when curve_fit fails."""
    if len(D) >= 2:
        # Use last 2 points: linear in 1/D
        x = 1.0 / D[-2:]
        y = E[-2:]
        # Extrapolate to 1/D = 0
        slope = (y[1] - y[0]) / (x[1] - x[0])
        e_inf = y[-1] - slope * x[-1]
    else:
        e_inf = E[-1]

    return ExtrapolatedEnergy(
        method="DMRG_D_extrapolation_linear_fallback",
        energy=e_inf,
        uncertainty=abs(E[-1] - E[-2]) if len(E) >= 2 else float("inf"),
        fit_params={"note": f"Curve fit failed: {error_msg}. Used linear 1/D extrapolation."},
        description=f"DMRG D→∞ (linear fallback): E_inf = {e_inf:.10f} Ha",
    )


# ──────────────────────────────────────────────────────────────────
# Method 2: CC composite energy
# ──────────────────────────────────────────────────────────────────

def cc_composite_energy(e_ccsdt_full: float,
                         e_ccsdtq_fno: float,
                         e_ccsdt_fno: float) -> ExtrapolatedEnergy:
    """Combine full-space CCSDT with FNO-CCSDTQ correction.

    E = E_CCSDT(full) + [E_CCSDTQ(FNO) - E_CCSDT(FNO)]

    The FNO (frozen natural orbital) calculation uses a truncated virtual
    space but higher excitation level. The difference captures the effect
    of quadruples.

    Args:
        e_ccsdt_full: CCSDT energy in full virtual space.
        e_ccsdtq_fno: CCSDTQ energy in FNO-truncated space.
        e_ccsdt_fno: CCSDT energy in same FNO-truncated space.

    Returns:
        ExtrapolatedEnergy with the composite result.
    """
    delta_q = e_ccsdtq_fno - e_ccsdt_fno
    e_composite = e_ccsdt_full + delta_q

    return ExtrapolatedEnergy(
        method="CC_composite",
        energy=e_composite,
        uncertainty=abs(delta_q) * 0.1,  # rough 10% error estimate
        fit_params={
            "E_CCSDT_full": e_ccsdt_full,
            "E_CCSDTQ_FNO": e_ccsdtq_fno,
            "E_CCSDT_FNO": e_ccsdt_fno,
            "delta_TQ": delta_q,
        },
        description=f"CC composite: E = {e_composite:.10f} Ha (delta_TQ = {delta_q:.6f} Ha)",
    )


# ──────────────────────────────────────────────────────────────────
# Method 3: FNO threshold extrapolation
# ──────────────────────────────────────────────────────────────────

def fno_extrapolation(thresholds: list,
                       corr_energies: list,
                       degree: int = 2) -> ExtrapolatedEnergy:
    """Extrapolate correlation energy to zero FNO threshold.

    As the FNO occupation threshold → 0, more virtual orbitals are
    included and E_corr approaches the full-virtual-space limit.

    Args:
        thresholds: FNO occupation thresholds (e.g., [1e-3, 1e-4, 1e-5]).
        corr_energies: Corresponding correlation energies.
        degree: Polynomial degree for fitting (default 2).

    Returns:
        ExtrapolatedEnergy at threshold = 0.
    """
    t = np.array(thresholds, dtype=float)
    E_corr = np.array(corr_energies, dtype=float)

    if len(t) < degree + 1:
        return ExtrapolatedEnergy(
            method="FNO_extrapolation",
            energy=E_corr[-1] if len(E_corr) > 0 else 0.0,
            uncertainty=float("inf"),
            description="Insufficient data for FNO extrapolation",
        )

    # Fit polynomial in threshold
    try:
        coeffs = np.polyfit(t, E_corr, degree)
        e_extrap = np.polyval(coeffs, 0.0)  # evaluate at threshold = 0

        # Estimate uncertainty from fit residuals
        E_fit = np.polyval(coeffs, t)
        residuals = E_corr - E_fit
        rmse = np.sqrt(np.mean(residuals ** 2))

        return ExtrapolatedEnergy(
            method="FNO_extrapolation",
            energy=e_extrap,
            uncertainty=rmse,
            fit_params={
                "coefficients": coeffs.tolist(),
                "thresholds": thresholds,
                "corr_energies": corr_energies,
                "degree": degree,
            },
            description=f"FNO extrapolation (degree {degree}): E_corr(0) = {e_extrap:.10f} ± {rmse:.2e}",
        )
    except Exception as e:
        return ExtrapolatedEnergy(
            method="FNO_extrapolation",
            energy=E_corr[-1],
            uncertainty=abs(E_corr[-1] - E_corr[-2]) if len(E_corr) >= 2 else float("inf"),
            fit_params={"error": str(e)},
            description=f"FNO extrapolation failed: {e}",
        )


# ──────────────────────────────────────────────────────────────────
# Method 4: Correlation increment ratio
# ──────────────────────────────────────────────────────────────────

def correlation_increment_ratio(e_target_low: float,
                                 e_ref_low: float,
                                 e_ref_high: float,
                                 e_ref_exact: float = None) -> ExtrapolatedEnergy:
    """Transfer correlation ratios from a reference compound.

    E_target(high) ≈ E_target(low) × [E_ref(high) / E_ref(low)]

    Used when a reference system (e.g., Fe2S2 dimer) is calculable at
    both low and high levels, and we assume the same ratio applies
    to the target.

    Args:
        e_target_low: Target system energy at low level of theory.
        e_ref_low: Reference system energy at same low level.
        e_ref_high: Reference system energy at high level.
        e_ref_exact: Optional exact reference energy (for error estimate).

    Returns:
        ExtrapolatedEnergy with the ratio-transferred result.
    """
    if abs(e_ref_low) < 1e-15:
        ratio = 1.0
    else:
        ratio = e_ref_high / e_ref_low

    e_target_high = e_target_low * ratio

    # Estimate uncertainty
    if e_ref_exact is not None:
        ref_error = abs(e_ref_high - e_ref_exact)
        uncertainty = abs(e_target_low) * ref_error / abs(e_ref_low) if abs(e_ref_low) > 1e-15 else float("inf")
    else:
        uncertainty = abs(e_target_high - e_target_low) * 0.1

    return ExtrapolatedEnergy(
        method="correlation_increment_ratio",
        energy=e_target_high,
        uncertainty=uncertainty,
        fit_params={
            "E_target_low": e_target_low,
            "E_ref_low": e_ref_low,
            "E_ref_high": e_ref_high,
            "ratio": ratio,
        },
        description=f"Correlation increment ratio: E_target(high) = {e_target_high:.10f} Ha (ratio={ratio:.6f})",
    )


# ──────────────────────────────────────────────────────────────────
# Method 5: MP2 space correction
# ──────────────────────────────────────────────────────────────────

def mp2_space_correction(e_small_cas: float,
                          e_mp2_small: float,
                          e_mp2_large: float) -> ExtrapolatedEnergy:
    """Estimate large-CAS energy from small-CAS + MP2 correction.

    E(large CAS) ≈ E(small CAS) + [E_MP2(large) - E_MP2(small)]

    The idea is that MP2 captures the orbital relaxation effect of
    including more orbitals, and this correction can be transferred
    to a higher-level calculation in the smaller space.

    Args:
        e_small_cas: High-level energy in small active space.
        e_mp2_small: MP2 energy in same small active space.
        e_mp2_large: MP2 energy in large active space.

    Returns:
        ExtrapolatedEnergy with the corrected estimate.
    """
    delta_mp2 = e_mp2_large - e_mp2_small
    e_corrected = e_small_cas + delta_mp2

    return ExtrapolatedEnergy(
        method="MP2_space_correction",
        energy=e_corrected,
        uncertainty=abs(delta_mp2) * 0.2,  # 20% of correction as error estimate
        fit_params={
            "E_small_CAS": e_small_cas,
            "E_MP2_small": e_mp2_small,
            "E_MP2_large": e_mp2_large,
            "delta_MP2": delta_mp2,
        },
        description=f"MP2 space correction: E = {e_corrected:.10f} Ha (delta_MP2 = {delta_mp2:.6f})",
    )
