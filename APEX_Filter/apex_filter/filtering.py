"""Module 5: Filtering Protocol

Implement the hierarchical filtering funnel for selecting the most promising
electronic configurations at increasing levels of theory.

Reference funnel (FeMo-co, Zhai et al. 2026):
    78,750 UHF guesses
    → 840 UHF solutions (24 per spin isomer)
    → ~456 CCSD calculations (12-24 per spin isomer)
    → 35 DMRG (D=5000)
    → 3 CCSDTQ
    → 2 DMRG (D=18000)
"""

from .models import (
    CAS,
    CalculationResult,
    ElectronicConfig,
    FilteringLevel,
    FilteringPlan,
)


# ──────────────────────────────────────────────────────────────────
# Filtering plan design
# ──────────────────────────────────────────────────────────────────

def design_filtering_funnel(n_configs: int,
                             active_space: CAS,
                             n_spin_isomers: int = None,
                             style: str = "femoco") -> FilteringPlan:
    """Design an adaptive filtering funnel.

    Args:
        n_configs: Total number of initial configurations.
        active_space: Active space specification.
        n_spin_isomers: Number of spin isomers (for per-isomer allocation).
        style: "femoco" (aggressive), "conservative" (keep more), or "minimal".

    Returns:
        FilteringPlan with recommended filtering levels.
    """
    n = active_space.n_orbitals

    # Ensure n_spin_isomers >= 1 so per-isomer math doesn't collapse to 0
    if n_spin_isomers is not None:
        n_spin_isomers = max(1, n_spin_isomers)

    if style == "femoco":
        # Follow the FeMo-co protocol: aggressive filtering
        levels = _femoco_style_funnel(n_configs, n_spin_isomers, n)
    elif style == "conservative":
        levels = _conservative_funnel(n_configs, n_spin_isomers, n)
    elif style == "minimal":
        levels = _minimal_funnel(n_configs, n)
    else:
        levels = _adaptive_funnel(n_configs, n_spin_isomers, n)

    return FilteringPlan(
        levels=levels,
        total_configs=n_configs,
        active_space=active_space,
        description=f"Filtering funnel: {n_configs} → {levels[-1].n_output if levels else 0}",
    )


def _femoco_style_funnel(n_configs, n_isomers, n_orbitals):
    """Aggressive funnel following the FeMo-co protocol."""
    if n_isomers is None:
        n_isomers = max(1, n_configs // 2250)  # estimate

    # Level 1: UHF screening
    n_uhf_keep = min(n_configs, n_isomers * 24)
    l1 = FilteringLevel(
        method="UHF",
        n_input=n_configs,
        n_output=n_uhf_keep,
        selection_criterion="energy",
        n_per_isomer=24,
        params={"basis": "def2-SVP", "max_cycle": 200},
    )

    # Level 2: UCCSD
    n_ccsd_keep = min(n_uhf_keep, n_isomers * 12)
    l2 = FilteringLevel(
        method="UCCSD",
        n_input=n_uhf_keep,
        n_output=n_ccsd_keep,
        selection_criterion="energy",
        n_per_isomer=12,
        params={"basis": "cc-pVDZ"},
    )

    # Level 3: UCCSD(T)
    n_ccsdt_keep = min(n_ccsd_keep, n_isomers * 1)
    l3 = FilteringLevel(
        method="UCCSD(T)",
        n_input=n_ccsd_keep,
        n_output=n_ccsdt_keep,
        selection_criterion="energy",
        n_per_isomer=1,
        params={"basis": "cc-pVDZ"},
    )

    # Level 4: DMRG (intermediate bond dimension)
    n_dmrg_keep = min(n_ccsdt_keep, max(5, n_isomers // 3))
    l4 = FilteringLevel(
        method="DMRG",
        n_input=n_ccsdt_keep,
        n_output=n_dmrg_keep,
        selection_criterion="energy",
        n_per_isomer=0,
        params={"bond_dim": 5000, "sweeps": 10},
    )

    # Level 5: CCSDTQ (on FNO-truncated space)
    n_ccsdtq_keep = min(n_dmrg_keep, 3)
    l5 = FilteringLevel(
        method="CCSDTQ",
        n_input=n_dmrg_keep,
        n_output=n_ccsdtq_keep,
        selection_criterion="energy",
        n_per_isomer=0,
        params={"basis": "cc-pVDZ", "fno_threshold": 1e-4},
    )

    # Level 6: DMRG (large bond dimension)
    l6 = FilteringLevel(
        method="DMRG",
        n_input=n_ccsdtq_keep,
        n_output=min(n_ccsdtq_keep, 2),
        selection_criterion="energy",
        n_per_isomer=0,
        params={"bond_dim": 18000, "sweeps": 20},
    )

    return [l1, l2, l3, l4, l5, l6]


def _conservative_funnel(n_configs, n_isomers, n_orbitals):
    """Keep more configurations at each level."""
    if n_isomers is None:
        n_isomers = max(1, n_configs // 500)

    l1 = FilteringLevel(
        method="UHF", n_input=n_configs,
        n_output=min(n_configs, n_isomers * 48),
        selection_criterion="energy", n_per_isomer=48,
    )
    l2 = FilteringLevel(
        method="UCCSD", n_input=l1.n_output,
        n_output=min(l1.n_output, n_isomers * 24),
        selection_criterion="energy", n_per_isomer=24,
    )
    l3 = FilteringLevel(
        method="UCCSD(T)", n_input=l2.n_output,
        n_output=min(l2.n_output, n_isomers * 3),
        selection_criterion="energy", n_per_isomer=3,
    )
    l4 = FilteringLevel(
        method="DMRG", n_input=l3.n_output,
        n_output=max(5, l3.n_output // 5),
        selection_criterion="energy",
        params={"bond_dim": 5000},
    )

    return [l1, l2, l3, l4]


def _minimal_funnel(n_configs, n_orbitals):
    """Minimal funnel: UHF → CCSD → final."""
    l1 = FilteringLevel(
        method="UHF", n_input=n_configs,
        n_output=min(n_configs, max(20, n_configs // 10)),
        selection_criterion="energy",
    )
    l2 = FilteringLevel(
        method="UCCSD", n_input=l1.n_output,
        n_output=min(l1.n_output, max(5, l1.n_output // 5)),
        selection_criterion="energy",
    )
    return [l1, l2]


def _adaptive_funnel(n_configs, n_isomers, n_orbitals):
    """Adaptively determine funnel levels based on system size."""
    if n_configs < 100:
        return _minimal_funnel(n_configs, n_orbitals)
    elif n_configs < 10000:
        return _conservative_funnel(n_configs, n_isomers, n_orbitals)
    else:
        return _femoco_style_funnel(n_configs, n_isomers, n_orbitals)


# ──────────────────────────────────────────────────────────────────
# Selection functions
# ──────────────────────────────────────────────────────────────────

def select_from_uhf(uhf_results: list[CalculationResult],
                     n_per_isomer: int = 24,
                     n_total: int = None) -> list[CalculationResult]:
    """Select configurations from UHF results.

    Selection strategy: keep n_per_isomer lowest-energy results per
    spin isomer family. Optionally cap total number.

    Args:
        uhf_results: List of UHF calculation results.
        n_per_isomer: Number to keep per spin isomer.
        n_total: Optional total cap.

    Returns:
        Selected results.
    """
    return _select_by_energy_per_group(
        uhf_results, n_per_isomer, n_total,
        group_key=lambda r: r.config.spin_isomer.family if r.config and r.config.spin_isomer else ""
    )


def select_from_ccsd(ccsd_results: list[CalculationResult],
                      n_per_isomer: int = 12,
                      n_total: int = None) -> list[CalculationResult]:
    """Select configurations from CCSD results."""
    return _select_by_energy_per_group(
        ccsd_results, n_per_isomer, n_total,
        group_key=lambda r: r.config.spin_isomer.family if r.config and r.config.spin_isomer else ""
    )


def select_from_ccsdt(ccsdt_results: list[CalculationResult],
                       n_per_isomer: int = 1,
                       n_total: int = None) -> list[CalculationResult]:
    """Select configurations from CCSD(T) results."""
    return _select_by_energy_per_group(
        ccsdt_results, n_per_isomer, n_total,
        group_key=lambda r: r.config.spin_isomer.family if r.config and r.config.spin_isomer else ""
    )


def select_lowest_energy(results: list[CalculationResult],
                          n_keep: int) -> list[CalculationResult]:
    """Select the n_keep lowest-energy converged results."""
    converged = [r for r in results if r.converged]
    converged.sort(key=lambda r: r.energy)
    return converged[:n_keep]


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _select_by_energy_per_group(results, n_per_group, n_total, group_key):
    """Select n_per_group lowest-energy results per group."""
    converged = [r for r in results if r.converged]

    # Group by spin isomer family
    groups = {}
    for r in converged:
        key = group_key(r)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    selected = []
    for key in groups:
        group = groups[key]
        group.sort(key=lambda r: r.energy)
        selected.extend(group[:n_per_group])

    # Sort all selected by energy
    selected.sort(key=lambda r: r.energy)

    # Apply total cap
    if n_total and len(selected) > n_total:
        selected = selected[:n_total]

    return selected
