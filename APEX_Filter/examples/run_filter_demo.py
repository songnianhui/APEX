#!/usr/bin/env python3
"""APEX_Filter Demo — Spin Isomer Enumeration and Filtering Funnel Design.

This script demonstrates the typical workflow of APEX_Filter:
  1. Build a minimal ClusterInfo from metal elements
  2. Enumerate spin isomers
  3. Design a filtering funnel
  4. Print results

Usage:
    python run_filter_demo.py
"""

import numpy as np

from apex_filter.models import (
    MetalCenter,
    ClusterInfo,
    CAS,
)
from apex_filter.spin_config import enumerate_spin_isomers
from apex_filter.filtering import design_filtering_funnel


def main():
    # -- Step 1: Build a minimal Fe4S4 cluster --
    metals = [
        MetalCenter(element="Fe", index=i, position=np.zeros(3), label=f"Fe{i+1}")
        for i in range(4)
    ]

    # Add bridging S atoms (for context, not strictly needed for spin enumeration)
    cluster_info = ClusterInfo(
        metals=metals,
        formula="Fe4S4",
        total_charge=-2,
        target_spin=0.0,
        symmetry_group="D2d",
    )

    print(f"Cluster: {cluster_info.formula}")
    print(f"  Metals: {[m.label for m in cluster_info.metals]}")
    print(f"  Charge: {cluster_info.total_charge}")
    print(f"  Target S: {cluster_info.target_spin}")
    print(f"  Symmetry: {cluster_info.symmetry_group}")

    # -- Step 2: Enumerate spin isomers --
    isomers = enumerate_spin_isomers(cluster_info)
    print(f"\nSpin isomers: {len(isomers)}")
    for iso in isomers[:5]:
        minority = [k + 1 for k, v in iso.spin_assignment.items() if v == -1]
        print(f"  {iso.label}: minority at sites {minority}")
    if len(isomers) > 5:
        print(f"  ... and {len(isomers) - 5} more")

    # -- Step 3: Design filtering funnel --
    active_space = CAS(n_electrons=54, n_orbitals=36)
    n_configs = len(isomers) * 5  # rough estimate

    plan = design_filtering_funnel(
        n_configs, active_space, len(isomers), style="femoco"
    )

    print(f"\nFiltering funnel for ({active_space.n_electrons}e, {active_space.n_orbitals}o):")
    print(f"  Total configs: {plan.total_configs}")
    for level in plan.levels:
        print(f"  {level.method:>10}: {level.n_input:>6} -> {level.n_output:>6}"
              f"  ({level.selection_criterion})")

    print("\nDone.")


if __name__ == "__main__":
    main()
