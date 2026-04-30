"""Report Generator

Generate markdown and JSON analysis reports from the pipeline results.
"""

import json
import os
from datetime import datetime

import numpy as np

from . import __version__
from .models import (
    CAS,
    ClusterInfo,
    ExtrapolatedEnergy,
    FilteringPlan,
    SpinIsomer,
    SpinIsomerFamily,
)


def generate_report(cluster_info: ClusterInfo,
                     active_space: CAS,
                     spin_families: list = None,
                     spin_isomers: list = None,
                     n_electronic_configs: int = 0,
                     filtering_plan: FilteringPlan = None,
                     results: list = None,
                     extrapolated: list = None,
                     output_format: str = "markdown") -> str:
    """Generate an analysis report.

    Args:
        cluster_info: Cluster description.
        active_space: Active space specification.
        spin_families: Spin isomer families.
        spin_isomers: All spin isomers.
        n_electronic_configs: Total number of electronic configurations.
        filtering_plan: Filtering funnel plan.
        results: List of CalculationResult objects.
        extrapolated: List of ExtrapolatedEnergy objects.
        output_format: "markdown" or "json".

    Returns:
        Report content as string.
    """
    if output_format == "json":
        return _generate_json_report(
            cluster_info, active_space, spin_families, spin_isomers,
            n_electronic_configs, filtering_plan, results, extrapolated,
        )
    else:
        return _generate_markdown_report(
            cluster_info, active_space, spin_families, spin_isomers,
            n_electronic_configs, filtering_plan, results, extrapolated,
        )


def _generate_markdown_report(cluster_info, active_space, spin_families,
                               spin_isomers, n_electronic_configs,
                               filtering_plan, results, extrapolated):
    """Generate a markdown report."""
    lines = [
        "# APEX Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## 1. Cluster Structure",
        "",
        f"- **Formula**: {cluster_info.formula}",
        f"- **Total charge**: {cluster_info.total_charge}",
        f"- **Target spin**: S = {cluster_info.target_spin}",
        f"- **Symmetry**: {cluster_info.symmetry_group}",
        "",
        "### Metal Centers",
        "",
        "| # | Element | Label | Coordination |",
        "|---|---------|-------|-------------|",
    ]

    for k, metal in enumerate(cluster_info.metals):
        lines.append(
            f"| {k + 1} | {metal.element} | {metal.label} | {metal.coordination} |"
        )

    lines.extend([
        "",
        "### Bridging Atoms",
        "",
        "| # | Element | Role | Bridges |",
        "|---|---------|------|---------|",
    ])

    for k, bridge in enumerate(cluster_info.bridging_atoms):
        bridged = ", ".join(str(m + 1) for m in bridge.bridged_metals)
        lines.append(
            f"| {k + 1} | {bridge.element} | {bridge.role} | {bridged} |"
        )

    # Active space
    lines.extend([
        "",
        "---",
        "",
        "## 2. Active Space",
        "",
        f"- **Level**: {active_space.level.value}",
        f"- **Size**: ({active_space.n_electrons}e, {active_space.n_orbitals}o)",
        f"- **Qubits**: {active_space.n_qubits}",
        f"- **Description**: {active_space.description}",
        "",
        "### Orbital Groups",
        "",
        "| Group | Atom | Type | Orbitals | Electrons |",
        "|-------|------|------|----------|-----------|",
    ])

    for og in active_space.orbital_groups:
        lines.append(
            f"| | {og.atom_label} | {og.orbital_type} | {og.n_orbitals} | {og.n_electrons} |"
        )

    # Spin isomers
    if spin_isomers is not None:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Spin Isomers",
            "",
            f"- **Total spin isomers**: {len(spin_isomers)}",
        ])

        if spin_families:
            lines.append(f"- **Symmetry families**: {len(spin_families)}")
            lines.append("")
            lines.append("### Spin Isomer Families")
            lines.append("")
            lines.append("| Family | N(minority) | N(isomers) |")
            lines.append("|--------|-------------|------------|")
            for fam in spin_families:
                lines.append(
                    f"| {fam.label} | {fam.n_minority} | {len(fam.isomers)} |"
                )

    # Electronic configurations
    if n_electronic_configs > 0:
        lines.extend([
            "",
            "---",
            "",
            "## 4. Electronic Configurations",
            "",
            f"- **Total configurations**: {n_electronic_configs:,}",
        ])

    # Filtering plan
    if filtering_plan:
        lines.extend([
            "",
            "---",
            "",
            "## 5. Filtering Funnel",
            "",
            "| Level | Method | Input | Output | Criterion |",
            "|-------|--------|-------|--------|-----------|",
        ])
        for level in filtering_plan.levels:
            lines.append(
                f"| | {level.method} | {level.n_input:,} | {level.n_output:,} | {level.selection_criterion} |"
            )

    # Results
    if results:
        lines.extend([
            "",
            "---",
            "",
            "## 6. Calculation Results",
            "",
        ])
        converged = [r for r in results if r.converged]
        lines.append(f"- **Converged**: {len(converged)} / {len(results)}")

        if converged:
            best = min(converged, key=lambda r: r.energy)
            lines.append(f"- **Lowest energy**: {best.energy:.10f} Ha ({best.method})")
            lines.append(f"- **Best config**: {best.config.label if best.config else 'N/A'}")

    # Extrapolated energies
    if extrapolated:
        lines.extend([
            "",
            "---",
            "",
            "## 7. Extrapolated Energies",
            "",
            "| Method | Energy (Ha) | Uncertainty |",
            "|--------|-------------|-------------|",
        ])
        for ext in extrapolated:
            lines.append(
                f"| {ext.method} | {ext.energy:.10f} | ±{ext.uncertainty:.2e} |"
            )

    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by APEX v{__version__} | Song@Elab*",
    ])

    return "\n".join(lines)


def _generate_json_report(cluster_info, active_space, spin_families,
                           spin_isomers, n_electronic_configs,
                           filtering_plan, results, extrapolated):
    """Generate a JSON report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "cluster": {
            "formula": cluster_info.formula,
            "charge": cluster_info.total_charge,
            "target_spin": cluster_info.target_spin,
            "symmetry": cluster_info.symmetry_group,
            "n_metals": len(cluster_info.metals),
            "metals": [
                {"element": m.element, "label": m.label, "coordination": m.coordination}
                for m in cluster_info.metals
            ],
            "n_bridging_atoms": len(cluster_info.bridging_atoms),
        },
        "active_space": {
            "n_electrons": active_space.n_electrons,
            "n_orbitals": active_space.n_orbitals,
            "n_qubits": active_space.n_qubits,
            "level": active_space.level.value,
            "description": active_space.description,
        },
    }

    if spin_isomers is not None:
        report["spin_isomers"] = {
            "total": len(spin_isomers),
            "families": len(spin_families) if spin_families else 0,
        }

    report["electronic_configs"] = {"total": n_electronic_configs}

    if filtering_plan:
        report["filtering"] = {
            "total_configs": filtering_plan.total_configs,
            "levels": [
                {
                    "method": l.method,
                    "input": l.n_input,
                    "output": l.n_output,
                }
                for l in filtering_plan.levels
            ],
        }

    if results:
        report["results"] = {
            "total": len(results),
            "converged": sum(1 for r in results if r.converged),
            "best_energy": min(
                (r.energy for r in results if r.converged), default=None
            ),
        }

    if extrapolated:
        report["extrapolated"] = [
            {
                "method": e.method,
                "energy": e.energy,
                "uncertainty": e.uncertainty,
            }
            for e in extrapolated
        ]

    return json.dumps(report, indent=2, default=_json_serializer)


def save_report(content: str, filepath: str):
    """Save report to file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
