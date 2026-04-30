"""Enumeration steps for the interactive APEX_Filter pipeline."""

import copy
import os

from .session import SessionManager
from .selection_guidance import write_selection_artifacts


_ENUMERATE_DEFAULTS = {
    "target_sz": None,
    "forced_oxidation": None,
    "max_configs": None,
}


def step_enumerate(session_dir: str, *, target_Sz=None, forced_oxidation=None, max_configs=None):
    """Enumerate spin isomers and electronic configurations."""
    sm = SessionManager(session_dir)
    sm.require_previous("step2_enumerate", "step1_load")
    controls = sm.resolve_method_controls(
        "enumerate",
        _ENUMERATE_DEFAULTS,
        {
            "target_sz": target_Sz,
            "forced_oxidation": forced_oxidation,
            "max_configs": max_configs,
        },
    )
    target_Sz = controls["target_sz"]
    forced_oxidation = controls["forced_oxidation"]
    max_configs = controls["max_configs"]
    state = sm.load_load_state()

    ci = state["cluster_info"]

    print("=" * 60)
    print("Step 2: Enumerating spin isomers & electronic configurations")
    print("=" * 60)

    if target_Sz is None:
        target_Sz = ci.target_spin
    from .elec_spin_config_generator import (
        canonicalize_config_spin_labels,
        generate_all_configs,
        reduce_configs_by_symmetry,
        summarize_enumeration_layers,
    )

    raw_configs = generate_all_configs(
        ci,
        target_Sz=target_Sz,
        max_configs=max_configs,
        forced_oxidation=forced_oxidation,
    )
    raw_configs_for_stats = copy.deepcopy(raw_configs)
    raw_configs, spin_isomers, families = canonicalize_config_spin_labels(raw_configs, ci)
    configs = reduce_configs_by_symmetry(raw_configs, ci)
    n_total = len(configs)
    enum_stats = summarize_enumeration_layers(raw_configs_for_stats, configs, spin_isomers, families)
    enum_stats["family_scheme"] = getattr(ci, "family_scheme", "") or ""
    enum_stats["benchmark_profile"] = getattr(ci, "benchmark_profile", "") or ""
    enum_stats["config_reduction_mode"] = getattr(ci, "config_reduction_mode", "none") or "none"

    print("\n  Enumeration layers")
    if enum_stats["family_scheme"]:
        print(f"    Family scheme                : {enum_stats['family_scheme']}")
    if enum_stats["benchmark_profile"]:
        print(f"    Benchmark profile            : {enum_stats['benchmark_profile']}")
    print(f"    Config reduction mode        : {enum_stats['config_reduction_mode']}")
    print(f"    Raw spin patterns            : {enum_stats['raw_spin_patterns']}")
    print(f"    Spin families                : {enum_stats['spin_families']}")
    print(f"    Spin x oxidation guesses     : {enum_stats['spin_x_oxidation']}")
    print(
        "    Spin x oxidation x d guesses : "
        f"{enum_stats['spin_x_oxidation_x_d_before_reduction']}"
    )
    print(f"    Total configs (saved)        : {enum_stats['total_configs_after_reduction']}")

    print(f"\n  Reported spin isomers   : {len(spin_isomers)}")
    print(f"  Reported families       : {len(families)}")

    family_counts = {}
    for cfg in configs:
        fam = cfg.spin_isomer.family if cfg.spin_isomer else "N/A"
        family_counts[fam] = family_counts.get(fam, 0) + 1
    print("\n  Per-family breakdown:")
    for fam_label, count in sorted(family_counts.items()):
        print(f"    {fam_label}: {count}")

    ox_counts = {}
    for cfg in configs:
        ox_desc = cfg.oxidation.description if cfg.oxidation else "N/A"
        ox_counts[ox_desc] = ox_counts.get(ox_desc, 0) + 1
    print("\n  Per-oxidation breakdown:")
    for ox_desc, count in sorted(ox_counts.items()):
        print(f"    {ox_desc}: {count}")

    sm.save_enumeration(configs, spin_isomers, families, n_total, enum_stats)
    selection_rows = [
        {
            "label": cfg.label,
            "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
            "energy": None,
            "converged": None,
            "oxidation": cfg.oxidation.description if cfg.oxidation else "",
            "config_id": cfg.config_id,
        }
        for cfg in configs
    ]
    write_selection_artifacts(
        os.path.join(sm.session_dir, "step2_enumerate"),
        step_name="Step 2 enumerate",
        next_step_name="uhf",
        summary=selection_rows,
        stats=enum_stats,
        keep_default="1",
    )
    with open(os.path.join(sm.session_dir, "step2_enumerate", "enumeration_layers.json"), "w", encoding="utf-8") as fh:
        import json

        json.dump(enum_stats, fh, indent=2, ensure_ascii=False)
    print(f"\nStep 2 complete. {n_total} configurations saved to session.")
