"""Pipeline: end-to-end execution of the progressive filtering funnel.

Orchestrates the cycle of:
  generate inputs -> execute -> parse results -> select -> next level
for each level in the :class:`FilteringPlan`.

Usage from CLI (``main.py``)::

    pipeline.run_pipeline(configs, active_space, cluster_info, plan,
                          workdir="run_output")
"""

import json
import os
import subprocess
import sys

import numpy as np

from .models import (
    CAS,
    CalculationResult,
    ClusterInfo,
    ElectronicConfig,
    FilteringPlan,
)
from .filtering import (
    select_from_ccsd,
    select_from_ccsdt,
    select_from_uhf,
    select_lowest_energy,
)
from .input_generator import generate_batch, generate_input
from .result_parser import parse_npz_result, to_calculation_result

# Maps FilteringLevel.method -> {input method, result suffix, select key, code override}
_METHOD_INFO = {
    "UHF": {"method": "uhf", "suffix": "_uhf.npz", "select": "uhf", "code": None},
    "UCCSD": {
        "method": "ccsd",
        "suffix": "_ccsd_results.npz",
        "select": "ccsd",
        "code": None,
    },
    "UCCSD(T)": {
        "method": "ccsdt",
        "suffix": "_ccsdt_results.npz",
        "select": "ccsdt",
        "code": None,
    },
    "DMRG": {
        "method": "dmrg",
        "suffix": "_dmrg_results.npz",
        "select": None,
        "code": "block2",
    },
    "CCSDTQ": {
        "method": "ccsdtq",
        "suffix": "_hast_ucc_results.npz",
        "select": None,
        "code": "hast_ucc",
    },
}


def _select(results, select_key, n_per_isomer, n_total):
    """Dispatch to the correct ``select_from_*`` function."""
    if select_key == "uhf":
        return select_from_uhf(results, n_per_isomer=n_per_isomer, n_total=n_total)
    elif select_key == "ccsd":
        return select_from_ccsd(results, n_per_isomer=n_per_isomer, n_total=n_total)
    elif select_key == "ccsdt":
        return select_from_ccsdt(results, n_per_isomer=n_per_isomer, n_total=n_total)
    else:
        return select_lowest_energy(results, n_keep=n_total or len(results))


# ──────────────────────────────────────────────────────────────────
# Core execution helpers
# ──────────────────────────────────────────────────────────────────


def _sanitize_label(cfg):
    """Return a filesystem-safe label from a config."""
    return cfg.label.replace("|", "_").replace(" ", "_")


def _execute_scripts(script_paths, workdir):
    """Run each Python script via subprocess, returning per-script outcome.

    Scripts run with cwd=workdir (the level directory) so output files
    are written there. The workdir is also passed so scripts can find
    inputs from previous levels via absolute paths.
    """
    outcomes = []
    for script in script_paths:
        script_abs = os.path.abspath(os.path.join(workdir, script))
        try:
            proc = subprocess.run(
                [sys.executable, script_abs],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            outcomes.append((script, proc.returncode == 0, proc))
        except subprocess.TimeoutExpired:
            outcomes.append((script, False, None))
        except Exception:
            outcomes.append((script, False, None))
    return outcomes


def _parse_results(configs, workdir, suffix):
    """Parse ``.npz`` result files for a batch of configs.

    Returns a list of :class:`CalculationResult` -- one per config whose
    result file exists and is readable.
    """
    results = []
    for cfg in configs:
        label = _sanitize_label(cfg)
        npz_path = os.path.join(workdir, f"{label}{suffix}")
        if not os.path.exists(npz_path):
            continue
        try:
            parsed = parse_npz_result(npz_path)
            results.append(to_calculation_result(parsed, config=cfg))
        except Exception:
            continue
    return results


def _generate_ccsd_inputs(
    configs,
    active_space,
    cluster_info,
    code,
    input_method,
    level_dir,
    prev_level_dir,
    prev_suffix,
    uhf_level_dir=None,
    ccsd_level_dir=None,
    **kwargs,
):
    """Generate CCSD inputs, one per config, injecting the UHF .npz path.

    The CCSD Jinja2 template checks ``config.uhf_npz``; we inject it via
    the ``**kwargs`` path by generating each input individually and passing
    ``uhf_npz`` as an extra keyword that ends up in the template context.

    When ``ccsd_level_dir`` is set (for UCCSD(T) level), the converged CCSD
    amplitudes NPZ is located and passed as ``ccsd_npz`` so that the template
    can skip the UCCSD kernel and only compute the (T) correction.
    """
    input_files = []
    ext = ".py" if code in ("pyscf", "block2", "hast_ucc") else ".inp"
    for cfg in configs:
        label = _sanitize_label(cfg)
        filename = f"{label}_{input_method}{ext}"
        filepath = os.path.join(level_dir, filename)

        per_config_kwargs = dict(kwargs)

        # Locate UHF .npz: prefer explicit uhf_level_dir, fall back to prev_level_dir
        uhf_search_dir = uhf_level_dir or prev_level_dir
        uhf_npz = os.path.join(uhf_search_dir, f"{label}_uhf.npz")
        if os.path.exists(uhf_npz):
            per_config_kwargs["uhf_npz"] = os.path.abspath(uhf_npz)

        # Locate CCSD amplitudes NPZ (for UCCSD(T) to skip CCSD kernel)
        if ccsd_level_dir:
            ccsd_npz = os.path.join(ccsd_level_dir, f"{label}_ccsd_results.npz")
            if os.path.exists(ccsd_npz):
                per_config_kwargs["ccsd_npz"] = os.path.abspath(ccsd_npz)

        content = generate_input(
            cfg,
            active_space,
            cluster_info,
            code=code,
            method=input_method,
            **per_config_kwargs,
        )
        #print(content)
        with open(filepath, "w") as fh:
            fh.write(content)
        input_files.append((filename, content))
    return input_files


def _get_d_count_key(config, cluster_info):
    """Return a hashable key for the d-count targets of a config.

    Configs sharing the same d_count_targets will also share the same
    high-spin UHF calculation (same oxidation state pattern).
    """
    if config.oxidation and cluster_info.metals:
        from apex_cas.CAS_builder_noncomputing import get_d_electron_count
        assignments = config.oxidation.assignments
        return tuple(
            (idx, get_d_electron_count(cluster_info.metals[idx].element, ox))
            for idx, ox in sorted(assignments.items())
            if idx < len(cluster_info.metals)
        )
    return ()


def _generate_uhf_inputs_deduped(
    configs,
    active_space,
    cluster_info,
    code,
    method,
    output_dir,
    **kwargs,
):
    """Generate UHF inputs with high-spin deduplication.

    Groups configs by (charge, spin, basis, d_count_targets).  For each group:
      1. Generate and execute ONE high-spin UHF script (Step 1 + Step 1b).
      2. Generate per-config BS UHF scripts that load the shared high-spin npz.
    """
    basis_set = kwargs.get("basis_set", "cc-pVDZ")

    # --- Group configs by shared high-spin calculation -----------------------
    groups = {}
    for cfg in configs:
        key = (
            cluster_info.total_charge,
            cluster_info.target_spin,
            basis_set,
            _get_d_count_key(cfg, cluster_info),
        )
        groups.setdefault(key, []).append(cfg)

    print(f"  UHF dedup: {len(configs)} configs -> {len(groups)} high-spin group(s)")

    all_input_files = []

    for group_key, group_configs in groups.items():
        group_label = f"group_{abs(hash(group_key)) % 100000:05d}"

        # --- Step A: Generate & run high-spin script -------------------------
        hs_script = f"{group_label}_highspin.py"
        hs_content = generate_input(
            group_configs[0],  # any config from the group (same molecule/oxidation)
            active_space,
            cluster_info,
            code=code,
            method="uhf_highspin",
            **kwargs,
        )
        hs_path = os.path.join(output_dir, hs_script)
        with open(hs_path, "w") as fh:
            fh.write(hs_content)

        print(f"  Running high-spin UHF for group {group_label} ...")
        outcomes = _execute_scripts([hs_script], output_dir)
        n_ok = sum(1 for _, ok, _ in outcomes if ok)
        if n_ok == 0:
            print(f"  WARNING: High-spin UHF failed for group {group_label}, "
                  f"falling back to per-config calculation")
            # Fallback: generate normal UHF scripts (no dedup)
            for cfg in group_configs:
                label = _sanitize_label(cfg)
                filename = f"{label}_{method}.py"
                filepath = os.path.join(output_dir, filename)
                content = generate_input(
                    cfg, active_space, cluster_info,
                    code=code, method=method, **kwargs,
                )
                with open(filepath, "w") as fh:
                    fh.write(content)
                all_input_files.append((filename, content))
            continue

        # Locate the high-spin npz
        hs_npz = os.path.join(output_dir, f"{group_label}_highspin.npz")
        if not os.path.exists(hs_npz):
            # The highspin template may use a different label
            hs_label = _sanitize_label(group_configs[0])
            hs_npz_alt = os.path.join(output_dir, f"{hs_label}_highspin.npz")
            if os.path.exists(hs_npz_alt):
                hs_npz = hs_npz_alt
            else:
                print(f"  WARNING: highspin npz not found for {group_label}, "
                      f"falling back")
                for cfg in group_configs:
                    label = _sanitize_label(cfg)
                    filename = f"{label}_{method}.py"
                    filepath = os.path.join(output_dir, filename)
                    content = generate_input(
                        cfg, active_space, cluster_info,
                        code=code, method=method, **kwargs,
                    )
                    with open(filepath, "w") as fh:
                        fh.write(content)
                    all_input_files.append((filename, content))
                continue

        hs_npz_abs = os.path.abspath(hs_npz)

        # --- Step B: Generate per-config BS UHF scripts ----------------------
        for cfg in group_configs:
            label = _sanitize_label(cfg)
            filename = f"{label}_{method}.py"
            filepath = os.path.join(output_dir, filename)
            content = generate_input(
                cfg, active_space, cluster_info,
                code=code, method=method,
                highspin_npz=hs_npz_abs,
                **kwargs,
            )
            with open(filepath, "w") as fh:
                fh.write(content)
            all_input_files.append((filename, content))

    return all_input_files


# ──────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────────────────────────


def run_pipeline(
    configs: list[ElectronicConfig],
    active_space: CAS,
    cluster_info: ClusterInfo,
    plan: FilteringPlan,
    workdir: str = "pipeline_output",
    code: str = "pyscf",
    basis_set: str = "cc-pVDZ",
    n_final: int = 5,
    continue_from: int = 0,
    generate_fcidump: bool = False,
) -> list[CalculationResult]:
    """Execute the progressive filtering funnel.

    For each level in *plan*:

    1. Generate input files for the surviving configs.
    2. Execute them (subprocess).
    3. Parse the ``.npz`` results.
    4. Call the appropriate ``select_from_*`` function.
    5. Pass surviving configs to the next level.

    Args:
        configs: Initial list of :class:`ElectronicConfig`.
        active_space: Active space specification.
        cluster_info: Cluster geometry / metadata.
        plan: :class:`FilteringPlan` with the funnel levels.
        workdir: Directory for generated inputs and results.
        code: QC package (``"pyscf"``).
        basis_set: Basis set for input generation.
        n_final: How many configs to return at the end.
        continue_from: Level index to resume from (0-based).
        generate_fcidump: If True, generate FCIDUMP files for final results.

    Returns:
        List of the best :class:`CalculationResult` objects after all
        filtering levels.
    """
    os.makedirs(workdir, exist_ok=True)

    # Symmetry deduplication
    from .electronic_config import reduce_configs_by_symmetry

    configs = reduce_configs_by_symmetry(configs, cluster_info)

    surviving_configs = list(configs)
    all_level_results = {}

    for level_idx, level in enumerate(plan.levels):
        if level_idx < continue_from:
            continue

        method = level.method
        info = _METHOD_INFO.get(method)
        if info is None:
            print(f"[Pipeline] Level {level_idx}: unknown method '{method}', skipping.")
            continue

        input_method = info["method"]
        suffix = info["suffix"]
        select_key = info["select"]
        level_code = info.get("code") or code

        level_dir = os.path.join(workdir, f"level_{level_idx}_{method}")
        os.makedirs(level_dir, exist_ok=True)

        # --- Generate inputs ------------------------------------------
        print(
            f"\n[Pipeline] Level {level_idx} ({method}): "
            f"{len(surviving_configs)} configs"
            + (f" [code={level_code}]" if level_code != code else "")
        )

        extra_kwargs = {"basis_set": basis_set}

        if method in ("UCCSD", "UCCSD(T)") and level_idx > 0:
            prev_level = plan.levels[level_idx - 1]
            prev_dir = os.path.join(
                workdir, f"level_{level_idx - 1}_{prev_level.method}"
            )
            prev_suffix = _METHOD_INFO.get(prev_level.method, {}).get(
                "suffix", "_uhf.npz"
            )

            # Find the UHF level directory (always needed for MO coefficients)
            uhf_level_idx = next(
                (i for i, lv in enumerate(plan.levels) if lv.method == "UHF"), 0
            )
            uhf_level = plan.levels[uhf_level_idx]
            uhf_dir = os.path.join(workdir, f"level_{uhf_level_idx}_{uhf_level.method}")

            # UCCSD(T)-specific: pass flags and locate CCSD amplitudes
            if method == "UCCSD(T)":
                extra_kwargs["run_triples"] = True
                extra_kwargs["result_suffix"] = "_ccsdt_results"
                extra_kwargs["output_suffix"] = "_ccsdt"
                ccsd_level_idx = next(
                    (i for i, lv in enumerate(plan.levels) if lv.method == "UCCSD"),
                    None,
                )
                ccsd_dir = (
                    os.path.join(workdir, f"level_{ccsd_level_idx}_UCCSD")
                    if ccsd_level_idx is not None else None
                )
            else:
                ccsd_dir = None

            input_files = _generate_ccsd_inputs(
                surviving_configs,
                active_space,
                cluster_info,
                level_code,
                input_method,
                level_dir,
                prev_dir,
                prev_suffix,
                uhf_level_dir=uhf_dir,
                ccsd_level_dir=ccsd_dir,
                **extra_kwargs,
            )
            #print(input_files)
        elif method in ("DMRG", "CCSDTQ") and level_idx > 0:
            # DMRG & CCSDTQ need UHF MOs from the UHF level
            uhf_level_idx = next(
                (i for i, lv in enumerate(plan.levels) if lv.method == "UHF"), 0
            )
            uhf_level = plan.levels[uhf_level_idx]
            uhf_dir = os.path.join(workdir, f"level_{uhf_level_idx}_{uhf_level.method}")
            input_files = _generate_ccsd_inputs(
                surviving_configs,
                active_space,
                cluster_info,
                level_code,
                input_method,
                level_dir,
                uhf_dir,
                "_uhf.npz",
                **extra_kwargs,
            )
        else:
            # UHF & CCSDTQ — with high-spin dedup for UHF
            if method == "UHF":
                input_files = _generate_uhf_inputs_deduped(
                    surviving_configs,
                    active_space,
                    cluster_info,
                    code=level_code,
                    method=input_method,
                    output_dir=level_dir,
                    **extra_kwargs,
                )
            else:
                # CCSDTQ & others — need UHF MOs from the UHF level
                uhf_level_idx = next(
                    (i for i, lv in enumerate(plan.levels) if lv.method == "UHF"), 0
                )
                uhf_level = plan.levels[uhf_level_idx]
                uhf_dir = os.path.join(workdir, f"level_{uhf_level_idx}_{uhf_level.method}")
                input_files = _generate_ccsd_inputs(
                    surviving_configs,
                    active_space,
                    cluster_info,
                    level_code,
                    input_method,
                    level_dir,
                    uhf_dir,
                    "_uhf.npz",
                    **extra_kwargs,
                )

        script_paths = [fname for fname, _ in input_files]
        print(f"  Generated {len(script_paths)} input scripts in {level_dir}/")

        # --- Execute --------------------------------------------------
        print(f"  Executing {len(script_paths)} calculations ...")
        outcomes = _execute_scripts(script_paths, level_dir)
        n_ok = sum(1 for _, ok, _ in outcomes if ok)
        n_fail = len(outcomes) - n_ok
        print(f"  Completed: {n_ok} ok, {n_fail} failed")

        _write_level_log(level_dir, outcomes)

        # --- Parse results --------------------------------------------
        results = _parse_results(surviving_configs, level_dir, suffix)
        n_converged = sum(1 for r in results if r.converged)
        print(f"  Parsed {len(results)} results ({n_converged} converged)")

        all_level_results[level_idx] = results
        #print(results)

        if not results:
            print("  WARNING: No results obtained, stopping pipeline.")
            break

        # --- Select ---------------------------------------------------
        n_per = level.n_per_isomer if level.n_per_isomer > 0 else len(results)
        n_total = level.n_output if level.n_output > 0 else None

        if select_key is not None:
            selected = _select(results, select_key, n_per, n_total)
        else:
            selected = select_lowest_energy(results, n_keep=n_total or len(results))

        print(f"  Selected {len(selected)} configs for next level")

        surviving_configs = [r.config for r in selected if r.config]

        if not surviving_configs:
            print("  WARNING: No configs survived selection.")
            break

    # --- Final selection --------------------------------------------------
    final_results = all_level_results.get(len(plan.levels) - 1, [])
    if not final_results:
        for idx in reversed(range(len(plan.levels))):
            if idx in all_level_results:
                final_results = all_level_results[idx]
                break

    final = select_lowest_energy(final_results, n_keep=n_final)

    # --- Summary ----------------------------------------------------------
    _print_summary(final, workdir)
    _save_summary(final, workdir)

    # --- FCIDUMP generation -----------------------------------------------
    if generate_fcidump and final:
        _generate_fcidumps(final, cluster_info, active_space, workdir, plan, basis_set)

    return final


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _write_level_log(level_dir, outcomes):
    """Write a short ``pipeline.log`` for the level."""
    log_path = os.path.join(level_dir, "pipeline.log")
    with open(log_path, "w") as fh:
        for script, ok, proc in outcomes:
            status = "OK" if ok else "FAIL"
            fh.write(f"[{status}] {script}\n")
            if proc and proc.stderr:
                fh.write(f"  stderr: {proc.stderr[:500]}\n")


def _print_summary(final, workdir):
    """Print the final ranking to stdout."""
    print("\n" + "=" * 70)
    print("Pipeline complete -- final ranking")
    print("=" * 70)
    for rank, r in enumerate(final, 1):
        cfg_label = r.config.label if r.config else "N/A"
        print(
            f"  #{rank:3d}  {r.method:>10s}  E = {r.energy:>.12f}  "
            f"<S^2> = {r.s_squared:>.4f}  converged={r.converged}  "
            f"[{cfg_label}]"
        )
    print("=" * 70)


def _save_summary(final, workdir):
    """Save a machine-readable summary to *workdir*/pipeline_summary.json."""
    summary = []
    for rank, r in enumerate(final, 1):
        entry = {
            "rank": rank,
            "method": r.method,
            "energy": r.energy,
            "correlation_energy": r.correlation_energy,
            "s_squared": r.s_squared,
            "converged": r.converged,
        }
        if r.config:
            entry["label"] = r.config.label
            entry["config_id"] = r.config.config_id
            if r.config.spin_isomer:
                entry["spin_isomer"] = r.config.spin_isomer.label
                entry["spin_family"] = r.config.spin_isomer.family
        summary.append(entry)

    path = os.path.join(workdir, "pipeline_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved to {path}")


def _generate_fcidumps(final, cluster_info, active_space, workdir, plan, basis_set):
    """Generate FCIDUMP files for the final pipeline results."""
    try:
        from .fcidump import generate_fcidump_for_results
    except ImportError:
        print("  [WARNING] fcidump module not available, skipping FCIDUMP generation")
        return

    # FCIDUMP output dir = sibling of pipeline_output dir
    parent_dir = os.path.dirname(workdir.rstrip("/"))
    fcidump_dir = os.path.join(parent_dir, "fcidump")

    print(f"\n[FCIDUMP] Generating FCIDUMP files in {fcidump_dir}/ ...")
    try:
        info_list = generate_fcidump_for_results(
            final,
            cluster_info,
            active_space,
            workdir,
            plan,
            basis_set=basis_set,
            output_dir=fcidump_dir,
        )
        if info_list:
            print(f"[FCIDUMP] Generated {len(info_list)} FCIDUMP set(s)")
        else:
            print("[FCIDUMP] No FCIDUMP files generated (no valid NPZ data found)")
    except Exception as e:
        print(f"  [WARNING] FCIDUMP generation failed: {e}")


# ──────────────────────────────────────────────────────────────────
# DMRG entropy feedback (Phase 2)
# ──────────────────────────────────────────────────────────────────

def run_entropy_feedback(
    workdir: str,
    cluster_info: "ClusterInfo",
    active_space: "CAS",
    dmrg_level_idx: int = None,
    entropy_threshold_removable: float = 0.1,
    entropy_threshold_critical: float = 0.5,
):
    """Analyze DMRG single-orbital entropy and generate feedback report.

    After DMRG calculations complete, this function reads the DMRG results,
    extracts single-orbital entropy values, and produces a report identifying:
    - Critical orbitals (high entropy, must keep)
    - Removable orbitals (low entropy, could be removed to reduce cost)

    Args:
        workdir: Pipeline output directory.
        cluster_info: ClusterInfo object.
        active_space: Current active space.
        dmrg_level_idx: Index of the DMRG level in the pipeline (auto-detected if None).
        entropy_threshold_removable: Entropy below this = removable.
        entropy_threshold_critical: Entropy above this = critical.

    Returns:
        dict with keys:
            "entropy_report": str (human-readable report)
            "orbital_classifications": list[dict]
            "n_removable": int
            "n_critical": int
            "recommendation": str
    """
    from apex_cas.active_space_quality import analyze_entropy

    # 1. Find DMRG results directory
    dmrg_dirs = []
    for entry in os.listdir(workdir):
        entry_path = os.path.join(workdir, entry)
        if os.path.isdir(entry_path) and "DMRG" in entry.upper():
            dmrg_dirs.append(entry_path)

    if not dmrg_dirs:
        return {
            "entropy_report": "No DMRG results found in workdir.",
            "orbital_classifications": [],
            "n_removable": 0,
            "n_critical": 0,
            "recommendation": "Run DMRG calculations first.",
        }

    # Use the specified or last DMRG directory
    if dmrg_level_idx is not None:
        target_dir = os.path.join(workdir, f"level_{dmrg_level_idx}_DMRG")
        if not os.path.isdir(target_dir):
            target_dir = dmrg_dirs[-1]
    else:
        target_dir = dmrg_dirs[-1]

    # 2. Load entropy data from DMRG results
    entropy_values = []
    orbital_labels = []

    # Look for npz files with entropy data
    import glob as glob_mod
    npz_files = sorted(glob_mod.glob(os.path.join(target_dir, "*_dmrg_results.npz")))

    if not npz_files:
        return {
            "entropy_report": f"No DMRG NPZ files found in {target_dir}.",
            "orbital_classifications": [],
            "n_removable": 0,
            "n_critical": 0,
            "recommendation": "Check DMRG output format.",
        }

    # Load entropy from the first (or best) result
    best_npz = npz_files[0]
    try:
        data = dict(np.load(best_npz, allow_pickle=True))
        if "entropy" in data:
            entropy_values = data["entropy"].tolist()
        elif "single_orbital_entropy" in data:
            entropy_values = data["single_orbital_entropy"].tolist()
    except Exception as e:
        return {
            "entropy_report": f"Error loading DMRG results: {e}",
            "orbital_classifications": [],
            "n_removable": 0,
            "n_critical": 0,
            "recommendation": "Check DMRG output format.",
        }

    if not entropy_values:
        return {
            "entropy_report": "No entropy data found in DMRG results.",
            "orbital_classifications": [],
            "n_removable": 0,
            "n_critical": 0,
            "recommendation": "Enable entropy calculation in DMRG settings.",
        }

    # 3. Generate orbital labels
    n_active = active_space.n_orbitals
    if len(entropy_values) >= n_active:
        entropy_values = entropy_values[:n_active]
    orbital_labels = [f"orb_{i}" for i in range(len(entropy_values))]

    # Try to get chemical labels from active space
    if active_space.orbital_labels:
        src_labels = active_space.orbital_labels
        for i in range(min(len(src_labels), len(orbital_labels))):
            orbital_labels[i] = src_labels[i]

    # 4. Run entropy analysis
    classifications = analyze_entropy(
        entropy_values, orbital_labels,
        threshold_critical=entropy_threshold_critical,
        threshold_removable=entropy_threshold_removable,
    )

    n_critical = sum(1 for c in classifications if c["classification"] == "critical")
    n_removable = sum(1 for c in classifications if c["classification"] == "removable")

    # 5. Build report
    report_lines = [
        "=" * 60,
        "DMRG Entropy Analysis Report",
        "=" * 60,
        f"Source: {best_npz}",
        f"Active space: ({active_space.n_electrons}e, {active_space.n_orbitals}o)",
        f"Orbitals analyzed: {len(entropy_values)}",
        f"Critical (entropy >= {entropy_threshold_critical}): {n_critical}",
        f"Removable (entropy < {entropy_threshold_removable}): {n_removable}",
        "",
        "Per-orbital breakdown:",
    ]
    for c in classifications:
        report_lines.append(
            f"  {c['label']:>20s}  entropy={c['entropy']:.4f}  "
            f"[{c['classification']:>10s}]  {c['recommendation']}"
        )

    if n_removable > 0:
        recommendation = (
            f"Found {n_removable} removable orbitals. Consider reducing "
            f"active space from ({active_space.n_electrons}e, {active_space.n_orbitals}o) "
            f"to ({active_space.n_electrons - 2*n_removable}e, "
            f"{active_space.n_orbitals - n_removable}o) to save DMRG cost."
        )
    else:
        recommendation = "All orbitals are significant. Current active space is appropriate."

    report_lines.append("")
    report_lines.append(f"Recommendation: {recommendation}")
    report_lines.append("=" * 60)

    return {
        "entropy_report": "\n".join(report_lines),
        "orbital_classifications": classifications,
        "n_removable": n_removable,
        "n_critical": n_critical,
        "recommendation": recommendation,
    }
