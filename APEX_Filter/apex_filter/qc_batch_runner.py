"""QC batch execution helpers for CC/UCC screening steps.

Provides the low-level helpers used by the interactive pipeline steps
(``steps.py``) for UCCSD and UCCSD(T) calculations.

Important:
The current PySCF script path reconstructs a full-molecule job from
geometry + basis set. It is therefore a compatibility path for CC
screening, not the canonical Chan-2026 active-space Hamiltonian route.
"""

import os
import subprocess
import sys
import logging

from .qc_input_renderer import generate_input
from .result_parser import parse_npz_result, to_calculation_result

logger = logging.getLogger(__name__)


def _sanitize_label(cfg):
    """Return a filesystem-safe label from a config."""
    return cfg.label.replace("|", "_").replace(" ", "_")


def _execute_scripts(script_paths, workdir):
    """Run each Python script via subprocess, returning per-script outcome."""
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
            logger.warning("Timed out while executing %s", script_abs)
            outcomes.append((script, False, None))
        except Exception as exc:
            logger.warning("Failed to execute %s: %s", script_abs, exc)
            outcomes.append((script, False, None))
    return outcomes


def _parse_results(configs, workdir, suffix):
    """Parse ``.npz`` result files for a batch of configs."""
    results = []
    for cfg in configs:
        label = _sanitize_label(cfg)
        npz_path = os.path.join(workdir, f"{label}{suffix}")
        if not os.path.exists(npz_path):
            logger.warning("Result file missing for %s: %s", label, npz_path)
            continue
        try:
            parsed = parse_npz_result(npz_path)
            results.append(to_calculation_result(parsed, config=cfg))
        except Exception as exc:
            logger.warning("Failed to parse result for %s from %s: %s", label, npz_path, exc)
            continue
    return results


def _generate_ccsd_inputs(
    configs,
    active_space,
    cluster_info,
    code,
    input_method,
    level_dir,
    uhf_level_dir,
    ccsd_level_dir=None,
    **kwargs,
):
    """Generate CCSD/UCCSD(T) inputs, one per config.

    Injects the UHF .npz path (and optionally CCSD amplitudes .npz)
    into each input file so the template can load the correct orbitals.
    The caller may disable this via ``use_uhf_npz=False`` when the backend
    is a non-canonical full-molecule route rather than the active-space
    Hamiltonian path.
    """
    input_files = []
    ext = ".py" if code in ("pyscf", "block2", "hast_ucc") else ".inp"
    use_uhf_npz = kwargs.pop("use_uhf_npz", True)
    os.makedirs(level_dir, exist_ok=True)

    for cfg in configs:
        label = _sanitize_label(cfg)
        filename = f"{label}_{input_method}{ext}"
        filepath = os.path.join(level_dir, filename)

        per_config_kwargs = dict(kwargs)
        result_suffix = per_config_kwargs.get("result_suffix")
        if result_suffix:
            per_config_kwargs["result_filename"] = f"{label}{result_suffix}.npz"

        # Locate UHF .npz
        uhf_npz = os.path.join(uhf_level_dir, f"{label}_uhf.npz")
        if use_uhf_npz and os.path.exists(uhf_npz):
            per_config_kwargs["uhf_npz"] = os.path.abspath(uhf_npz)

        # Locate CCSD amplitudes NPZ (for UCCSD(T))
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
        with open(filepath, "w") as fh:
            fh.write(content)
        input_files.append((filename, content))

    return input_files
