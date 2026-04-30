"""Module 9: Result Parser

Parse QC output files to feed the filtering funnel and extrapolation modules.
Supports PySCF checkpoint files, BLOCK2 output, and HAST-UCC output.
"""

import json
import os
import re

import numpy as np

from .models import CalculationResult, ElectronicConfig


def parse_pyscf_uhf(chkfile: str) -> dict:
    """Parse PySCF UHF results from checkpoint file.

    Args:
        chkfile: Path to PySCF checkpoint file (.chk).

    Returns:
        Dict with energy, MO coefficients, orbital energies, <S^2>, convergence.
    """
    if not os.path.exists(chkfile):
        raise FileNotFoundError(f"Checkpoint file not found: {chkfile}")

    result = {
        "method": "UHF",
        "energy": 0.0,
        "mo_coeff": None,
        "mo_energy": None,
        "mo_occ": None,
        "s_squared": 0.0,
        "converged": False,
    }

    try:
        from pyscf import lib as pyscf_lib
        data = pyscf_lib.chkfile.load(chkfile, "scf")
        result["energy"] = data.get("energy", 0.0)
        result["converged"] = data.get("converged", False)

        if "mo_coeff" in data:
            mc = data["mo_coeff"]
            if isinstance(mc, np.ndarray) and mc.ndim == 3:
                result["mo_coeff"] = [mc[0], mc[1]]
            else:
                result["mo_coeff"] = mc

        if "mo_energy" in data:
            me = data["mo_energy"]
            if isinstance(me, np.ndarray) and me.ndim == 2:
                result["mo_energy"] = [me[0], me[1]]
            else:
                result["mo_energy"] = me

        if "mo_occ" in data:
            mo = data["mo_occ"]
            if isinstance(mo, np.ndarray) and mo.ndim == 2:
                result["mo_occ"] = [mo[0], mo[1]]
            else:
                result["mo_occ"] = mo

        if "spin_square" in data:
            result["s_squared"] = float(data["spin_square"])
    except ImportError:
        result["parse_error"] = "PySCF not installed"
    except Exception as e:
        result["parse_error"] = str(e)

    return result


def parse_pyscf_ccsd(chkfile: str) -> dict:
    """Parse PySCF UCCSD/UCCSD(T) results.

    Args:
        chkfile: Path to checkpoint file.

    Returns:
        Dict with total and correlation energies.
    """
    if not os.path.exists(chkfile):
        raise FileNotFoundError(f"Checkpoint file not found: {chkfile}")

    result = {
        "method": "UCCSD",
        "energy": 0.0,
        "correlation_energy": 0.0,
        "t1_norm": 0.0,
        "converged": False,
    }

    try:
        from pyscf import lib as pyscf_lib
        data = pyscf_lib.chkfile.load(chkfile, "ccsd")
        result["energy"] = data.get("e_tot", 0.0)
        result["correlation_energy"] = data.get("e_corr", 0.0)
        result["converged"] = data.get("converged", False)

        if "t1_norm" in data:
            result["t1_norm"] = float(data["t1_norm"])

        # Check for (T) correction
        try:
            data_t = pyscf_lib.chkfile.load(chkfile, "ccsd(t)")
            if data_t:
                result["method"] = "UCCSD(T)"
                result["energy_t"] = data_t.get("e_tot", 0.0)
                result["e_t_correction"] = (
                    data_t.get("e_corr", 0.0) - result["correlation_energy"]
                )
        except Exception:
            pass
    except ImportError:
        result["parse_error"] = "PySCF not installed"
    except Exception as e:
        result["parse_error"] = str(e)

    return result


def parse_block2_dmrg(output_file: str) -> dict:
    """Parse BLOCK2 DMRG output file.

    Extracts energies per sweep, final energy, bond dimension, discarded weight.

    Args:
        output_file: Path to BLOCK2 stdout/log file.

    Returns:
        Dict with DMRG results.
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    result = {
        "method": "DMRG",
        "energy": 0.0,
        "energies_per_sweep": [],
        "bond_dims": [],
        "discarded_weights": [],
        "n_sweeps": 0,
        "converged": False,
    }

    with open(output_file, "r") as f:
        content = f.read()

    # Parse sweep energies: "Sweep N ... E = -123.456789"
    sweep_pattern = re.compile(
        r"Sweep\s+(\d+)\s+.*?E\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)"
    )
    for match in sweep_pattern.finditer(content):
        energy = float(match.group(2))
        result["energies_per_sweep"].append(energy)

    # Parse discarded weights
    dw_pattern = re.compile(r"DW\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
    for match in dw_pattern.finditer(content):
        dw = float(match.group(1))
        result["discarded_weights"].append(dw)

    # Parse bond dimensions: "M = N"
    m_pattern = re.compile(r"\bM\s*=\s*(\d+)")
    for match in m_pattern.finditer(content):
        result["bond_dims"].append(int(match.group(1)))

    # Final energy = last sweep energy
    if result["energies_per_sweep"]:
        result["energy"] = result["energies_per_sweep"][-1]
        result["n_sweeps"] = len(result["energies_per_sweep"])

    # Check convergence
    if "converged" in content.lower():
        result["converged"] = True
    elif result["discarded_weights"] and result["discarded_weights"][-1] < 1e-5:
        result["converged"] = True

    return result


def parse_hast_output(output_file: str) -> dict:
    """Parse HAST-UCC (high-order CC) output.

    Args:
        output_file: Path to HAST-UCC output file.

    Returns:
        Dict with CCSDT/CCSDTQ energies.
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    result = {
        "method": "HAST-UCC",
        "energy": 0.0,
        "correlation_energy": 0.0,
        "converged": False,
    }

    with open(output_file, "r") as f:
        content = f.read()

    # Detect method level
    if "CCSDTQ" in content.upper():
        result["method"] = "CCSDTQ"
    elif "CCSDT" in content.upper():
        result["method"] = "CCSDT"
    elif "CCSD" in content.upper():
        result["method"] = "CCSD"

    # Parse total energy
    e_pattern = re.compile(
        r"(?:Total\s+)?E(?:nergy)?\s*(?:=|:)\s*([+-]?\d+\.\d+)",
        re.IGNORECASE,
    )
    matches = e_pattern.findall(content)
    if matches:
        result["energy"] = float(matches[-1])

    # Parse correlation energy
    corr_pattern = re.compile(
        r"(?:Correlation\s+)?E(?:_corr|C)\s*(?:=|:)\s*([+-]?\d+\.\d+)",
        re.IGNORECASE,
    )
    corr_matches = corr_pattern.findall(content)
    if corr_matches:
        result["correlation_energy"] = float(corr_matches[-1])

    # Check convergence
    if "converged" in content.lower():
        result["converged"] = True

    return result


def check_convergence(parsed_result: dict) -> bool:
    """Verify SCF/CC/DMRG convergence from parsed result.

    Args:
        parsed_result: Dict from one of the parse_* functions.

    Returns:
        True if the calculation converged satisfactorily.
    """
    if not parsed_result.get("converged", False):
        return False

    method = parsed_result.get("method", "")

    if method == "DMRG":
        dws = parsed_result.get("discarded_weights", [])
        if dws and dws[-1] > 1e-4:
            return False

    return True


def parse_npz_result(npz_path: str) -> dict:
    """Parse results from a .npz file saved by generated PySCF scripts.

    The UHF template saves ``{label}_uhf.npz`` and the CCSD template saves
    ``{label}_ccsd_results.npz``.  This function reads either format and
    returns a dict compatible with :func:`to_calculation_result`.

    Args:
        npz_path: Path to the .npz result file.

    Returns:
        Dict with method, energy, convergence info, etc.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Result file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # Detect which kind of result this is from the keys present.
    keys = set(data.files)

    if "ccsd_corr" in keys or "ccsd_total" in keys:
        # CCSD / CCSD(T) result
        result = {
            "method": "UCCSD",
            "energy": float(data.get("ccsd_total", 0.0)),
            "correlation_energy": float(data.get("ccsd_corr", 0.0)),
            "converged": bool(data.get("ccsd_converged", False)),
            "t1_norm": 0.0,
        }
        if "spin_sq" in keys:
            result["s_squared"] = float(data["spin_sq"])
        if "et_correction" in keys:
            result["method"] = "UCCSD(T)"
            result["energy_t"] = float(data.get("ccsd_t_total", 0.0))
            result["e_t_correction"] = float(data["et_correction"])
        return result

    if "dmrg_total" in keys or "dmrg_corr" in keys:
        # DMRG result (block2/pyblock2)
        result = {
            "method": "DMRG",
            "energy": float(data.get("dmrg_total", 0.0)),
            "correlation_energy": float(data.get("dmrg_corr", 0.0)),
            "converged": bool(data.get("dmrg_converged", False)),
        }
        if "spin_sq" in keys:
            result["s_squared"] = float(data["spin_sq"])
        if "dmrg_extrapolated" in keys:
            result["energy_extrapolated"] = float(data["dmrg_extrapolated"])
        return result

    if "hast_total" in keys or "ccsdtq_total" in keys:
        # HAST-UCC / CCSDTQ result
        energy = float(data.get("hast_total",
                                data.get("ccsdtq_total", 0.0)))
        result = {
            "method": "CCSDTQ",
            "energy": energy,
            "correlation_energy": float(data.get("hast_corr",
                                                  data.get("ccsdtq_corr", 0.0))),
            "converged": bool(data.get("hast_converged",
                                       data.get("ccsdtq_converged", False))),
        }
        if "spin_sq" in keys:
            result["s_squared"] = float(data["spin_sq"])
        return result

    # Default: treat as UHF result
    result = {
        "method": "UHF",
        "energy": float(data.get("energy", 0.0)),
        "converged": bool(data.get("converged", False)),
        "s_squared": float(data.get("spin_sq", 0.0)),
    }
    return result


def to_calculation_result(parsed: dict,
                           config: ElectronicConfig = None) -> CalculationResult:
    """Convert parsed dict to CalculationResult dataclass.

    Args:
        parsed: Dict from one of the parse_* functions.
        config: Associated ElectronicConfig.

    Returns:
        CalculationResult object.
    """
    return CalculationResult(
        config=config,
        method=parsed.get("method", ""),
        energy=parsed.get("energy", 0.0),
        correlation_energy=parsed.get("correlation_energy", 0.0),
        s_squared=parsed.get("s_squared", 0.0),
        converged=check_convergence(parsed),
        params=parsed,
    )
