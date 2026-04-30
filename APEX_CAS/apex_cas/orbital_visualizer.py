"""Orbital visualization utilities.

Generates visualization artifacts for interactive active-space selection:
  - YAML orbital report (template for user editing)
  - Gaussian cube files (VESTA/Jmol)
  - NOON bar plot (matplotlib)
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def generate_orbital_report(
    mol,
    mo_coeff_loc: np.ndarray,
    occupations: np.ndarray,
    labels: list[str],
    cluster_info=None,
    output_path: str = "orbital_report.yaml",
    occ_active_lo: float = 0.02,
    occ_active_hi: float = 1.98,
) -> str:
    """Generate a YAML orbital report that doubles as a selection template.

    Each orbital entry contains index, occupation, chemical label, block
    classification, and top AO contributions.  Users edit the ``selected``
    field to choose active orbitals.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff_loc : ndarray (nao, nmo)
        Localized MO coefficients.
    occupations : ndarray (nmo,)
        UNO occupation numbers.
    labels : list[str]
        Chemical labels from split-localization.
    cluster_info : ClusterInfo, optional
        Used for AO contribution analysis on metal/bridge atoms.
    output_path : str
        Output YAML file path.
    occ_active_lo, occ_active_hi : float
        Thresholds defining the active block.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    import yaml

    nmo = len(occupations)
    aoslices = mol.aoslice_by_atom()
    ao_labels = mol.ao_labels()

    # Determine target atoms for AO contribution analysis
    target_atom_indices = _get_target_atom_indices(mol, cluster_info)

    orbitals = []
    for i in range(nmo):
        occ = float(occupations[i])

        # Block classification
        if occ > occ_active_hi:
            block = "core"
        elif occ < occ_active_lo:
            block = "virtual"
        else:
            block = "active"

        # AO contributions
        ao_contrib = _compute_ao_contributions(
            mol, mo_coeff_loc[:, i], target_atom_indices, aoslices, ao_labels,
        )

        orbitals.append({
            "index": i,
            "occupation": round(occ, 6),
            "auto_label": labels[i] if i < len(labels) else f"orb_{i}",
            "chemical_label": _best_chemical_label(ao_contrib),
            "block": block,
            "ao_contribution": ao_contrib,
            "selected": block == "active",
        })

    n_core = sum(1 for o in orbitals if o["block"] == "core")
    n_active = sum(1 for o in orbitals if o["block"] == "active")
    n_virtual = sum(1 for o in orbitals if o["block"] == "virtual")

    report = {
        "metadata": {
            "n_total": nmo,
            "n_core": n_core,
            "n_active": n_active,
            "n_virtual": n_virtual,
            "threshold_active_lo": occ_active_lo,
            "threshold_active_hi": occ_active_hi,
            "total_electrons": round(float(np.sum(occupations)), 2),
        },
        "orbitals": orbitals,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Orbital report written to %s (%d orbitals: %d core, %d active, %d virtual)",
                output_path, nmo, n_core, n_active, n_virtual)
    return os.path.abspath(output_path)


def generate_orbital_cubes(
    mol,
    mo_coeff: np.ndarray,
    indices: Optional[list[int]] = None,
    output_dir: str = "cubes",
    prefix: str = "orb",
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
) -> list[str]:
    """Generate Gaussian cube files for selected orbitals.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : ndarray (nao, nmo)
        Full MO coefficient matrix.
    indices : list[int], optional
        Orbital indices to visualize.  ``None`` visualises all columns.
    output_dir : str
    prefix : str
        File name prefix.
    nx, ny, nz : int
        Cube grid resolution.

    Returns
    -------
    list[str]
        Paths to generated cube files.
    """
    from pyscf.tools import cubegen

    os.makedirs(output_dir, exist_ok=True)

    if indices is None:
        indices = list(range(mo_coeff.shape[1]))

    paths = []
    for i in indices:
        filepath = os.path.join(output_dir, f"{prefix}_{i:04d}.cube")
        cubegen.orbital(mol, filepath, mo_coeff[:, i], nx=nx, ny=ny, nz=nz)
        paths.append(filepath)

    logger.info("Generated %d cube files in %s", len(paths), output_dir)
    return paths


def generate_noon_plot(
    occupations: np.ndarray,
    labels: Optional[list[str]] = None,
    output_path: str = "noon_plot.png",
    active_lo: float = 0.02,
    active_hi: float = 1.98,
    show_top_n: int = 50,
) -> str:
    """Generate a NOON (Natural Orbital Occupation Number) bar plot.

    Active-range orbitals are highlighted in red; core and virtual in grey.
    Threshold lines are drawn at *active_lo* and *active_hi*.

    Parameters
    ----------
    occupations : ndarray (nmo,)
    labels : list[str], optional
    output_path : str
    active_lo, active_hi : float
    show_top_n : int
        Max number of chemical labels to annotate.

    Returns
    -------
    str
        Absolute path of the saved image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping NOON plot")
        print("WARNING: matplotlib not installed. Skipping NOON plot generation.")
        print("  Install with: pip install matplotlib")
        return ""

    nmo = len(occupations)
    colors = []
    for occ in occupations:
        if active_lo <= occ <= active_hi:
            colors.append("#d62728")   # red for active
        else:
            colors.append("#aaaaaa")   # grey for core/virtual

    fig, ax = plt.subplots(figsize=(max(12, nmo * 0.06), 5))
    ax.bar(range(nmo), occupations, color=colors, width=1.0, edgecolor="none")
    ax.axhline(y=active_hi, color="blue", linestyle="--", linewidth=0.8, label=f"occ = {active_hi}")
    ax.axhline(y=active_lo, color="blue", linestyle="--", linewidth=0.8, label=f"occ = {active_lo}")
    ax.set_xlabel("Orbital index")
    ax.set_ylabel("UNO occupation number")
    ax.set_title("Natural Orbital Occupation Numbers (NOON)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 2.05)

    # Annotate active-range labels
    if labels:
        active_indices = [i for i in range(nmo) if active_lo <= occupations[i] <= active_hi]
        for k, i in enumerate(active_indices[:show_top_n]):
            lab = labels[i] if i < len(labels) else ""
            if lab:
                ax.annotate(lab, (i, occupations[i]),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=4, rotation=90, ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("NOON plot saved to %s", output_path)
    return os.path.abspath(output_path)


def load_user_selection(
    yaml_path: str,
) -> tuple[list[int], list[str], dict]:
    """Read a user-edited orbital report YAML and extract selected orbitals.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file with ``selected: true`` entries.

    Returns
    -------
    tuple[list[int], list[str], dict]
        (selected_indices, selected_labels, metadata)
    """
    import yaml

    with open(yaml_path) as f:
        report = yaml.safe_load(f)

    metadata = report.get("metadata", {})
    orbitals = report.get("orbitals", [])

    selected_indices = []
    selected_labels = []
    for orb in orbitals:
        if orb.get("selected", False):
            idx = orb["index"]
            label = orb.get("chemical_label", "") or orb.get("auto_label", f"orb_{idx}")
            selected_indices.append(idx)
            selected_labels.append(label)

    selected_indices.sort()
    logger.info("Loaded %d selected orbitals from %s", len(selected_indices), yaml_path)
    return selected_indices, selected_labels, metadata


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _get_target_atom_indices(mol, cluster_info) -> list[int]:
    """Return atom indices of interest (metals + bridging atoms)."""
    if cluster_info is not None:
        indices = [m.index for m in cluster_info.metals]
        indices += [b.index for b in cluster_info.bridging_atoms]
        if indices:
            return indices
    # Fallback: use all atoms
    return list(range(mol.natm))


def _compute_ao_contributions(
    mol,
    mo_col: np.ndarray,
    target_atoms: list[int],
    aoslices,
    ao_labels: list[str],
    top_n: int = 3,
) -> dict[str, float]:
    """Compute the top-N AO contributions for a single molecular orbital.

    Returns a dict like ``{"Fe1_3d": 0.65, "S2_3p": 0.15}`` with at most
    *top_n* entries sorted by contribution (descending).
    """
    contributions = {}
    for atom_idx in target_atoms:
        if atom_idx >= len(aoslices):
            continue
        _, _, ao_s, ao_e = aoslices[atom_idx]
        local_coeffs_sq = mo_col[ao_s:ao_e] ** 2
        total = float(np.sum(local_coeffs_sq))
        if total < 1e-6:
            continue

        # Determine element + dominant AO type
        elem = mol.atom_symbol(atom_idx)
        dominant = int(np.argmax(local_coeffs_sq))
        label = ao_labels[ao_s + dominant] if ao_s + dominant < len(ao_labels) else ""
        parts = label.split()
        ao_type = parts[-1] if len(parts) > 1 else ""

        # Map to broader type
        import re
        ao_broad = re.sub(r"\d+[sp]", lambda m: m.group()[-1], ao_type) if ao_type else "orb"

        key = f"{elem}{atom_idx}_{ao_broad}"
        contributions[key] = round(total, 4)

    # Sort by contribution, keep top_n
    sorted_items = sorted(contributions.items(), key=lambda x: -x[1])[:top_n]
    return dict(sorted_items)


def _best_chemical_label(ao_contrib: dict) -> str:
    """Return the dominant chemical label from AO contributions."""
    if not ao_contrib:
        return ""
    return max(ao_contrib, key=ao_contrib.get)


# ──────────────────────────────────────────────────────────────────
# High-level convenience functions
# ──────────────────────────────────────────────────────────────────

def plot_orbitals(
    cas,
    mol,
    output_dir: str,
    cluster_info=None,
    generate_cubes: bool = True,
    cube_grid: str = "80x80x80",
    occ_active_lo: float = 0.02,
    occ_active_hi: float = 1.98,
) -> dict:
    """Generate full orbital visualizations from a CAS dataclass.

    Extracts ``mo_coeff_full``, ``occupations_full``, ``orbital_labels_full``
    from *cas* and calls the lower-level visualizer functions.

    Parameters
    ----------
    cas : CAS
        Must have ``mo_coeff_full``, ``occupations_full``, and
        ``orbital_labels_full`` populated.
    mol : pyscf.gto.Mole
    output_dir : str
        Output directory (e.g. ``outputs/orbitals/``).
    cluster_info : ClusterInfo, optional
    generate_cubes : bool
    cube_grid : str
        Grid resolution, e.g. ``"80x80x80"``.
    occ_active_lo, occ_active_hi : float

    Returns
    -------
    dict
        ``{"report_path", "noon_path", "cube_dir"}``
    """
    mo_coeff_loc = cas.mo_coeff_full
    occ = cas.occupations_full
    labels = cas.orbital_labels_full

    if mo_coeff_loc is None or occ is None:
        raise ValueError("CAS.mo_coeff_full and CAS.occupations_full must be populated.")

    os.makedirs(output_dir, exist_ok=True)

    # 1. YAML report
    report_path = os.path.join(output_dir, "orbital_report.yaml")
    generate_orbital_report(
        mol, mo_coeff_loc, occ, labels,
        cluster_info=cluster_info,
        output_path=report_path,
        occ_active_lo=occ_active_lo,
        occ_active_hi=occ_active_hi,
    )

    # 2. NOON bar plot
    noon_path = os.path.join(output_dir, "noon_plot.png")
    generate_noon_plot(
        occ, labels, output_path=noon_path,
        active_lo=occ_active_lo, active_hi=occ_active_hi,
    )

    # 3. Cube files (optional)
    cube_dir = ""
    if generate_cubes:
        active_indices = [i for i in range(len(occ))
                          if occ_active_lo <= occ[i] <= occ_active_hi]
        grid_parts = cube_grid.split("x")
        nx, ny, nz = int(grid_parts[0]), int(grid_parts[1]), int(grid_parts[2])
        cube_dir = os.path.join(output_dir, "cubes")
        generate_orbital_cubes(
            mol, mo_coeff_loc,
            indices=active_indices,
            output_dir=cube_dir,
            nx=nx, ny=ny, nz=nz,
        )

    return {"report_path": report_path, "noon_path": noon_path, "cube_dir": cube_dir}


def _compute_energy_decomposition(cas, mf):
    """Compute orbital energy contributions: E_core, E_act, E_vir.

    Projects the Fock matrix into the UNO/LUO orbital basis, then sums
    orbital energies weighted by occupation numbers.

    Returns:
        Tuple of (E_core, E_act, E_vir) as floats.
    """
    occ_full = cas.occupations_full
    mo_coeff = cas.mo_coeff_full
    if occ_full is None or mo_coeff is None:
        return 0.0, 0.0, 0.0

    mol = mf.mol

    # Build Fock matrix: for UKS/UHF use total (alpha+beta) Fock
    try:
        # get_fock returns the Fock matrix for the given density
        dm_alpha, dm_beta = mf.make_rdm1()
        if isinstance(dm_alpha, np.ndarray) and dm_alpha.ndim == 2:
            # Restricted-style or alpha only
            if isinstance(mf.mo_coeff, (list, tuple)):
                # Unrestricted: average alpha and beta Fock
                veff_a = mf.get_veff(mol, dm_alpha)
                veff_b = mf.get_veff(mol, dm_beta)
                hcore = mf.get_hcore()
                fock_a = hcore + veff_a
                fock_b = hcore + veff_b
                fock = 0.5 * (fock_a + fock_b)
            else:
                fock = mf.get_fock()
        else:
            fock = mf.get_fock()
    except Exception:
        return 0.0, 0.0, 0.0

    # Project Fock into UNO basis: e_i = C_i^T F C_i
    n = min(len(occ_full), mo_coeff.shape[1])
    fock_uno = mo_coeff[:, :n].T @ fock @ mo_coeff[:, :n]
    orbital_energies = np.diag(fock_uno)

    occ = occ_full[:n]
    core_mask = occ > 1.98
    act_mask = (occ >= 0.02) & (occ <= 1.98)
    vir_mask = occ < 0.02

    e_core = float(np.sum(occ[core_mask] * orbital_energies[core_mask]))
    e_act = float(np.sum(occ[act_mask] * orbital_energies[act_mask]))
    e_vir = float(np.sum(occ[vir_mask] * orbital_energies[vir_mask]))

    return e_core, e_act, e_vir

def save_cas_state(
    cas,
    mol,
    mf,
    output_dir: str,
) -> str:
    """Save CAS full state to disk (HDF5).

    Saves:
    - ``outputs/scf/<descriptive>.chk`` (via mf.chkfile, written by PySCF during SCF)
    - ``outputs/orbitals/cas_data.h5`` (mo_coeff_full, occupations_full,
      orbital_labels_full, and active-space data)
    - ``outputs/scf/scf_info.json`` (energy, convergence)

    Parameters
    ----------
    cas : CAS
    mol : pyscf.gto.Mole
    mf : SCF object
    output_dir : str
        Root output directory (e.g. ``outputs/``).

    Returns
    -------
    str
        Path to the saved HDF5 file.
    """
    import json as _json
    import h5py

    scf_dir = os.path.join(output_dir, "scf")
    orbitals_dir = os.path.join(output_dir, "orbitals")
    os.makedirs(scf_dir, exist_ok=True)
    os.makedirs(orbitals_dir, exist_ok=True)

    # 1. Save SCF info
    # Compute energy decomposition: E_core, E_act, E_vir from orbital energies
    e_core, e_act, e_vir = _compute_energy_decomposition(cas, mf)

    scf_info = {
        "energy": float(mf.e_tot),
        "E_core": e_core,
        "E_act": e_act,
        "E_vir": e_vir,
        "E_tot": float(mf.e_tot),
        "converged": bool(mf.converged),
        "n_electrons": cas.n_electrons,
        "n_orbitals": cas.n_orbitals,
        "cpt_cas_type": cas.cpt_cas_type,
        "source_method": cas.source_method,
    }
    scf_info_path = os.path.join(scf_dir, "scf_info.json")
    with open(scf_info_path, "w") as f:
        _json.dump(scf_info, f, indent=2)

    # 3. Save orbital data to HDF5 with gzip-9 compression
    h5_path = os.path.join(orbitals_dir, "cas_data.h5")
    h5_kwargs = dict(compression="gzip", compression_opts=9)

    with h5py.File(h5_path, "w") as f:
        # Full orbital data
        if cas.mo_coeff_full is not None:
            f.create_dataset("mo_coeff_full", data=cas.mo_coeff_full, **h5_kwargs)
        if cas.occupations_full is not None:
            f.create_dataset("occupations_full", data=cas.occupations_full, **h5_kwargs)

        # Active-space orbital data
        if cas.mo_coeff_alpha is not None:
            f.create_dataset("mo_coeff_alpha", data=cas.mo_coeff_alpha, **h5_kwargs)
        if cas.mo_coeff_beta is not None:
            f.create_dataset("mo_coeff_beta", data=cas.mo_coeff_beta, **h5_kwargs)
        if cas.occupations is not None:
            f.create_dataset("occupations", data=cas.occupations, **h5_kwargs)

        # Labels (stored as variable-length strings)
        if cas.orbital_labels_full:
            f.create_dataset("orbital_labels_full",
                             data=[str(l) for l in cas.orbital_labels_full])
        if cas.orbital_labels:
            f.create_dataset("orbital_labels",
                             data=[str(l) for l in cas.orbital_labels])

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["n_electrons"] = cas.n_electrons
        meta.attrs["n_orbitals"] = cas.n_orbitals
        meta.attrs["cpt_cas_type"] = cas.cpt_cas_type
        meta.attrs["source_method"] = cas.source_method
        meta.attrs["description"] = cas.description

    logger.info("CAS state saved to %s", h5_path)
    return h5_path


def load_cas_state(case_dir: str):
    """Load CAS full state from disk.

    Restores mol + mf from chkfile, and CAS data from HDF5.

    Parameters
    ----------
    case_dir : str
        Case directory containing ``outputs/``.

    Returns
    -------
    tuple[CAS, mol, mf]
    """
    import h5py
    from pyscf import gto, scf as _scf

    output_dir = os.path.join(case_dir, "outputs")
    h5_path = os.path.join(output_dir, "orbitals", "cas_data.h5")

    # Find the chkfile (name is dynamically generated by init_computing)
    import glob
    scf_dir = os.path.join(output_dir, "scf")
    chk_candidates = [
        p for p in glob.glob(os.path.join(scf_dir, "*.chk"))
        if os.path.getsize(p) > 0
    ]

    if not chk_candidates:
        raise FileNotFoundError(f"No valid chkfile found in {scf_dir}/")

    # Sort by size descending (largest first as default)
    chk_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)

    if len(chk_candidates) == 1:
        chkfile = chk_candidates[0]
    else:
        print(f"  Multiple chkfiles found in {scf_dir}/:")
        for i, p in enumerate(chk_candidates):
            size_kb = os.path.getsize(p) / 1024
            print(f"    [{i}] {os.path.basename(p)}  ({size_kb:.1f} KB)")
        resp = input(f"  Select [0-{len(chk_candidates)-1}] (default=0): ").strip()
        if resp == "":
            idx = 0
        else:
            idx = int(resp)
        chkfile = chk_candidates[idx]

    print(f"  Using chkfile: {os.path.basename(chkfile)}")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"CAS data file not found: {h5_path}")

    # 1. Restore mol + mf from chkfile
    from pyscf import lib
    mol = lib.chkfile.load_mol(chkfile)

    # Determine SCF type from chkfile
    scf_data = lib.chkfile.load(chkfile, "scf")
    is_uhf = isinstance(scf_data.get("mo_coeff"), (list, np.ndarray)) and \
             np.ndim(scf_data["mo_coeff"]) == 3

    if is_uhf:
        mf = _scf.UHF(mol)
    else:
        mf = _scf.RHF(mol)
    mf.__dict__.update(scf_data)
    mf.chkfile = chkfile

    # 2. Load CAS from HDF5
    from . import CAS

    with h5py.File(h5_path, "r") as f:
        cas = CAS()

        # Arrays
        if "mo_coeff_full" in f:
            cas.mo_coeff_full = f["mo_coeff_full"][:]
        if "occupations_full" in f:
            cas.occupations_full = f["occupations_full"][:]
        if "mo_coeff_alpha" in f:
            cas.mo_coeff_alpha = f["mo_coeff_alpha"][:]
        if "mo_coeff_beta" in f:
            cas.mo_coeff_beta = f["mo_coeff_beta"][:]
        if "occupations" in f:
            cas.occupations = f["occupations"][:]

        # Labels
        if "orbital_labels_full" in f:
            cas.orbital_labels_full = [str(l) for l in f["orbital_labels_full"][:]]
        if "orbital_labels" in f:
            cas.orbital_labels = [str(l) for l in f["orbital_labels"][:]]

        # Metadata
        meta = f["metadata"]
        cas.n_electrons = int(meta.attrs["n_electrons"])
        cas.n_orbitals = int(meta.attrs["n_orbitals"])
        cas.cpt_cas_type = str(meta.attrs.get("cpt_cas_type", "uno"))
        cas.source_method = str(meta.attrs.get("source_method", ""))
        cas.description = str(meta.attrs.get("description", ""))
        cas.n_qubits = 2 * cas.n_orbitals

    logger.info("CAS state loaded from %s", h5_path)
    return cas, mol, mf
