#!/usr/bin/env python3
"""orbital_rotater.py — Rotate terminal S p-orbitals towards Fe-S bond direction.

This is a standalone script that does NOT import any apex_cas code.
Dependencies: pyscf, numpy, h5py, matplotlib, py3Dmol (optional for gallery).

The user must specify which orbital triplets to rotate via --rotate-group.
Each group is: ATOM_NAME IDX1 IDX2 IDX3, where ATOM_NAME must appear in the
Atom Roles table of the orbital_report.md (e.g. S3, S4), and IDX1/IDX2/IDX3
are orbital report idx values (from the orbital_report.md Orbitals table).
The atom name is used to look up the bonded Fe atom for the bond direction.

Usage:
    # Rotate S3, S4, S7, S8 5p orbitals (core, occ ≈ 2.0) in fe2s2:
    python scripts/orbital_rotater.py examples/fe2s2 \
        --rotate-group S3 58 84 81 \
        --rotate-group S4 60 86 77 \
        --rotate-group S7 59 74 79 \
        --rotate-group S8 61 82 72

    # With FCIDUMP generation:
    python scripts/orbital_rotater.py examples/fe2s2 \
        --rotate-group S3 58 84 81 \
        --rotate-group S4 60 86 77 \
        --rotate-group S7 59 74 79 \
        --rotate-group S8 61 82 72 \
        --generate-fcidump

    # After editing the generated selection.txt, generate FCIDUMP with custom active space:
    python scripts/orbital_rotater.py examples/fe2s2 \
        --generate-fcidump \
        --selection-file examples/fe2s2/outputs/orbitals/rotated/*_selection.txt
"""

import argparse
import json
import os
import re
import sys
import shutil
from collections import defaultdict

import h5py
import numpy as np


# ============================================================================
# Step 1: Load data
# ============================================================================


def load_h5_data(h5_path):
    """Load all data from the HDF5 file.

    Returns dict with keys:
        mo_coeff_full, occupations_full, orbital_labels_full,
        mo_coeff_alpha, mo_coeff_beta, occupations, orbital_labels,
        active_indices, projection_weights, projection_weights_metal,
        projection_weights_bridging,
        metadata (dict of attrs)
    """
    data = {}
    with h5py.File(h5_path, "r") as f:
        # Top-level datasets
        for key in ["mo_coeff_full", "occupations_full",
                     "mo_coeff_alpha", "mo_coeff_beta",
                     "occupations"]:
            if key in f:
                data[key] = f[key][()]

        # String datasets
        for key in ["orbital_labels_full", "orbital_labels"]:
            if key in f:
                arr = f[key][()]
                data[key] = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]

        # Metadata group
        meta = f.get("metadata")
        if meta is not None:
            data["metadata"] = dict(meta.attrs)
            # Datasets inside metadata
            for key in ["active_indices", "projection_weights",
                         "projection_weights_metal", "projection_weights_bridging"]:
                if key in meta:
                    data[key] = meta[key][()]
            # active_indices -> int
            if "active_indices" in data:
                data["active_indices"] = np.array(data["active_indices"], dtype=int)

    return data


def rebuild_mf_from_chk(chk_path, h5_metadata):
    """Rebuild a PySCF mf object from chkfile + metadata.

    Returns (mol, mf).
    """
    from pyscf import lib, scf, dft

    # Restore mol
    mol = lib.chkfile.load_mol(chk_path)

    # Read SCF data from chkfile
    scf_data = lib.chkfile.load(chk_path, "scf")

    # Determine SCF method from metadata
    scf_method = str(h5_metadata.get("scf_method", "uks"))
    xc_functional = str(h5_metadata.get("xc_functional", "BP86"))
    relativistic = str(h5_metadata.get("relativistic", "none"))
    solvation_model = str(h5_metadata.get("solvation_model", "none"))
    solvation_epsilon = float(h5_metadata.get("solvation_epsilon", 4.0))
    frac_occ = bool(h5_metadata.get("frac_occ", False))
    smearing_method = str(h5_metadata.get("smearing_method", "none"))
    smearing_sigma = float(h5_metadata.get("smearing_sigma", 0.01))

    # Build base SCF object
    if scf_method == "uhf":
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc_functional

    # Relativistic correction
    if relativistic == "sf-x2c":
        mf = mf.sfx2c1e()
    elif relativistic == "dkh":
        mol.set(relativistic="DKH")
        if scf_method == "uks":
            mf = dft.UKS(mol)
            mf.xc = xc_functional
        else:
            mf = scf.UHF(mol)

    # Solvation model
    if solvation_model == "ddcosmo":
        from pyscf import solvent
        mf = solvent.ddcosmo.ddcosmo_for_scf(mf)
        mf.with_solvent.epsilon = solvation_epsilon

    # Fractional occupation
    if frac_occ:
        mf = scf.addons.frac_occ(mf)

    # Smearing
    if smearing_method != "none":
        mf = scf.addons.smearing_(mf, sigma=smearing_sigma, method=smearing_method)

    # Inject converged SCF data
    mf.__dict__.update(scf_data)
    mf.chkfile = chk_path

    return mol, mf


# ============================================================================
# Step 2: Identify terminal S and their p-orbitals
# ============================================================================

def parse_atom_roles(report_path):
    """Parse the 'Atom Roles' table from orbital_report.md.

    Returns list of dicts:
        [{atom: "S3", element: "S", index: 2, role: "terminal", bonded_to: "Fe1"}, ...]
    Only terminal entries are returned.
    """
    terminal_entries = []
    in_atom_roles = False

    with open(report_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("## Atom Roles"):
                in_atom_roles = True
                continue
            if in_atom_roles and stripped.startswith("## "):
                break
            if not in_atom_roles:
                continue
            # Match table rows with 'terminal' role
            # | S3 | S | 2 | terminal | Fe1 |
            m = re.match(
                r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*terminal\s*\|\s*(\w+)\s*\|',
                stripped,
            )
            if m:
                terminal_entries.append({
                    "atom": m.group(1),
                    "element": m.group(2),
                    "index": int(m.group(3)),
                    "role": "terminal",
                    "bonded_to": m.group(4),
                })

    return terminal_entries


def parse_metal_indices(report_path):
    """Parse metal atom indices from the Atom Roles table.

    Returns dict: {atom_label: atom_index} for metals.
    """
    metals = {}
    in_atom_roles = False

    with open(report_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("## Atom Roles"):
                in_atom_roles = True
                continue
            if in_atom_roles and stripped.startswith("## "):
                break
            if not in_atom_roles:
                continue
            m = re.match(
                r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*metal\s*\|',
                stripped,
            )
            if m:
                metals[m.group(1)] = int(m.group(3))

    return metals


def find_valence_p_ao_indices(mol, atom_index):
    """Find the valence p AO basis function indices for a given atom.

    For S (Z=16) with tzp-dkh basis, the valence p shell is 3p.
    This function finds the AO indices of the valence p shell by looking
    at the principal quantum number of the p AOs and selecting the
    one matching the physical valence shell (3p for S, 2p for C, etc.).

    Args:
        mol: PySCF Mole object
        atom_index: 0-based atom index

    Returns:
        dict with keys 'x', 'y', 'z' mapping to AO indices, or None.
        Also returns the shell label string (e.g. "3p").
    """
    aoslices = mol.aoslice_by_atom()
    ao_labels = mol.ao_labels()
    ao_start, ao_end = aoslices[atom_index][2], aoslices[atom_index][3]

    elem = mol.atom_symbol(atom_index)
    # Valence p quantum numbers for common elements
    valence_p = {"S": 3, "O": 2, "N": 2, "C": 2, "Se": 4, "Fe": None, "Mo": None}
    target_n = valence_p.get(elem)

    # Collect all p AOs on this atom, grouped by principal quantum number
    p_shells = defaultdict(dict)
    for j in range(ao_start, ao_end):
        label = ao_labels[j]
        # PySCF label format: "2 S 3px   " or "2 S 3pz^2 "
        m = re.search(r'(\d+)p([xyz])', label)
        if m:
            shell_n = int(m.group(1))
            axis = m.group(2)
            p_shells[shell_n][axis] = j

    # Pick the shell: use target_n if known, otherwise pick the smallest
    # complete p shell that is >= the physical valence shell
    if target_n is not None and target_n in p_shells:
        shell = p_shells[target_n]
        if all(a in shell for a in ("x", "y", "z")):
            return shell, f"{target_n}p"

    # Fallback: find the smallest complete p shell
    for n in sorted(p_shells.keys()):
        shell = p_shells[n]
        if all(a in shell for a in ("x", "y", "z")):
            return shell, f"{n}p"

    return None, None


def find_terminal_s_p_orbitals(mol, mo_coeff, occupations, terminal_s_atoms):
    """Find the MO triplets for terminal S valence p rotation.

    For each terminal S:
      1. Locate its valence p AO basis function indices (e.g. 3px, 3py, 3pz)
      2. Compute Mulliken p-AO projection weight for each MO:
         p_weight_i = sum_{j in p_ao} C[j,i] * (S @ C)[j,i]
      3. Select the 3 MOs with the highest p-AO Mulliken weight

    The rotation is then applied at the MO level (rotating 3 MO columns),
    which preserves MO orthonormality and keeps all integrals valid.

    Args:
        mol: PySCF Mole object
        mo_coeff: MO coefficient matrix (nao, nmo)
        occupations: occupation numbers (nmo,)
        terminal_s_atoms: list of dicts from parse_atom_roles

    Returns list of dicts with p-orbital triplet info.
    """
    S = mol.intor_symmetric("int1e_ovlp")
    SC = S @ mo_coeff  # precompute for Mulliken analysis

    results = []

    for ts in terminal_s_atoms:
        atom_name = ts["atom"]
        atom_idx = ts["index"]

        # Find valence p AO indices
        p_ao_dict, shell_label = find_valence_p_ao_indices(mol, atom_idx)

        if p_ao_dict is None:
            print(f"  WARNING: No valence p shell found for {atom_name} (idx={atom_idx})")
            continue

        p_ao_indices = [p_ao_dict["x"], p_ao_dict["y"], p_ao_dict["z"]]
        print(f"  {atom_name}: valence p shell = {shell_label}, "
              f"AO indices = {p_ao_indices}")

        # Mulliken p-AO weight for each MO:
        # p_weight_i = sum_{j in p_ao} C[j,i] * (S @ C)[j,i]
        nmo = mo_coeff.shape[1]
        p_weights = np.zeros(nmo)
        for j in p_ao_indices:
            p_weights += mo_coeff[j, :] * SC[j, :]

        # Select top 3 MOs by Mulliken p-AO weight
        top3_indices = np.argsort(p_weights)[-3:][::-1]
        top3_weights = p_weights[top3_indices]
        top3_occs = occupations[top3_indices]

        print(f"    Top 3 MOs by Mulliken p-AO weight:")
        for k in range(3):
            print(f"      MO {top3_indices[k]}: weight={top3_weights[k]:.4f}, "
                  f"occ={top3_occs[k]:.6f}")

        total_p_weight = np.sum(p_weights)
        captured = np.sum(top3_weights)
        print(f"    Total p-AO weight across all MOs: {total_p_weight:.4f}")
        print(f"    Captured by top 3 MOs: {captured:.4f} ({100*captured/total_p_weight:.1f}%)")

        results.append({
            "atom": atom_name,
            "atom_index": atom_idx,
            "fe_label": ts["bonded_to"],
            "fe_index": None,  # filled later
            "shell": shell_label,
            "p_indices": [int(x) for x in top3_indices],
            "p_ao_indices": p_ao_indices,
            "p_weights": [float(x) for x in top3_weights],
        })

    return results


# ============================================================================
# Step 3: Build rotation matrix
# ============================================================================

def build_rotation_matrix(bond_vec):
    """Build a 3x3 rotation matrix from S→Fe bond vector.

    The first axis (row 0) aligns with the bond direction.
    The other two axes are constructed via Gram-Schmidt.

    Args:
        bond_vec: 3-element array, S→Fe direction (pos_S - pos_Fe)

    Returns:
        R: 3x3 rotation matrix. R[0] = bond direction.
    """
    e1 = bond_vec / np.linalg.norm(bond_vec)

    # Find a vector not parallel to e1 for Gram-Schmidt
    if abs(e1[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])

    e2 = v - np.dot(v, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    e3 = np.cross(e1, e2)
    e3 = e3 / np.linalg.norm(e3)

    R = np.array([e1, e2, e3])  # shape (3, 3)
    return R


# ============================================================================
# Step 4: Apply rotation
# ============================================================================

def apply_rotation(mo_coeff, rotation_groups, mol):
    """Apply MO-level rotation to user-specified orbital triplets.

    For each group (atom + 3 orbital indices), rotates the 3 MO columns:

        C_new[:, p_indices] = C_old[:, p_indices] @ R.T

    where R = [e1, e2, e3] with e1 along S→Fe bond direction.

    This is a unitary rotation of 3 MO columns, preserving MO orthonormality
    (C^T S C = I) and keeping all integrals valid for CASCI.

    Returns (mo_coeff_new, rotation_info).
    """
    mo_coeff_new = mo_coeff.copy()
    rotation_info = []

    for tg in rotation_groups:
        s_idx = tg["atom_index"]
        fe_idx = tg["fe_index"]
        p_indices = tg["p_indices"]  # [idx1, idx2, idx3]

        bond_vec = mol.atom_coord(s_idx) - mol.atom_coord(fe_idx)
        R = build_rotation_matrix(bond_vec)

        # MO-level rotation: C_new[:, p_indices] = C_old[:, p_indices] @ R.T
        old_cols = mo_coeff_new[:, p_indices]  # (nao, 3)
        new_cols = old_cols @ R.T
        mo_coeff_new[:, p_indices] = new_cols

        rotation_info.append({
            "atom": tg["atom"],
            "p_indices": p_indices,
            "R": R,
            "bond_vec": bond_vec,
        })
        print(f"  Rotated {tg['atom']} MOs (indices {p_indices})")
        print(f"    Bond vector (S→Fe): {bond_vec}")
        print(f"    Bond direction:     {R[0]}")
        print(f"    Perp 1:             {R[1]}")
        print(f"    Perp 2:             {R[2]}")

    return mo_coeff_new, rotation_info


# ============================================================================
# Step 5: Recompute occupation
# ============================================================================

def recompute_occupations(mo_coeff, mf):
    """Recompute occupation numbers from density matrix.

    occ_i = C_i^T S D_total S C_i  where D_total = D_alpha + D_beta.

    Returns ndarray of shape (nmo,).
    """
    mol = mf.mol
    dm = mf.make_rdm1()
    if isinstance(dm, (list, tuple)) or (isinstance(dm, np.ndarray) and dm.ndim == 3):
        dm_total = dm[0] + dm[1]
    else:
        dm_total = 2.0 * dm

    S = mol.intor_symmetric("int1e_ovlp")
    SDS = S @ dm_total @ S

    # occ_i = sum_mu C[mu,i] * (SDS @ C)[mu,i] = diag(C^T SDS C)
    occ = np.einsum("ji,jk,ki->i", mo_coeff, SDS, mo_coeff)
    return occ


# ============================================================================
# Step 6: Update labels
# ============================================================================

def update_labels(orbital_labels_full, rotation_info):
    """Update orbital labels for rotated orbital triplets.

    For each group, the 3 rotated MOs get relabeled:
      {atom}_bond   — first MO (now aligned with bond)
      {atom}_perp1  — second MO (perpendicular direction 1)
      {atom}_perp2  — third MO (perpendicular direction 2)
    """
    labels = list(orbital_labels_full)
    new_suffixes = ["_bond", "_perp1", "_perp2"]

    for ri in rotation_info:
        atom = ri["atom"]
        p_indices = ri["p_indices"]
        for i, idx in enumerate(p_indices):
            old_label = labels[idx]
            labels[idx] = f"{atom}{new_suffixes[i]}"
            print(f"  Label: {old_label} -> {labels[idx]}")

    return labels


# ============================================================================
# Step 7: --select-bond-dir
# ============================================================================

def select_bond_dir(active_indices, occupations_full, rotation_info, orbital_labels_full):
    """Optionally add p_bond orbitals to active space.

    For each terminal S, if the p_bond orbital has occ < 1.98, add it to active.
    Update n_electrons accordingly.

    Returns (updated_active_indices, updated_n_electrons, messages).
    """
    active_set = set(active_indices)
    messages = []
    n_electrons_delta = 0

    for ri in rotation_info:
        p_indices = ri["p_indices"]
        if len(p_indices) < 1:
            continue
        bond_idx = p_indices[0]  # _bond is the first (highest p-AO weight)
        label = orbital_labels_full[bond_idx]
        occ = occupations_full[bond_idx]

        if occ >= 1.98:
            messages.append(
                f"  {label} (idx={bond_idx}, occ={occ:.4f}): still in core (occ >= 1.98), "
                f"not forced into active space."
            )
        elif bond_idx in active_set:
            messages.append(
                f"  {label} (idx={bond_idx}, occ={occ:.4f}): already in active space."
            )
        else:
            active_set.add(bond_idx)
            n_electrons_delta += int(round(occ))
            messages.append(
                f"  {label} (idx={bond_idx}, occ={occ:.4f}): ADDED to active space "
                f"(+{int(round(occ))} electrons)."
            )

    new_active = sorted(active_set)
    return new_active, n_electrons_delta, messages


# ============================================================================
# Step 8: Generate output files
# ============================================================================

def save_rotated_h5(output_dir, stem, original_data, mo_coeff_new, occ_new,
                    labels_new, active_indices_new=None, n_electrons_new=None):
    """Save rotated data to a new h5 file in the same format as the original."""
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"{stem}_cas_data.h5")
    h5_kwargs = dict(compression="gzip", compression_opts=9)

    with h5py.File(h5_path, "w") as f:
        # Full orbital data (rotated)
        f.create_dataset("mo_coeff_full", data=mo_coeff_new, **h5_kwargs)
        f.create_dataset("occupations_full", data=occ_new, **h5_kwargs)

        # Active-space data: extract from the rotated full data
        ai = active_indices_new if active_indices_new is not None else original_data.get("active_indices")
        if ai is not None:
            ai = np.array(ai, dtype=int)
            mo_alpha = mo_coeff_new[:, ai]
            occ_active = occ_new[ai]
            labels_active = [labels_new[i] for i in ai]

            f.create_dataset("mo_coeff_alpha", data=mo_alpha, **h5_kwargs)
            f.create_dataset("mo_coeff_beta", data=mo_alpha, **h5_kwargs)  # same for UNO
            f.create_dataset("occupations", data=occ_active, **h5_kwargs)
            f.create_dataset("orbital_labels", data=[str(l) for l in labels_active])

        # Labels
        f.create_dataset("orbital_labels_full", data=[str(l) for l in labels_new])

        # Metadata: copy from original, update if needed
        meta = f.create_group("metadata")
        orig_meta = original_data.get("metadata", {})
        for key, val in orig_meta.items():
            try:
                meta.attrs[key] = val
            except TypeError:
                meta.attrs[key] = str(val)

        # Update n_electrons and n_orbitals if changed
        if n_electrons_new is not None:
            meta.attrs["n_electrons"] = n_electrons_new
        if active_indices_new is not None:
            meta.attrs["n_orbitals"] = len(active_indices_new)
            meta.create_dataset("active_indices", data=np.array(active_indices_new, dtype=int))

        # Projection weights (copy from original)
        for key in ["projection_weights", "projection_weights_metal", "projection_weights_bridging"]:
            if key in original_data:
                meta.create_dataset(key, data=original_data[key], **h5_kwargs)

        # Also copy active_indices if we didn't update
        if active_indices_new is None and "active_indices" in original_data:
            meta.create_dataset("active_indices", data=original_data["active_indices"])

    print(f"  Saved: {h5_path}")
    return h5_path


def generate_selection_file(output_dir, stem, active_indices, n_electrons):
    """Generate a selection.txt file."""
    path = os.path.join(output_dir, f"{stem}_selection.txt")
    indices = sorted(active_indices)
    norb = len(indices)

    lines = [
        "# Active space selection (auto-generated by orbital_rotater)",
        "# Edit n-electrons and indices, then run: orbital_rotater.py <case_dir> --generate-fcidump --selection-file <this_file>",
        f"n-electrons: {n_electrons}",
        f"n-orbital: {norb}",
    ]
    for i in range(0, len(indices), 10):
        chunk = indices[i:i + 10]
        lines.append("\t".join(str(idx) for idx in chunk))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved: {path}")
    return path


def parse_selection_file(selection_path):
    """Parse a selection.txt file and return (sorted_indices, n_electrons).

    Expected format:
        # comments
        n-electrons: <int>
        n-orbital: <int>
        88\t89\t90\t...
    """
    n_electrons = None
    indices = []

    with open(selection_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("n-electrons:"):
                n_electrons = int(line.split(":")[1].strip())
            elif line.startswith("n-orbital:"):
                pass  # informational, we count from indices
            else:
                # Tab-separated orbital indices
                parts = line.split()
                indices.extend(int(x) for x in parts)

    if n_electrons is None:
        raise ValueError(f"n-electrons not found in {selection_path}")
    if not indices:
        raise ValueError(f"No orbital indices found in {selection_path}")

    return sorted(set(indices)), n_electrons


def generate_noon_plot(occupations, labels, output_path,
                       active_lo=0.02, active_hi=1.98):
    """Generate NOON dot-line plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nmo = len(occupations)
    x = np.arange(nmo)

    active_mask = np.array([(active_lo <= occ <= active_hi) for occ in occupations])
    active_indices = np.where(active_mask)[0]

    fig, ax = plt.subplots(figsize=(max(12, nmo * 0.06), 5))
    ax.plot(x, occupations, color="#333333", linewidth=0.8, zorder=1)
    ax.scatter(x[~active_mask], occupations[~active_mask],
               color="#aaaaaa", s=10, zorder=2, label="core/virtual")
    ax.scatter(x[active_mask], occupations[active_mask],
               color="#d62728", s=14, zorder=3, label="active")

    ax.axhline(y=active_hi, color="blue", linestyle="--", linewidth=0.5, alpha=0.7,
               label=f"threshold hi={active_hi}")
    ax.axhline(y=active_lo, color="blue", linestyle="--", linewidth=0.5, alpha=0.7,
               label=f"threshold lo={active_lo}")

    # Annotate active orbitals
    for i in active_indices[:50]:
        lab = labels[i] if labels and i < len(labels) else str(i)
        ax.annotate(lab, (i, occupations[i]),
                    fontsize=4, rotation=90, textcoords="offset points",
                    xytext=(3, 5), ha="left", va="bottom")

    # Highlight rotated orbitals
    for i, lab in enumerate(labels):
        if "_bond" in lab or "_perp1" in lab or "_perp2" in lab:
            ax.scatter([i], [occupations[i]], color="#2ca02c", s=40, zorder=5,
                       edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Orbital index")
    ax.set_ylabel("Occupation number")
    ax.set_title("NOON (after rotation)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.05, 2.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_orbital_report(output_dir, stem, mol, mo_coeff, occupations, labels,
                            active_indices, original_report_path, rotation_info,
                            metadata_dict):
    """Generate a simplified orbital_report.md for rotated data.

    Copies metadata, atom roles, and AO shell analysis from the original report.
    Regenerates the Orbitals table with updated occ and labels.
    """
    import re

    output_path = os.path.join(output_dir, f"{stem}_orbital_report.md")

    # Read original report to extract static sections
    with open(original_report_path) as f:
        original_text = f.read()

    # Extract sections we want to copy verbatim
    # 1. Everything from start through AO Shell Analysis
    # 2. Skip the old Orbitals table and generate a new one

    lines = []

    # -- Metadata section (updated) --
    nmo = len(occupations)
    n_core = sum(1 for o in occupations if o > 1.98)
    n_active = len(active_indices) if active_indices is not None else 0
    n_virtual = nmo - n_core - sum(1 for o in occupations if 0.02 <= o <= 1.98)

    lines.append("# Orbital Report (rotated)")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|-----|-------|")
    lines.append(f"| cas_type | {metadata_dict.get('cpt_cas_type', '')} |")
    lines.append(f"| selection_method | {metadata_dict.get('selection_method', '')} |")
    lines.append(f"| n_total | {nmo} |")
    lines.append(f"| n_core | {n_core} |")
    lines.append(f"| n_active | {n_active} |")
    lines.append(f"| n_virtual | {n_virtual} |")
    lines.append(f"| threshold_active_lo | 0.02 |")
    lines.append(f"| threshold_active_hi | 1.98 |")
    lines.append(f"| total_electrons | {round(float(np.sum(occupations)), 2)} |")
    lines.append("| rotation | terminal S p-orbitals rotated to Fe-S bond direction |")
    lines.append("")

    # -- Copy Atom Roles and AO Shell Analysis from original --
    # Extract from "## Atom Roles" to just before "## Orbitals"
    sections = re.split(r'(^## )', original_text, flags=re.MULTILINE)
    atom_roles_text = ""
    ao_shell_text = ""
    for i in range(1, len(sections), 2):
        section_header = sections[i]
        section_body = sections[i + 1] if i + 1 < len(sections) else ""
        if section_header.startswith("## Atom Roles"):
            atom_roles_text = "## " + section_header[3:] + section_body
        elif section_header.startswith("## AO Shell"):
            ao_shell_text = "## " + section_header[3:] + section_body

    if atom_roles_text:
        lines.append(atom_roles_text.rstrip())
        lines.append("")
    if ao_shell_text:
        lines.append(ao_shell_text.rstrip())
        lines.append("")

    # -- Rotation info section --
    lines.append("## Rotation Details")
    lines.append("")
    lines.append("| atom | MO_indices | bond_direction |")
    lines.append("|------|------------|----------------|")
    for ri in rotation_info:
        p_idx = ri["p_indices"]
        bd = ri["R"][0]
        lines.append(
            f"| {ri['atom']} | {p_idx} "
            f"| [{bd[0]:.3f}, {bd[1]:.3f}, {bd[2]:.3f}] |"
        )
    lines.append("")

    # -- Orbitals table --
    active_set = set(active_indices) if active_indices is not None else set()
    lines.append("## Orbitals")
    lines.append("")
    lines.append("<!-- Edit the 'selected' column to choose active orbitals, then run: apex-cas fcidump -->")
    lines.append("| idx | occ | block | label | selected | rotated |")
    lines.append("|-----|------|-------|-------|----------|---------|")

    rotated_set = set()
    for ri in rotation_info:
        rotated_set.update(ri["p_indices"])

    for i in range(nmo):
        occ = float(occupations[i])
        if occ > 1.98:
            block = "core"
        elif occ < 0.02:
            block = "virtual"
        else:
            block = "active"
        sel_str = "true" if i in active_set else "false"
        rot_str = "**YES**" if i in rotated_set else "no"
        lab = labels[i] if i < len(labels) else f"orb_{i}"
        lines.append(f"| {i} | {occ:.6f} | {block} | {lab} | {sel_str} | {rot_str} |")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {output_path}")
    return output_path


def generate_cube_files(mol, mo_coeff, output_dir, stem, labels,
                        active_indices, rotation_info, cube_grid="80x80x80"):
    """Generate cube files for active and rotated orbitals."""
    from pyscf.tools import cubegen

    cubes_dir = os.path.join(output_dir, "cubes")
    os.makedirs(cubes_dir, exist_ok=True)

    # Collect indices to generate cubes for
    cube_indices = set()
    if active_indices is not None:
        cube_indices.update(active_indices)
    for ri in rotation_info:
        cube_indices.update(ri["p_indices"])

    cube_paths = []
    for idx in sorted(cube_indices):
        lab = labels[idx] if idx < len(labels) else f"orb_{idx:04d}"
        # Sanitize label for filename
        safe_lab = lab.replace("^", "").replace(" ", "_")
        cube_name = f"{stem}_orb_{idx:04d}_{safe_lab}.cube"
        cube_path = os.path.join(cubes_dir, cube_name)
        print(f"  Generating cube: {cube_name}")
        try:
            cubegen.orbital(mol, cube_path, mo_coeff[:, idx])
            cube_paths.append(cube_path)
        except Exception as e:
            print(f"  WARNING: Failed to generate cube for orbital {idx}: {e}")

    print(f"  Generated {len(cube_paths)} cube files in {cubes_dir}")
    return cube_paths


def generate_orbital_gallery(cube_paths, labels, occupations, indices,
                             output_dir, stem, isovalue=0.04, opacity=0.85):
    """Generate an HTML orbital gallery with 3Dmol.js."""
    gallery_name = f"{stem}_orbital_gallery.html"
    html_path = os.path.join(output_dir, gallery_name)

    n = len(cube_paths)
    if n == 0:
        print("  No cube files for gallery, skipping.")
        return ""

    # Build display labels
    display_labels = []
    display_occs = []
    for i, idx in enumerate(indices):
        lab = labels[idx] if idx < len(labels) else f"orb_{idx}"
        display_labels.append(f"{idx}: {lab}")
        display_occs.append(float(occupations[idx]))

    # Relative paths from HTML to cubes
    cube_rel_paths = []
    for p in cube_paths:
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(output_dir))
        cube_rel_paths.append(rel.replace("\\", "/"))

    js_cdn = "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"
    cube_paths_js = "[" + ", ".join(f'"{p}"' for p in cube_rel_paths) + "]"

    nav_items = "\n".join(
        f'<button onclick="show({i})" id="btn_{i}" '
        f'title="{display_labels[i]} (occ={display_occs[i]:.4f})">'
        f'{display_labels[i]}'
        f' <span style="color:#888;">occ={display_occs[i]:.4f}</span></button>\n'
        for i in range(n)
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Orbital Gallery (rotated)</title>
<script src="{js_cdn}"></script>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; width:100%; height:100%; background:black; font-family:monospace; overflow:hidden; }}
body {{ display:flex; }}
#sidebar {{
    width: 300px; min-width: 300px; height: 100vh;
    background: #1a1a1a; border-right: 1px solid #333;
    display: flex; flex-direction: column; overflow: hidden;
}}
#sidebar-title {{
    padding: 8px 10px; color: #aaa; font-size: 12px;
    background: #222; border-bottom: 1px solid #333; flex-shrink: 0;
}}
#sidebar-title span {{ color: #4477AA; }}
#btn-list {{
    flex: 1; overflow-y: auto; padding: 4px;
}}
#btn-list button {{
    display: block; width: 100%; text-align: left;
    margin: 1px 0; padding: 5px 8px; cursor: pointer;
    font-family: monospace; font-size: 11px; line-height: 1.4;
    border: 1px solid #444; background: #2a2a2a; color: #ddd; border-radius: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
#btn-list button:hover {{ background: #444; }}
#btn-list button.active {{ background: #4477AA; color: white; border-color: #4477AA; }}
#btn-list button.active span {{ color: #ccc; }}
#main {{
    flex: 1; display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}}
#info {{
    padding: 6px 12px; background: #1a1a1a; border-bottom: 1px solid #333;
    color: #ccc; font-size: 13px; text-align: center; flex-shrink: 0;
}}
#viewer-wrap {{
    flex: 1; display: flex; align-items: center; justify-content: center;
    position: relative;
}}
#viewer {{ width: 100%; height: 100%; }}
#loading {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #888; font-size: 16px; z-index: 200; display: none;
    pointer-events: none;
}}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-title">Orbital Gallery (rotated) &mdash; <span id="counter">1 / {n}</span></div>
  <div id="btn-list">{nav_items}</div>
</div>
<div id="main">
  <div id="info"></div>
  <div id="viewer-wrap">
    <div id="viewer"></div>
    <div id="loading">Loading...</div>
  </div>
</div>
<script>
var cubePaths = {cube_paths_js};
var currentViewer = null;
var n = {n};
var isoval = {isovalue};
var opacity = {opacity};

function loadOrbital(idx) {{
    var viewer = document.getElementById("viewer");
    viewer.innerHTML = "";
    currentViewer = $3Dmol.createViewer(viewer, {{backgroundColor: "black"}});
    var cubeUrl = cubePaths[idx];
    document.getElementById("loading").style.display = "block";
    fetch(cubeUrl).then(r => r.text()).then(data => {{
        currentViewer.addVolumetricData(data, "cube",
            {{isoval: isoval, color: "blue", opacity: opacity, smoothness: 5}});
        currentViewer.addVolumetricData(data, "cube",
            {{isoval: -isoval, color: "red", opacity: opacity, smoothness: 5}});
        currentViewer.addModel(data, "cube");
        var elColors = {{Fe:'#E06633',Mo:'#54B5B5',Co:'#F090A0',Ni:'#50D050',Cu:'#C88033',Mn:'#9C7AC7',Cr:'#8A99C7'}};
        currentViewer.setStyle({{}}, {{stick: {{radius: 0.05, color: '#888'}}}});
        for (var el in elColors) {{
            currentViewer.addStyle({{elem: el}}, {{sphere: {{scale: 0.25, color: elColors[el]}}, stick: {{color: elColors[el]}}}});
        }}
        currentViewer.addStyle({{elem: 'S'}}, {{sphere: {{scale: 0.25, color: '#FFFF30'}}}});
        currentViewer.addStyle({{elem: 'O'}}, {{sphere: {{scale: 0.25, color: '#FF0D0D'}}}});
        currentViewer.addStyle({{elem: 'N'}}, {{sphere: {{scale: 0.25, color: '#3050F8'}}}});
        currentViewer.addStyle({{elem: 'C'}}, {{sphere: {{scale: 0.2, color: '#909090'}}}});
        currentViewer.addStyle({{elem: 'H'}}, {{sphere: {{scale: 0.1, color: '#FFFFFF'}}}});
        currentViewer.zoomTo();
        currentViewer.render();
        document.getElementById("loading").style.display = "none";
    }}).catch(err => {{
        document.getElementById("loading").innerText = "Error: " + err.message;
    }});
}}

function show(idx) {{
    for (var i = 0; i < n; i++) {{
        var btn = document.getElementById("btn_" + i);
        if (btn) btn.className = (i === idx) ? "active" : "";
    }}
    var btn = document.getElementById("btn_" + idx);
    document.getElementById("info").innerText = btn.textContent;
    document.getElementById("counter").innerText = (idx + 1) + " / " + n;
    btn.scrollIntoView({{block: "nearest", behavior: "smooth"}});
    loadOrbital(idx);
}}

show(0);

document.addEventListener("keydown", function(e) {{
    var current = 0;
    for (var i = 0; i < n; i++) {{
        if (document.getElementById("btn_" + i).className === "active") {{ current = i; break; }}
    }}
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {{
        show(Math.min(current + 1, n - 1)); e.preventDefault();
    }} else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {{
        show(Math.max(current - 1, 0)); e.preventDefault();
    }}
}});
</script>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html)

    # Generate server launcher
    server_script = f"""#!/usr/bin/env python3
import http.server, socketserver, webbrowser, os, sys
PORT = 0
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    port = httpd.server_address[1]
    html_files = [f for f in os.listdir(DIRECTORY) if f.endswith("_orbital_gallery.html")]
    url = f"http://localhost:{{port}}/{{html_files[0]}}" if html_files else f"http://localhost:{{port}}/"
    print(f"Serving {{DIRECTORY}}")
    print(f"Open: {{url}}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nServer stopped.")
"""
    server_path = os.path.join(output_dir, f"{stem}_orbital_gallery_server.py")
    with open(server_path, "w") as f:
        f.write(server_script)

    print(f"  Saved: {html_path}")
    print(f"  Saved: {server_path}")
    return html_path


# ============================================================================
# Step 9: --generate-fcidump
# ============================================================================

def generate_fcidump(mol, mf, mo_coeff, active_indices, n_electrons, target_spin,
                     output_dir, stem):
    """Generate FCIDUMP using PySCF CASCI integrals."""
    from pyscf import mcscf, ao2mo
    from pyscf.mcscf import casci
    from pyscf.tools import fcidump as fcidump_mod

    fcidump_dir = os.path.join(output_dir, "fcidump")
    os.makedirs(fcidump_dir, exist_ok=True)

    selected_indices = sorted(active_indices)
    n_active = len(selected_indices)

    # Determine frozen core indices (occ > 1.98 and not in active)
    occupations = recompute_occupations(mo_coeff, mf)
    selected_set = set(selected_indices)
    frozen_core_idx = sorted([
        i for i in range(len(occupations))
        if occupations[i] > 1.98 and i not in selected_set
    ])
    ncore = len(frozen_core_idx)

    ms2 = int(round(2 * target_spin))
    nalpha = (n_electrons + ms2) // 2
    nbeta = (n_electrons - ms2) // 2

    print(f"  FCIDUMP: n_active={n_active}, n_electrons={n_electrons}, "
          f"ncore={ncore}, ms2={ms2}, (nalpha, nbeta)=({nalpha}, {nbeta})")

    # Reorder MOs: [frozen_core | active]
    mo_frozen = mo_coeff[:, frozen_core_idx]
    mo_active = mo_coeff[:, selected_indices]
    mo_reordered = np.hstack([mo_frozen, mo_active])

    # Build CASCI object
    casci_obj = casci.CASCI(mf, n_active, (nalpha, nbeta), ncore=ncore)
    casci_obj.mo_coeff = mo_reordered

    # Get integrals
    print("  Computing h1eff and h2eff...")
    h1eff, e_core = casci_obj.get_h1eff()
    eri = casci_obj.get_h2eff()
    eri = ao2mo.restore(8, eri, n_active)

    # Write FCIDUMP
    fcidump_path = os.path.join(fcidump_dir, f"FCIDUMP.{stem}")
    fcidump_mod.from_integrals(
        fcidump_path, h1eff, eri,
        n_active, n_electrons,
        nuc=0.0, ms=ms2,
    )

    # Write .ecore file
    ecore_path = fcidump_path + ".ecore"
    with open(ecore_path, "w") as f:
        f.write(f"{e_core:.15f}\n")

    print(f"  Saved: {fcidump_path}")
    print(f"  Saved: {ecore_path} (E_core = {e_core:.10f})")

    return fcidump_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rotate terminal S p-orbitals towards Fe-S bond direction"
    )
    parser.add_argument(
        "case_dir",
        help="Case directory (e.g. examples/fe2s2) containing outputs/",
    )
    parser.add_argument(
        "--rotate-group",
        nargs=4,
        action="append",
        default=[],
        metavar=("ATOM", "IDX1", "IDX2", "IDX3"),
        help="Atom name and 3 orbital idx values from orbital_report.md. "
             "Repeat for each terminal S atom. "
             "Example: --rotate-group S3 58 84 81 --rotate-group S4 60 86 77",
    )
    parser.add_argument(
        "--select-bond-dir",
        action="store_true",
        help="Automatically add p_bond orbitals to active space if occ < 1.98",
    )
    parser.add_argument(
        "--generate-fcidump",
        action="store_true",
        help="Generate FCIDUMP from rotated orbitals",
    )
    parser.add_argument(
        "--selection-file",
        default=None,
        help="Path to a user-edited selection.txt file. "
             "When used with --generate-fcidump, reads active indices and "
             "n-electrons from this file instead of auto-generated values. "
             "Implies --source outputs/orbitals/rotated unless --source is also given.",
    )
    parser.add_argument(
        "--no-cubes",
        action="store_true",
        help="Skip cube file generation (faster)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source subdirectory under outputs/ (default: 'orbitals'). "
             "E.g. 'uno/orbitals' for examples/fe2s2/outputs/uno/orbitals/",
    )

    args = parser.parse_args()

    # Validate: need at least one of --rotate-group or --selection-file
    if not args.rotate_group and not args.selection_file:
        parser.error("At least one of --rotate-group or --selection-file is required.")
    if args.selection_file and not args.generate_fcidump:
        parser.error("--selection-file requires --generate-fcidump.")

    fcidump_only = (args.selection_file is not None and not args.rotate_group)

    case_dir = os.path.abspath(args.case_dir)
    print(f"=== orbital_rotater.py ===")
    print(f"Case dir: {case_dir}")
    if fcidump_only:
        print(f"Mode: FCIDUMP-only (from selection file)")
    else:
        print(f"Mode: Rotation + output generation")

    # --- Step 1: Load data ---
    print("\n--- Step 1: Loading data ---")

    # Determine source directory for orbitals
    if args.source:
        orbitals_source_dir = os.path.join(case_dir, "outputs", args.source)
    elif fcidump_only:
        orbitals_source_dir = os.path.join(case_dir, "outputs", "orbitals", "rotated")
    else:
        orbitals_source_dir = os.path.join(case_dir, "outputs", "orbitals")

    # Find h5 in the source directory
    h5_candidates = [
        os.path.join(orbitals_source_dir, f)
        for f in os.listdir(orbitals_source_dir)
        if f.endswith("_cas_data.h5") and os.path.getsize(os.path.join(orbitals_source_dir, f)) > 0
    ]
    if not h5_candidates:
        print(f"ERROR: No *_cas_data.h5 found in {orbitals_source_dir}")
        sys.exit(1)

    h5_path = sorted(h5_candidates, key=lambda p: os.path.getsize(p), reverse=True)[0]
    stem = os.path.basename(h5_path).replace("_cas_data.h5", "")

    # Find chkfile
    scf_dir = os.path.join(case_dir, "outputs", "scf")
    chk_name = f"{stem}.chk"
    chk_path = os.path.join(scf_dir, chk_name)
    if not os.path.isfile(chk_path):
        chk_candidates = [
            os.path.join(scf_dir, f)
            for f in os.listdir(scf_dir)
            if f.endswith(".chk") and os.path.getsize(os.path.join(scf_dir, f)) > 0
        ]
        if not chk_candidates:
            print(f"ERROR: No .chk file found in {scf_dir}")
            sys.exit(1)
        chk_path = sorted(chk_candidates, key=lambda p: os.path.getsize(p), reverse=True)[0]

    print(f"  H5:   {h5_path}")
    print(f"  CHK:  {chk_path}")
    print(f"  Stem: {stem}")

    # Load h5
    data = load_h5_data(h5_path)
    mo_coeff_full = data["mo_coeff_full"]
    occupations_full = data["occupations_full"]
    orbital_labels_full = data["orbital_labels_full"]
    active_indices = data.get("active_indices")
    metadata = data.get("metadata", {})

    print(f"  MO matrix shape: {mo_coeff_full.shape}")
    print(f"  N orbitals: {len(occupations_full)}")
    print(f"  Active indices: {active_indices}")

    # Rebuild mf from chkfile
    print("  Rebuilding mf from chkfile...")
    mol, mf = rebuild_mf_from_chk(chk_path, metadata)
    print(f"  mol: {mol.natm} atoms, basis={mol.basis}")

    # Find original report
    report_path = os.path.join(orbitals_source_dir, f"{stem}_orbital_report.md")
    if not fcidump_only and not os.path.isfile(report_path):
        print(f"WARNING: No orbital_report.md found at {report_path}")
        print("  Cannot identify terminal S atoms without the report.")
        sys.exit(1)

    # ==================================================================
    # FCIDUMP-only mode: load selection file, generate FCIDUMP, exit
    # ==================================================================
    if fcidump_only:
        print(f"\n--- Parsing selection file ---")
        sel_path = os.path.abspath(args.selection_file)
        print(f"  Selection file: {sel_path}")
        selected_indices, n_electrons = parse_selection_file(sel_path)
        print(f"  n-electrons: {n_electrons}")
        print(f"  n-orbitals:  {len(selected_indices)}")
        print(f"  Indices: {selected_indices}")

        # Validate indices
        nmo = mo_coeff_full.shape[1]
        for idx in selected_indices:
            if idx < 0 or idx >= nmo:
                print(f"ERROR: Index {idx} out of range [0, {nmo})")
                sys.exit(1)

        target_spin = float(metadata.get("target_spin", 0.0))
        output_dir = os.path.dirname(h5_path)  # same dir as the h5

        print(f"\n--- Generating FCIDUMP ---")
        generate_fcidump(
            mol, mf, mo_coeff_full, np.array(selected_indices, dtype=int),
            n_electrons, target_spin, output_dir, stem,
        )
        print(f"\n=== Done! FCIDUMP in: {output_dir}/fcidump/ ===")
        return

    # ==================================================================
    # Rotation mode: Steps 2-8 (existing flow)
    # ==================================================================

    # --- Step 2: Parse user-supplied rotation groups ---
    print("\n--- Step 2: Parsing user-supplied rotation groups ---")

    terminal_s_atoms = parse_atom_roles(report_path)
    metal_indices = parse_metal_indices(report_path)

    # Build lookup: atom_name -> terminal info
    terminal_lookup = {ts["atom"]: ts for ts in terminal_s_atoms}

    rotation_groups = []
    for group_args in args.rotate_group:
        atom_name = group_args[0]
        try:
            p_indices = [int(x) for x in group_args[1:4]]
        except ValueError:
            print(f"ERROR: --rotate-group indices must be integers: {group_args}")
            sys.exit(1)

        if len(p_indices) != 3:
            print(f"ERROR: Each --rotate-group must have exactly 3 indices, got: {group_args}")
            sys.exit(1)

        # Validate indices are in range
        nmo = mo_coeff_full.shape[1]
        for idx in p_indices:
            if idx < 0 or idx >= nmo:
                print(f"ERROR: Index {idx} out of range [0, {nmo})")
                sys.exit(1)

        # Look up atom info from terminal_s_atoms
        if atom_name in terminal_lookup:
            ts = terminal_lookup[atom_name]
            atom_index = ts["index"]
            fe_label = ts["bonded_to"]
        else:
            # Atom not in Atom Roles terminal list — try to find it in all atoms
            print(f"  WARNING: {atom_name} not found in terminal atoms of Atom Roles table.")
            print(f"           Will try to resolve from molecule coordinates.")
            # Parse atom index from name: "S3" -> element "S", index 2 (0-based)
            m = re.match(r'([A-Z][a-z]?)(\d+)', atom_name)
            if m:
                atom_index = int(m.group(2)) - 1
            else:
                print(f"ERROR: Cannot parse atom name: {atom_name}")
                sys.exit(1)
            fe_label = None

        if fe_label and fe_label in metal_indices:
            fe_index = metal_indices[fe_label]
        else:
            print(f"ERROR: Cannot find bonded metal for {atom_name} (fe_label={fe_label})")
            sys.exit(1)

        # Print info for this group
        occs = [occupations_full[idx] for idx in p_indices]
        labels = [orbital_labels_full[idx] for idx in p_indices]
        print(f"  {atom_name} (atom_idx={atom_index}) -> bonded to {fe_label} (idx={fe_index})")
        print(f"    Orbital indices: {p_indices}")
        for j, idx in enumerate(p_indices):
            print(f"      idx={idx}: label='{labels[j]}', occ={occs[j]:.6f}")

        rotation_groups.append({
            "atom": atom_name,
            "atom_index": atom_index,
            "fe_label": fe_label,
            "fe_index": fe_index,
            "p_indices": p_indices,
        })

    if not rotation_groups:
        print("  No rotation groups specified. Nothing to do.")
        sys.exit(0)

    # --- Step 3 & 4: Build rotation and apply ---
    print("\n--- Step 3 & 4: Building rotation matrices and applying ---")
    mo_coeff_new, rotation_info = apply_rotation(mo_coeff_full, rotation_groups, mol)

    # --- Step 5: Recompute occupation ---
    print("\n--- Step 5: Recomputing occupations ---")
    occ_new = recompute_occupations(mo_coeff_new, mf)

    # Print comparison for rotated orbitals
    print("  Occupation changes for rotated MOs:")
    for ri in rotation_info:
        p_indices = ri["p_indices"]
        suffixes = ["_bond", "_perp1", "_perp2"]
        for i, idx in enumerate(p_indices):
            old_occ = occupations_full[idx]
            new_occ = occ_new[idx]
            print(f"    idx={idx} {ri['atom']}{suffixes[i]}: "
                  f"occ {old_occ:.6f} -> {new_occ:.6f} (delta={new_occ - old_occ:+.6f})")

    # --- Step 6: Update labels ---
    print("\n--- Step 6: Updating labels ---")
    labels_new = update_labels(orbital_labels_full, rotation_info)

    # --- Step 7: --select-bond-dir ---
    active_indices_new = np.array(active_indices, dtype=int) if active_indices is not None else np.array([], dtype=int)
    n_electrons = int(metadata.get("n_electrons", 0))

    if args.select_bond_dir:
        print("\n--- Step 7: --select-bond-dir ---")
        active_indices_new, ne_delta, messages = select_bond_dir(
            active_indices_new, occ_new, rotation_info, labels_new
        )
        for msg in messages:
            print(msg)
        if ne_delta != 0:
            n_electrons += ne_delta
            print(f"  Updated n_electrons: {n_electrons - ne_delta} -> {n_electrons}")
        active_indices_new = np.array(active_indices_new, dtype=int)

    # --- Step 8: Generate output files ---
    output_dir = os.path.join(case_dir, "outputs", "orbitals", "rotated")
    print(f"\n--- Step 8: Generating output files ---")
    print(f"  Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save h5
    save_rotated_h5(
        output_dir, stem, data, mo_coeff_new, occ_new, labels_new,
        active_indices_new, n_electrons,
    )

    # Selection.txt
    generate_selection_file(output_dir, stem, active_indices_new, n_electrons)

    # NOON plot
    noon_plot_path = os.path.join(output_dir, f"{stem}_noon_plot.png")
    generate_noon_plot(occ_new, labels_new, noon_plot_path)

    # Orbital report
    generate_orbital_report(
        output_dir, stem, mol, mo_coeff_new, occ_new, labels_new,
        active_indices_new, report_path, rotation_info, metadata,
    )

    # Copy SCF info
    scf_info_src = os.path.join(scf_dir, f"{stem}_scf_info.json")
    scf_info_dst = os.path.join(output_dir, f"{stem}_scf_info.json")
    if os.path.isfile(scf_info_src):
        shutil.copy2(scf_info_src, scf_info_dst)
        print(f"  Copied: {scf_info_dst}")

    # Cube files and gallery
    if not args.no_cubes:
        print("\n  Generating cube files...")
        cube_paths = generate_cube_files(
            mol, mo_coeff_new, output_dir, stem, labels_new,
            active_indices_new, rotation_info,
        )

        # Collect indices for gallery
        gallery_indices = sorted(set(list(active_indices_new) +
                                     [idx for ri in rotation_info
                                      for idx in ri["p_indices"]]))

        print("  Generating orbital gallery...")
        generate_orbital_gallery(
            cube_paths, labels_new, occ_new, gallery_indices,
            output_dir, stem,
        )
    else:
        print("  Skipping cube file generation (--no-cubes)")

    # --- Step 9: --generate-fcidump ---
    if args.generate_fcidump:
        print(f"\n--- Step 9: Generating FCIDUMP ---")
        target_spin = float(metadata.get("target_spin", 0.0))
        generate_fcidump(
            mol, mf, mo_coeff_new, active_indices_new, n_electrons,
            target_spin, output_dir, stem,
        )

    print(f"\n=== Done! Output in: {output_dir} ===")


if __name__ == "__main__":
    main()
