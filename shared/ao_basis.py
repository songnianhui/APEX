"""Small AO-basis helpers shared across APEX packages."""

from __future__ import annotations


def get_d_ao_indices(mol, atom_idx):
    """Return d-type AO indices for a given atom in a PySCF Mole."""
    aoslices = mol.aoslice_by_atom()
    if atom_idx >= len(aoslices):
        return []

    _, _, ao_s, ao_e = aoslices[atom_idx]
    ao_labels = mol.ao_labels()

    d_indices = []
    for i in range(ao_s, min(ao_e, len(ao_labels))):
        label = ao_labels[i]
        parts = label.split()
        if len(parts) >= 3 and "d" in parts[-1].lower():
            d_indices.append(i)

    return d_indices
