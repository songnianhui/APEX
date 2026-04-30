"""Shared element and shell metadata for APEX_CAS/APEX_Filter.

This module centralizes:
- atomic numbers
- covalent radii
- default bridging-element labels
- transition-metal / d-series classification
- simple valence-shell heuristics

It is intentionally lightweight and free of PySCF dependencies so that both
structure parsing and active-space construction can reuse the same element
knowledge.
"""

from __future__ import annotations

# Atomic numbers
ELEMENTS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

L_CHAR_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}

METALS_3D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
METALS_4D = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
METALS_5D = {"La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
LANTHANIDES = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ACTINIDES = {"Th", "Pa", "U", "Np", "Pu"}

TRANSITION_METALS = METALS_3D | METALS_4D | METALS_5D | LANTHANIDES | ACTINIDES

BRIDGING_ELEMENTS = {"S", "O", "N", "Se", "Cl", "P", "F", "Br", "I", "H"}

# Covalent radii in Angstroms (Cordero et al. 2008)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
    "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 1.87, "Hf": 1.75, "Ta": 1.70,
    "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32,
    "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92,
    "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87,
}

_AUFBAU = [
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
    (4, "s", 2),
    (3, "d", 10),
    (4, "p", 6),
    (5, "s", 2),
    (4, "d", 10),
    (5, "p", 6),
    (6, "s", 2),
    (4, "f", 14),
    (5, "d", 10),
    (6, "p", 6),
    (7, "s", 2),
    (5, "f", 14),
    (6, "d", 10),
    (7, "p", 6),
]

_NOBLE_GAS = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}


def get_atomic_number(element: str) -> int:
    return ELEMENTS.get(element, 0)


def get_period(Z: int) -> int:
    for zmax, p in [(2, 1), (10, 2), (18, 3), (36, 4), (54, 5), (86, 6), (118, 7)]:
        if Z <= zmax:
            return p
    return 7


def build_electron_config(Z: int):
    remaining = Z
    config = []
    for n, lc, cap in _AUFBAU:
        if remaining <= 0:
            break
        e = min(remaining, cap)
        if e > 0:
            config.append((n, lc, e))
        remaining -= e
    return config


def get_electron_config(Z: int):
    """Return occupied shells plus string renderings for atomic config."""
    config = build_electron_config(Z)
    occupied = {(n, lc) for n, lc, _ in config}

    sup = str.maketrans(
        "0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    )
    cfg = " ".join(f"{n}{lc}{str(e).translate(sup)}" for n, lc, e in config)

    core_Z = max((z for z in _NOBLE_GAS if z < Z), default=0)
    ng = ""
    if core_Z:
        core_occ = {(n, lc) for n, lc, _ in build_electron_config(core_Z)}
        outer = [(n, lc, e) for n, lc, e in config if (n, lc) not in core_occ]
        ng = (
            "["
            + _NOBLE_GAS[core_Z]
            + "] "
            + " ".join(f"{n}{lc}{str(e).translate(sup)}" for n, lc, e in outer)
        )
    return occupied, cfg, ng


def get_valence_shells(Z: int):
    """Period-based heuristic: ns, np of period; (n-1)d; (n-2)f."""
    occupied, _, _ = get_electron_config(Z)
    p = get_period(Z)
    v = set()
    if (p, "s") in occupied:
        v.add((p, "s"))
    if (p, "p") in occupied:
        v.add((p, "p"))
    if p >= 4 and (p - 1, "d") in occupied:
        v.add((p - 1, "d"))
    if p >= 6 and (p - 2, "f") in occupied:
        v.add((p - 2, "f"))
    return v


def is_transition_metal(element: str) -> bool:
    return element in TRANSITION_METALS


def is_bridging_element(element: str) -> bool:
    return element in BRIDGING_ELEMENTS


def get_covalent_radius(element: str, default: float = 1.0) -> float:
    return COVALENT_RADII.get(element, default)


def get_3d_metals():
    return set(METALS_3D)


def get_4d_metals():
    return set(METALS_4D)


def get_5d_metals():
    return set(METALS_5D)


def get_metal_row(element: str) -> str:
    if element in METALS_3D:
        return "3d"
    if element in METALS_4D:
        return "4d"
    if element in METALS_5D:
        return "5d"
    if element in LANTHANIDES:
        return "4f"
    if element in ACTINIDES:
        return "5f"
    return ""


def get_default_d_label(element: str, fallback: str = "4d") -> str:
    row = get_metal_row(element)
    if row in {"3d", "4d", "5d"}:
        return row
    return fallback
