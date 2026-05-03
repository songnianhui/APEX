"""AO-shell classification helpers for buildcas reporting.

This module is an internal helper behind
``orbital_visualizer.plot_orbitals(...)``. It groups PySCF AO labels into
shells, classifies them into core/valence/polarization/diffuse buckets, and
renders the markdown fragments embedded in ``orbital_report.md``.
"""

import re
from collections import OrderedDict as _OrderedDict, defaultdict as _defaultdict
from shared.element_data import (
    ELEMENTS as _ELEMENTS,
    L_CHAR_TO_INT,
    get_electron_config as _get_electron_config,
    get_valence_shells as _get_valence_shells,
)


# ================================================================
#  AO-Label Parsing
# ================================================================

_LABEL_RE = re.compile(r"^(\d+)\s+(\S+)\s+(\d+)([spdfghij])(\S*)")


def _parse(label):
    m = _LABEL_RE.match(label.strip())
    return (int(m[1]), m[2], int(m[3]), m[4], m[5]) if m else None


def _group(labels):
    g = _OrderedDict()
    for lbl in labels:
        p = _parse(lbl)
        if p is None:
            continue
        key = (p[0], p[1], p[2], p[3])
        g.setdefault(key, [])
        if p[4]:
            g[key].append(p[4])
    return g


# ================================================================
#  Classification
# ================================================================

_ICONS = {
    "Core": "\u25cf",
    "Valence": "\u25c9",
    "Diffuse": "\u25cb",
    "Polarization": "\u25c7",
}


def _classify(n, lc, occupied, valence):
    if (n, lc) in occupied:
        return ("Valence", True) if (n, lc) in valence else ("Core", True)
    same_l = any(c == lc for _, c in occupied)
    return ("Diffuse", False) if same_l else ("Polarization", False)


# ================================================================
#  Main Analysis
# ================================================================


def _analyze_ao_shells(mol):
    """
    Print a detailed table of AO-shell vs physical-shell correspondence.

    Parameters
    ----------
    mol : pyscf.gto.Mole   (must already be built)
    """
    labels = mol.ao_labels(fmt=True)
    groups = _group(labels)

    by_atom = _defaultdict(list)
    for (aidx, elem, n, lc), comps in groups.items():
        by_atom[aidx].append((elem, n, lc, comps))

    # --- header ---
    bname = getattr(mol, "basis", "unknown")
    if isinstance(bname, dict):
        bname = ", ".join(f"{k}:{v}" for k, v in bname.items())
    elif not isinstance(bname, str):
        bname = str(bname)

    sep = "=" * 80
    thin = "\u2500" * 80
    print(f"\n{sep}")
    print("  AO Shell  vs  Physical Shell  Correspondence  Analysis")
    print(f"  Basis      : {bname}")
    print(f"  Total AOs  : {len(labels)}")
    print(f"  Atoms      : {mol.natm}")
    print(sep)

    # --- per-atom ---
    for aidx in sorted(by_atom):
        shells = by_atom[aidx]
        elem = shells[0][0]
        Z = _ELEMENTS.get(elem, 0)
        if Z == 0:
            print(f"\n  Atom {aidx}: {elem} -- element not in table, skipped.")
            continue

        occ, cfg, ng = _get_electron_config(Z)
        val = _get_valence_shells(Z)

        occ_s = ", ".join(
            f"{principal_n}{shell}"
            for principal_n, shell in sorted(
                occ, key=lambda x: (x[0], L_CHAR_TO_INT.get(x[1], 99))
            )
        )
        val_s = ", ".join(
            f"{principal_n}{shell}" for principal_n, shell in sorted(val)
        )

        print(f"\n{thin}")
        print(f"  Atom {aidx} : {elem}  (Z = {Z})")
        print(f"  Ground-state config : {cfg}")
        if ng:
            print(f"  Noble-gas notation  : {ng}")
        print(f"  Physical shells     : {occ_s}")
        print(f"  Valence shells      : {val_s}")
        print(thin)
        print(
            f"  {'Basis Shell':<12} {'#AO':>4}  "
            f"{'Components':<42} {'Category':<14} {'Physical?':<10}"
        )
        print(f"  {'':-<12} {'':->4}  {'':-<42} {'':-<14} {'':-<10}")

        shells_sorted = sorted(
            shells, key=lambda x: (L_CHAR_TO_INT.get(x[2], 99), x[1])
        )

        ctr = _defaultdict(lambda: [0, 0])  # cat -> [n_shells, n_AOs]

        for _, n, lc, comps in shells_sorted:
            cat, phys = _classify(n, lc, occ, val)
            n_ao = len(comps) if comps else 1
            c_str = ", ".join(comps) if comps else "\u2014"
            p_str = f"{n}{lc}" if phys else "\u2014"
            icon = _ICONS.get(cat, "?")
            print(f"  {icon} {n}{lc:<11} {n_ao:>4}  {c_str:<42} {cat:<14} {p_str:<10}")
            ctr[cat][0] += 1
            ctr[cat][1] += n_ao

        print(f"  {'':-<12} {'':->4}  {'':-<42} {'':-<14} {'':-<10}")

        # summary
        print(f"\n  Summary for {elem} (atom {aidx}):")
        print(f"    {'Category':<16} {'Shells':>7}  {'AOs':>5}")
        print(f"    {'-' * 16} {'-' * 7}  {'-' * 5}")
        for cat in ("Core", "Valence", "Diffuse", "Polarization"):
            ns, na = ctr[cat]
            if ns:
                print(f"    {cat:<16} {ns:>7}  {na:>5}")
        ts = sum(v[0] for v in ctr.values())
        ta = sum(v[1] for v in ctr.values())
        print(f"    {'-' * 16} {'-' * 7}  {'-' * 5}")
        print(f"    {'Total':<16} {ts:>7}  {ta:>5}")

    print(f"\n{sep}\n")
    return ctr


def _generate_ao_shell_markdown(mol, cluster_info=None):
    """Generate a Markdown-formatted AO Shell Analysis section for the orbital report.

    Deduplicates atoms by element symbol, showing one analysis table per
    unique element.  Proj_weight target shells are rendered in **bold** with
    a dagger (†) marker; other valence shells are shown in normal weight.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Built molecule object.
    cluster_info : ClusterInfo, optional
        When provided, role-filtered valence shells of metals (d/f only)
        and bridging atoms (p only) are annotated as the proj_weight target
        subspace, matching ``_build_target_ao_subspace()`` behaviour.

    Returns
    -------
    str
        Markdown string (may be empty if no valid elements found).
    """
    labels_list = mol.ao_labels(fmt=True)
    groups = _group(labels_list)

    # Per-atom shell lists
    by_atom: dict[int, list] = _defaultdict(list)
    atom_elem: dict[int, str] = {}
    for (aidx, elem, n, lc), comps in groups.items():
        by_atom[aidx].append((n, lc, comps))
        atom_elem[aidx] = elem

    if not by_atom:
        return ""

    # Determine atom-level role categories for projection annotation
    atom_role_key: dict[int, str] = {}
    if cluster_info is not None:
        for m in cluster_info.metals:
            atom_role_key[m.index] = "metal"
        for b in cluster_info.bridging_atoms:
            atom_role_key[b.index] = "bridging"

    def _get_proj_weight_target_shells(Z, role):
        """Return valence shells filtered by atom role for proj_weight target."""
        val = _get_valence_shells(Z)
        if role == "metal":
            return {(n, lc) for n, lc in val if L_CHAR_TO_INT.get(lc, 0) >= 2}
        else:  # bridging
            return {(n, lc) for n, lc in val if L_CHAR_TO_INT.get(lc, 0) == 1}

    # Deduplicate by (element, role category) so mixed-role elements such as
    # bridging/terminal sulfur are not collapsed into a single misleading row.
    seen: set[tuple[str, str]] = set()
    unique_entries: list[tuple[int, str]] = []
    for aidx in sorted(by_atom):
        elem = atom_elem.get(aidx, "")
        role_key = atom_role_key.get(aidx, "other")
        key = (elem, role_key)
        if elem and key not in seen:
            seen.add(key)
            unique_entries.append((aidx, role_key))

    # Basis name
    bname = getattr(mol, "basis", "unknown")
    if isinstance(bname, dict):
        bname = ", ".join(f"{k}:{v}" for k, v in bname.items())
    elif not isinstance(bname, str):
        bname = str(bname)

    lines: list[str] = []
    lines.append("## AO Shell Analysis")
    lines.append("")
    lines.append(f"*Basis: {bname}*")
    lines.append("")

    for aidx, role_key in unique_entries:
        elem = atom_elem.get(aidx, "")
        Z = _ELEMENTS.get(elem, 0)
        if Z == 0:
            continue

        occ, cfg, ng = _get_electron_config(Z)
        val = _get_valence_shells(Z)

        val_s = ", ".join(
            f"{principal_n}{shell}" for principal_n, shell in sorted(val)
        )
        ng_display = ng if ng else cfg

        # Determine proj_weight target shells based on atom role
        if role_key == "metal":
            target_shells = _get_proj_weight_target_shells(Z, "metal")
            target_s = ", ".join(
                f"{principal_n}{shell}"
                for principal_n, shell in sorted(target_shells)
            )
            target_note = f" — proj_weight target: {target_s}"
        elif role_key == "bridging":
            target_shells = _get_proj_weight_target_shells(Z, "bridging")
            target_s = ", ".join(
                f"{principal_n}{shell}"
                for principal_n, shell in sorted(target_shells)
            )
            target_note = f" — proj_weight target: {target_s}"
        else:
            target_shells = set()
            target_note = ""

        role_note = "" if role_key == "other" else f" [{role_key}]"
        lines.append(
            f"### {elem}{role_note} (Z={Z}) — {ng_display} — Valence: {val_s}{target_note}"
        )
        lines.append("")
        lines.append("| Shell | #AO | Category | Physical |")
        lines.append("|-------|-----|----------|----------|")

        shells = by_atom[aidx]
        shells_sorted = sorted(
            shells, key=lambda x: (L_CHAR_TO_INT.get(x[1], 99), x[0])
        )

        for n, lc, comps in shells_sorted:
            cat, phys = _classify(n, lc, occ, val)
            n_ao = len(comps) if comps else 1
            p_str = f"{n}{lc}" if phys else "\u2014"

            if (n, lc) in target_shells:
                # Proj_weight target shell: bold + dagger
                lines.append(
                    f"| **{n}{lc}\u2020** | **{n_ao}** | **{cat}** | **{p_str}** |"
                )
            elif (n, lc) in val:
                # Other valence shell: normal weight, show role in Category
                lines.append(
                    f"| {n}{lc} | {n_ao} | {cat} ({lc}) | {p_str} |"
                )
            else:
                lines.append(f"| {n}{lc} | {n_ao} | {cat} | {p_str} |")

        lines.append("")

    lines.append(
        "*Shells marked with \u2020 are the proj_weight target subspace "
        "(metal d/f + bridging p, per Chan\u2019s convention). "
        "Other valence shells (e.g. metal s, bridging s) are excluded "
        "from the projection.*"
    )
    lines.append("")

    return "\n".join(lines)
