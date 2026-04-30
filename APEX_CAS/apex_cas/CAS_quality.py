"""Active space quality assessment.

Validates active spaces using NOON analysis, entropy analysis,
and chemical intuition checks.
"""

import numpy as np

from . import (
    CAS,
    ActiveSpaceQuality,
)



def validate_noon(
    active_orbitals: CAS,
    expected_types: list[dict] | None = None,
    noon_lo: float = 0.02,
    noon_hi: float = 1.98,
) -> ActiveSpaceQuality:
    """Validate active space quality using Natural Orbital Occupation Numbers (NOON).

    Parameters
    ----------
    active_orbitals : CAS
        CAS object containing ``occupations`` (NOON array) and
        ``orbital_labels``.
    expected_types : list[dict] or None
        Optional list of dicts describing expected orbital types (from Stage 1).
        Each dict may contain keys like ``"atom_label"``, ``"ao_type"``, etc.
    noon_lo : float
        Threshold below which orbitals are considered empty (default 0.02).
    noon_hi : float
        Threshold above which orbitals are considered doubly occupied (default 1.98).

    Returns
    -------
    ActiveSpaceQuality
        Populated quality assessment object.
    """
    noon = np.asarray(active_orbitals.occupations, dtype=float)
    labels = active_orbitals.orbital_labels

    # ── Count doubly occupied and empty orbitals ──────────────────
    n_doubly = int(np.sum(noon > noon_hi))
    n_empty = int(np.sum(noon < noon_lo))

    # ── Build warning messages ────────────────────────────────────
    noon_warnings: list[str] = []
    warnings: list[str] = []

    if n_doubly > 0:
        msg = (
            f"{n_doubly} orbital(s) with n > {noon_hi} "
            f"— likely doubly occupied, consider removing"
        )
        noon_warnings.append(msg)
        warnings.append(msg)

    if n_empty > 0:
        msg = (
            f"{n_empty} orbital(s) with n < {noon_lo} "
            f"— likely empty, consider removing"
        )
        noon_warnings.append(msg)
        warnings.append(msg)

    # Check for narrow spread (all near 1.0)
    if len(noon) > 0 and np.all((noon >= 0.95) & (noon <= 1.05)):
        msg = (
            "All orbitals near n=1.0 — may indicate insufficient active space"
        )
        noon_warnings.append(msg)
        warnings.append(msg)

    # ── Expected types coverage ───────────────────────────────────
    missing_orbital_types: list[str] = []
    orbital_character_map: dict = {}

    # Build character map from orbital labels
    if labels:
        for idx, label in enumerate(labels):
            orbital_character_map[idx] = label

    if expected_types is not None:
        for exp in expected_types:
            # Construct a search key from expected type info
            atom_label = exp.get("atom_label", "")
            ao_type = exp.get("ao_type", "")
            element = exp.get("element", "")
            search_key = f"{atom_label}_{ao_type}" if atom_label and ao_type else ""

            found = False
            if labels and search_key:
                for label in labels:
                    # Match if the label contains the expected atom + AO type
                    if search_key.lower() in label.lower():
                        found = True
                        break
            # Fallback: match on element + ao_type if no atom_label
            if not found and labels and element and ao_type:
                for label in labels:
                    if ao_type.lower() in label.lower() and element.lower() in label.lower():
                        found = True
                        break

            if not found:
                # Build a human-readable description of the missing type
                parts = [p for p in (atom_label, element, ao_type) if p]
                missing_orbital_types.append(" ".join(parts))

        # Warnings for missing types
        if missing_orbital_types:
            for mt in missing_orbital_types:
                warnings.append(f"Missing expected orbital type: {mt}")

    # ── Compute quality score ─────────────────────────────────────
    quality_score = 1.0

    # Penalty for doubly occupied orbitals
    quality_score -= 0.1 * n_doubly

    # Penalty for empty orbitals
    quality_score -= 0.05 * n_empty

    # Penalty for missing expected types
    if expected_types is not None:
        quality_score -= 0.1 * len(missing_orbital_types)

    # Clamp to [0, 1]
    quality_score = float(max(0.0, min(1.0, quality_score)))

    return ActiveSpaceQuality(
        noon_values=noon,
        noon_warning=noon_warnings,
        n_doubly_occupied=n_doubly,
        n_empty=n_empty,
        quality_score=quality_score,
        orbital_character_map=orbital_character_map,
        missing_orbital_types=missing_orbital_types,
        warnings=warnings,
    )


def analyze_entropy(
    entropy_values,
    orbital_labels: list[str],
    threshold_critical: float = 0.5,
    threshold_removable: float = 0.1,
) -> list[dict]:
    """Analyze single-orbital entropy values (from DMRG calculations).

    Parameters
    ----------
    entropy_values : array-like
        Single-orbital entropy values.
    orbital_labels : list[str]
        Orbital label strings corresponding to each entropy value.
    threshold_critical : float
        Entropy at or above this threshold marks a critical orbital (default 0.5).
    threshold_removable : float
        Entropy below this threshold marks a potentially removable orbital
        (default 0.1).

    Returns
    -------
    list[dict]
        Each dict contains ``orbital_idx``, ``label``, ``entropy``,
        ``classification``, and ``recommendation``.
    """
    entropy = np.asarray(entropy_values, dtype=float)
    results: list[dict] = []

    for idx in range(len(entropy)):
        s = float(entropy[idx])
        label = orbital_labels[idx] if idx < len(orbital_labels) else f"orb_{idx}"

        if s >= threshold_critical:
            classification = "critical"
            recommendation = "Must keep in active space"
        elif s >= threshold_removable:
            classification = "active"
            recommendation = "Important for correlation"
        else:
            classification = "removable"
            recommendation = "Consider removing to reduce cost"

        results.append({
            "orbital_idx": idx,
            "label": label,
            "entropy": s,
            "classification": classification,
            "recommendation": recommendation,
        })

    return results


def compute_quality_score(
    noon_values,
    orbital_character_map: dict,
    expected_types: list[dict],
    entropy_values=None,
) -> float:
    """Compute a composite quality score for the active space.

    Parameters
    ----------
    noon_values : array-like
        NOON values for the active orbitals.
    orbital_character_map : dict
        Mapping of ``{orb_idx: "Fe1_3d"}`` chemical labels.
    expected_types : list[dict]
        Expected orbital type dicts from Stage 1 analysis.
    entropy_values : array-like or None
        Optional single-orbital entropy values.

    Returns
    -------
    float
        Quality score between 0 and 1.
    """
    noon = np.asarray(noon_values, dtype=float)

    # ── 1. NOON quality (weight 0.4) ─────────────────────────────
    if len(noon) > 0:
        good_noon = np.sum((noon > 0.02) & (noon < 1.98))
        noon_quality = float(good_noon) / float(len(noon))
    else:
        noon_quality = 0.0

    # ── 2. Chemical coverage quality (weight 0.4) ────────────────
    if expected_types:
        n_found = 0
        for exp in expected_types:
            atom_label = exp.get("atom_label", "")
            ao_type = exp.get("ao_type", "")
            element = exp.get("element", "")
            search_key = f"{atom_label}_{ao_type}" if atom_label and ao_type else ""

            found = False
            if search_key:
                for label in orbital_character_map.values():
                    if search_key.lower() in label.lower():
                        found = True
                        break
            if not found and element and ao_type:
                for label in orbital_character_map.values():
                    if ao_type.lower() in label.lower() and element.lower() in label.lower():
                        found = True
                        break
            if found:
                n_found += 1

        coverage_quality = float(n_found) / float(len(expected_types))
    else:
        # No expected types specified — neutral score
        coverage_quality = 1.0

    # ── 3. Entropy quality (weight 0.2) ──────────────────────────
    if entropy_values is not None and len(entropy_values) > 0:
        entropy = np.asarray(entropy_values, dtype=float)
        n_critical_or_active = int(np.sum(entropy >= 0.1))
        entropy_quality = float(n_critical_or_active) / float(len(entropy))
    else:
        entropy_quality = 1.0  # neutral when no entropy data available

    # ── 4. Weighted sum ──────────────────────────────────────────
    score = 0.4 * noon_quality + 0.4 * coverage_quality + 0.2 * entropy_quality
    return float(max(0.0, min(1.0, score)))


def print_quality_report(quality: ActiveSpaceQuality) -> str:
    """Generate a human-readable quality report string.

    Parameters
    ----------
    quality : ActiveSpaceQuality
        Populated quality assessment object.

    Returns
    -------
    str
        Multi-line report string.
    """
    lines: list[str] = []
    sep = "=" * 50

    lines.append(sep)
    lines.append("Active Space Quality Report")
    lines.append(sep)

    # ── NOON summary ──────────────────────────────────────────────
    if quality.noon_values is not None and len(quality.noon_values) > 0:
        noon = quality.noon_values
        lines.append("")
        lines.append("NOON Summary:")
        lines.append(f"  Min occupation : {float(np.min(noon)):.4f}")
        lines.append(f"  Max occupation : {float(np.max(noon)):.4f}")
        lines.append(f"  Mean occupation: {float(np.mean(noon)):.4f}")
    else:
        lines.append("")
        lines.append("NOON Summary: (no NOON data available)")

    # ── Doubly occupied ───────────────────────────────────────────
    lines.append("")
    if quality.n_doubly_occupied > 0:
        lines.append(
            f"  Doubly occupied orbitals: {quality.n_doubly_occupied}  [WARNING]"
        )
    else:
        lines.append("  Doubly occupied orbitals: 0")

    # ── Empty orbitals ────────────────────────────────────────────
    if quality.n_empty > 0:
        lines.append(
            f"  Empty orbitals:          {quality.n_empty}  [WARNING]"
        )
    else:
        lines.append("  Empty orbitals:          0")

    # ── Missing orbital types ─────────────────────────────────────
    lines.append("")
    if quality.missing_orbital_types:
        lines.append("Missing Orbital Types:")
        for mt in quality.missing_orbital_types:
            lines.append(f"  - {mt}")
    else:
        lines.append("Missing Orbital Types: (none)")

    # ── Quality score ─────────────────────────────────────────────
    lines.append("")
    score = quality.quality_score
    if score >= 0.8:
        rating = "GOOD"
    elif score >= 0.5:
        rating = "WARNING"
    else:
        rating = "POOR"
    lines.append(f"Quality Score: {score:.3f}  [{rating}]")

    # ── Individual warnings ───────────────────────────────────────
    if quality.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in quality.warnings:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)
