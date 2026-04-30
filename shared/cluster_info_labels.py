"""Helpers for user-facing labels sourced from ClusterInfo annotations."""

from __future__ import annotations


def is_authoritative_cluster_info(cluster_info) -> bool:
    """Return True when ClusterInfo comes from finalized cluster_info.yaml."""
    return getattr(cluster_info, "annotation_source", "") == "cluster_info_yaml"


def resolve_explicit_label(
    label: str | None,
    fallback: str,
    *,
    cluster_info=None,
    context: str = "cluster_info object",
) -> str:
    """Return an explicit label or a legacy fallback.

    In strict authority mode, user-visible labels must come from the finalized
    ``cluster_info.yaml`` annotations. Missing labels therefore raise instead
    of silently manufacturing a replacement.
    """
    if label:
        return label
    if is_authoritative_cluster_info(cluster_info):
        raise ValueError(
            f"Missing explicit label for {context} while using authoritative "
            "cluster_info.yaml annotations."
        )
    return fallback


def resolve_metal_site_label(cluster_info, site_idx: int) -> str:
    """Return the user-facing label for a metal site."""
    metal = cluster_info.metals[site_idx]
    return resolve_explicit_label(
        getattr(metal, "label", ""),
        f"{metal.element}{site_idx + 1}",
        cluster_info=cluster_info,
        context=f"metal site {site_idx}",
    )
