"""Metadata helpers for orbital-method outputs."""

from __future__ import annotations


DMRG_BASIS_SOURCE_METHOD = "UCCSD-NO/split-localized/paired/GA-ordered"


def build_source_method_prefix(settings) -> str:
    """Build the source-method prefix used in metadata/reporting."""
    scf_method = settings.scf_method.upper()
    xc = settings.xc_functional
    return f"UKS-{xc}" if scf_method == "UKS" else "UHF"
