"""APEX_CAS package root.

This module intentionally exposes only the minimal user-facing model and helper
surface needed by the staged `apex-cas` workflow. Construction, persistence,
visualization, and FCIDUMP internals live in their dedicated modules.
"""

__version__ = "0.1.0"

# ── Re-export a minimal user-facing model surface ──
from shared.models import (
    MetalCenter,
    BridgingAtom,
    TerminalLigand,
    ClusterInfo,
    CAS,
    ComputationSettings,
)

# ── Re-export minimal user-facing helpers ──
from shared.structure_parser import parse_structure
from shared.setting_utils import (
    load_cas_settings_file,
)

__all__ = [
    "MetalCenter",
    "BridgingAtom",
    "TerminalLigand",
    "ClusterInfo",
    "CAS",
    "ComputationSettings",
    "parse_structure",
    "load_cas_settings_file",
]
