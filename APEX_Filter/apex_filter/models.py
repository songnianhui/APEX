"""APEX_Filter — Data models (re-exported from shared)."""

import os
import sys

# Ensure the APEX root directory (parent of APEX_Filter/) is on sys.path
# so that `shared.models` is importable.
_APEX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _APEX_ROOT not in sys.path:
    sys.path.insert(0, _APEX_ROOT)

from shared.models import (           # noqa: E402
    MetalCenter,
    BridgingAtom,
    TerminalLigand,
    ClusterInfo,
    ActiveSpaceLevel,
    OrbitalGroup,
    CAS,
    SpinIsomer,
    SpinIsomerFamily,
    OxidationAssignment,
    ElectronicConfig,
    FilteringLevel,
    FilteringPlan,
    CalculationResult,
    ExtrapolatedEnergy,
    ActiveSpaceQuality,
    ComputationSettings,
    NonComputingMethod,
    NonComputingMethodConfig,
    AVASConfig,
)
