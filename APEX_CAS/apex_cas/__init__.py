"""APEX_CAS — Automated Chemical Active Space Construction for Transition Metal Clusters.

Provides tools for:
  - Parsing molecular structures into ClusterInfo
  - Building chemical active spaces (CAS) via non_computing methods (cas_builder_noncomputing.py)and computing methods (cas_builder_computing.py)
  - Quality validation of CAS using natural orbital occupation numbers (CAS_quality.py)
  - Orbital visualization and interactive active-space selection (orbital_visualizer.py)
  - FCIDUMP generation (FCIDUMP_generator.py)
"""

__version__ = "0.1.0"
__author__ = "Song@Elab"

# ── Re-export all models ──
from .models import (
    MetalCenter,
    BridgingAtom,
    TerminalLigand,
    ClusterInfo,
    ActiveSpaceLevel,
    OrbitalGroup,
    CAS,
    ActiveSpaceQuality,
    ComputationSettings,
    NonComputingMethod,
    NonComputingMethodConfig,
    AVASConfig,
)

# ── Re-export key public functions ──
from .structure_analyzer import parse_structure
from .CAS_builder_noncomputing import build_NC_CAS
from .CAS_builder_computing import build_computed_CAS, init_computing
from .CAS_quality import validate_noon, print_quality_report
from .computation_defaults import PRESETS, apply_overrides
from .orbital_visualizer import (
    generate_orbital_report,
    generate_orbital_cubes,
    generate_noon_plot,
    load_user_selection,
    plot_orbitals,
    save_cas_state,
    load_cas_state,
)
from .FCIDUMP_generator import (
    transform_active_integrals,
    write_fcidump,
    compare_fcidumps,
    generate_fcidump_from_selection,
)
