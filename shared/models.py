"""APEX — Automated Progressive Electronic structure eXploration

Toolkit for automated active space analysis, spin/electronic configuration
enumeration, and quantum chemistry input generation for transition metal clusters.
"""

__version__ = "0.1.0"

from dataclasses import dataclass as _dataclass, field as _field
from enum import Enum as _Enum
from typing import Optional as _Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Structure-level data classes
# ──────────────────────────────────────────────────────────────────

@_dataclass
class MetalCenter:
    """A transition metal site in the cluster."""
    element: str            # e.g., "Fe", "Mo"
    index: int              # atom index in the full structure
    position: np.ndarray    # 3D coordinates (Å)
    neighbors: list = _field(default_factory=list)  # indices of bonded atoms
    coordination: int = 0
    label: str = ""         # e.g., "Fe1", "Mo8" (crystallographic label)
    role: str = "metal"
    charge: int = 0
    projection_role: str = "metal_df"


@_dataclass
class BridgingAtom:
    """An atom bridging two or more metal centers."""
    element: str            # e.g., "S", "O", "C"
    index: int              # atom index in the full structure
    position: np.ndarray    # 3D coordinates
    bridged_metals: list = _field(default_factory=list)  # indices into ClusterInfo.metals
    role: str = "bridging"  # "bridging", "interstitial", "terminal"
    label: str = ""
    charge: int = 0
    ligand_type: str = ""
    projection_role: str = "bridging_p"


@_dataclass
class TerminalLigand:
    """A terminal ligand coordinated to a metal center."""
    name: str               # e.g., "cysteine_thiolate", "histidine_imidazole"
    atom_indices: list = _field(default_factory=list)  # atom indices in full structure
    donor_atom_index: int = -1
    charge: int = 0
    metal_index: int = -1   # index into ClusterInfo.metals
    label: str = ""
    role: str = "terminal"
    ligand_type: str = ""
    projection_role: str = "exclude"


@_dataclass
class ClusterInfo:
    """Complete description of a transition metal cluster."""
    metals: list[MetalCenter] = _field(default_factory=list)
    bridging_atoms: list[BridgingAtom] = _field(default_factory=list)
    terminal_ligands: list[TerminalLigand] = _field(default_factory=list)
    all_elements: list[str] = _field(default_factory=list)
    all_positions: _Optional[np.ndarray] = None
    formula: str = ""               # e.g., "Fe7MoS9C"
    total_charge: int = 0
    target_spin: float = 0.0        # total S quantum number
    symmetry_group: str = "C1"      # approximate point group
    metal_framework_symmetry: str = "C1"
    reduction_symmetry: str = "C1"
    symmetry_axis_atoms: list = _field(default_factory=list)  # e.g., [Fe1_idx, Mo_idx] for C3
    symmetry_source: str = "auto"
    symmetry_confidence: float = 0.0
    symmetry_candidates: list = _field(default_factory=list)
    family_scheme: str = ""
    benchmark_profile: str = ""
    config_reduction_mode: str = "none"
    cluster_info_path: str = ""
    annotation_source: str = "auto"


# ──────────────────────────────────────────────────────────────────
# Active space data classes
# ──────────────────────────────────────────────────────────────────

class ActiveSpaceLevel(_Enum):
    """Level of active space construction."""
    MINIMAL = "minimal"         # metal d only
    STANDARD = "standard"       # metal d + bridging p + interstitial
    EXTENDED = "extended"       # + ligand donors + correlating orbitals


@_dataclass
class OrbitalGroup:
    """A group of orbitals in the active space."""
    atom_label: str         # e.g., "Fe1", "S3", "C(interstitial)"
    orbital_type: str       # e.g., "3d", "3p", "2s2p"
    n_orbitals: int         # number of spatial orbitals
    n_electrons: int        # number of electrons contributed
    orbital_indices: list = _field(default_factory=list)  # indices in the active space


@_dataclass
class CAS:
    """Unified complete active space definition and orbital data.

    Combines the active space specification (electrons, orbitals, groups)
    with the computed orbital data (MO coefficients, occupations, labels).

    The ``stage`` field tracks the pipeline progression:
      ``"rule_based"`` → ``"computed"`` → ``"validated"``
    """
    # ── Shared ──
    n_electrons: int = 0
    n_orbitals: int = 0

    # ── Active-space specification ──
    orbital_groups: list[OrbitalGroup] = _field(default_factory=list)
    level: ActiveSpaceLevel = ActiveSpaceLevel.STANDARD
    n_qubits: int = 0       # 2 × n_orbitals for qubit mapping
    description: str = ""   # e.g., "(113e, 76o) LLDUC model"
    quality: _Optional['ActiveSpaceQuality'] = _field(default=None)
    stage: str = "rule_based"  # "rule_based" | "computed" | "validated"

    # ── Active-orbital data ──
    mo_coeff_alpha: _Optional[np.ndarray] = None  # (nao, nmo_active)
    mo_coeff_beta: _Optional[np.ndarray] = None   # (nao, nmo_active)
    occupations: _Optional[np.ndarray] = None     # UNO occupation numbers
    orbital_labels: list[str] = _field(default_factory=list)
    cpt_cas_type: str = "uno"  # "uno" | "luo" | "avas"
    source_method: str = ""  # e.g., "UKS-B3LYP/UNO"
    orbital_ordering: _Optional[np.ndarray] = None  # reordering for DMRG

    # ── Full orbital data (for visualization and user selection) ──
    mo_coeff_full: _Optional[np.ndarray] = None       # (nao, nmo) full localized MOs
    occupations_full: _Optional[np.ndarray] = None     # (nmo,) full UNO occupation numbers
    orbital_labels_full: list[str] = _field(default_factory=list)  # full orbital labels
    active_indices: list[int] = _field(default_factory=list)
    selection_method: str = ""
    projection_weights: _Optional[np.ndarray] = None
    projection_weights_metal: _Optional[np.ndarray] = None
    projection_weights_bridging: _Optional[np.ndarray] = None

    def __post_init__(self):
        self.n_qubits = 2 * self.n_orbitals

@_dataclass
class AVASConfig:
    """Configuration for AVAS-based active space construction."""
    avas_threshold: float = 0.4
    avas_valence_orbitals: dict = _field(default_factory=dict)
    # e.g. {"Fe": ["3d", "4s"], "S": ["3p"], "Mo": ["4d"]}


@_dataclass
class ActiveSpaceQuality:
    """Quality assessment result for an active space."""
    noon_values: _Optional[np.ndarray] = None          # natural orbital occupation numbers
    noon_warning: list[str] = _field(default_factory=list)  # e.g. ["3 orbitals with n > 1.98"]
    n_doubly_occupied: int = 0                        # orbitals with n ~ 2 (should not be in active space)
    n_empty: int = 0                                  # orbitals with n ~ 0
    entropy_per_orbital: list = _field(default_factory=list)  # single-orbital entropy (after DMRG)
    quality_score: float = 0.0                        # composite quality score 0-1
    orbital_character_map: dict = _field(default_factory=dict)  # {orb_idx: "Fe1_3d"} chemical labels
    missing_orbital_types: list[str] = _field(default_factory=list)  # e.g. ["S 3p"] expected but absent
    warnings: list[str] = _field(default_factory=list)  # consolidated warning list


# ──────────────────────────────────────────────────────────────────
# Spin configuration data classes
# ──────────────────────────────────────────────────────────────────

@_dataclass
class SpinIsomer:
    """A broken-symmetry spin isomer."""
    label: str                          # e.g., "BS8-237"
    spin_assignment: dict = _field(default_factory=dict)  # {metal_idx: +1 or -1}
    n_minority: int = 0                 # number of minority-spin metals
    family: str = ""                    # e.g., "BS8"
    Sz: float = 0.0                     # total Sz value
    symmetry_equivalent: list = _field(default_factory=list)  # labels of equivalent isomers


@_dataclass
class SpinIsomerFamily:
    """A family of symmetry-equivalent spin isomers."""
    label: str                          # e.g., "BS8"
    n_minority: int = 0
    isomers: list[SpinIsomer] = _field(default_factory=list)
    representative: _Optional[SpinIsomer] = None


# ──────────────────────────────────────────────────────────────────
# Electronic configuration data classes
# ──────────────────────────────────────────────────────────────────

@_dataclass
class OxidationAssignment:
    """Assignment of oxidation states to all metals."""
    assignments: dict = _field(default_factory=dict)  # {metal_idx: oxidation_state}
    # e.g., {0: +3, 1: +2, 2: +3, 3: +2, 4: +3, 5: +2, 6: +3}  (Fe indices)
    description: str = ""   # e.g., "3×Fe(II) + 4×Fe(III)"


@_dataclass
class ElectronicConfig:
    """A complete electronic configuration for a QC calculation.

    Combines spin isomer + oxidation assignment + d-orbital occupancy choices.
    This is the fundamental unit that maps to a single UHF initial guess.
    """
    spin_isomer: _Optional[SpinIsomer] = None
    oxidation: _Optional[OxidationAssignment] = None
    # d-orbital choice: for each metal with partially-filled minority-spin shell,
    # which d-orbital (0-4) gets the extra electron
    # e.g., {1: 2, 3: 0, 5: 4} means metal 1 → d_orbital #2, etc.
    d_orbital_assignments: dict = _field(default_factory=dict)
    # Derived convenience fields
    minority_spin_sites: list = _field(default_factory=list)
    spin_assignment: dict = _field(default_factory=dict)  # {metal_idx: +1/-1}
    config_id: int = 0      # unique index among all enumerated configs
    label: str = ""         # human-readable label


# ──────────────────────────────────────────────────────────────────
# Energy and results
# ──────────────────────────────────────────────────────────────────

@_dataclass
class CalculationResult:
    """Result from a single QC calculation."""
    config: _Optional[ElectronicConfig] = None
    method: str = ""        # e.g., "UHF", "UCCSD", "DMRG"
    energy: float = 0.0     # total energy in Hartrees
    correlation_energy: float = 0.0
    s_squared: float = 0.0  # ⟨S²⟩ expectation value
    converged: bool = False
    params: dict = _field(default_factory=dict)  # method-specific params


@_dataclass
class ExtrapolatedEnergy:
    """Extrapolated energy estimate."""
    method: str = ""        # e.g., "DMRG_D_extrapolation", "CC_composite"
    energy: float = 0.0
    uncertainty: float = 0.0
    fit_params: dict = _field(default_factory=dict)
    description: str = ""



# ──────────────────────────────────────────────────────────────────
# Stage 2 computation settings (Chan 2019 defaults)
# ──────────────────────────────────────────────────────────────────

@_dataclass
class ComputationSettings:
    """scf computation settings.

    Design principles:
    - All parameters are overridable for full user control.
    - **Defaults match the Chan 2019 paper**, ensuring reproducibility.
    - Use --preset fast to switch to a lightweight configuration (testing / small systems).

    Chan 2019 original: TZP-DKH (ADF), PySCF equivalent is def2-TZVP + sf-X2C.
    """
    # SCF method
    scf_method: str = "uks"                    # "uks" | "uhf"

    # DFT functional (only effective for uks mode)
    xc_functional: str = "B3LYP"

    # Basis set (default = Chan 2019 mixed basis)
    basis_set_default: str = "def2-TZVP"
    basis_set_per_element: dict = _field(default_factory=lambda: {
        "Fe": "def2-TZVP",
        "Mo": "def2-TZVP",
        "S":  "def2-TZVP",
        "C":  "def2-SVP",
        "H":  "def2-SVP",
        "O":  "def2-SVP",
        "N":  "def2-SVP",
    })
    # Priority: per_element > default

    # Relativistic correction (default = Chan 2019 sf-X2C)
    relativistic: str = "sf-x2c"               # "none" | "sf-x2c" | "dkh"

    # Solvation model (default = Chan 2019 ddCOSMO)
    solvation_model: str = "ddcosmo"           # "none" | "ddcosmo"
    solvation_epsilon: float = 4.0

    # SCF convergence parameters
    conv_tol: float = 1e-8
    max_cycle: int = 2000
    scf_verbose: int = 4          # PySCF SCF verbosity (4 = show iterations)
    scf_spin: float | None = None  # Optional high-spin SCF override as total S

    # SCF convergence helpers
    init_guess: str = "atom"          # "atom" | "minao" | "huckel" | "vsap"
    scf_damp: float = 0.0             # density damping (0 = off, try 0.1-0.5)
    scf_level_shift: float = 0.0      # level shift for virtual orbitals (0 = off, try 0.05-0.3)
    diis_space: int = 8               # number of DIIS vectors

    # SCF runtime decorators / stabilization
    density_fit: bool = False
    density_fit_auxbasis: str | None = None
    density_fit_only_dfj: bool = False
    grids_level: int = 3
    grids_small_rho_cutoff: float = 1e-7
    grids_prune: str = "nwchem"       # "nwchem" | "sg1" | "treutler" | "none"
    frac_occ: bool = False
    smearing_method: str = "none"     # "none" | "fermi" | "gauss"
    smearing_sigma: float = 0.01

    # Staged SCF controls
    scf_stage1_rough: bool = False
    scf_stage3_newton: bool = False
    newton_max_cycle: int = 10
    newton_conv_tol: float = 1e-10
    scf_allow_unconverged: bool = False

    # Localization controls shared by staged workflows
    pm_pop_method: str = "mulliken"
    pm_conv_tol: float = 1e-8
    pm_conv_tol_grad: float | None = None
    pm_max_cycle: int = 100
    pm_exponent: int = 2
    pm_init_guess: str = "atomic"
    boys_conv_tol: float = 1e-7
    boys_conv_tol_grad: float | None = None
    boys_max_cycle: int = 150

    def get_basis(self, element: str) -> str:
        """Return the basis set for a given element.

        Uses per-element config first, falls back to the default basis set.
        """
        return self.basis_set_per_element.get(element, self.basis_set_default)
