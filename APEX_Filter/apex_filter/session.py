"""Session state management for the APEX_Filter interactive pipeline.

Manages per-step persistence so each pipeline step can be run independently,
loading inputs from the previous step's saved state.
"""

import dataclasses
import json
import os
import shutil
from typing import Optional

import numpy as np
import yaml

from .hdf5_state_io import load_uhf_state_h5, save_uhf_state_h5
from .models import (
    BridgingAtom,
    CAS,
    ClusterInfo,
    ComputationSettings,
    ElectronicConfig,
    MetalCenter,
    OrbitalGroup,
    OxidationAssignment,
    SpinIsomer,
    SpinIsomerFamily,
    TerminalLigand,
)


# ──────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────

_JSON_FILE_DESCRIPTIONS = {
    "session.json": (
        "session_state",
        "Internal session state. Records which pipeline steps are completed and the original filter config path.",
    ),
    "cluster_info.json": (
        "cluster_info_snapshot",
        "Resolved cluster annotations used by APEX_Filter Step 1. This snapshot drives later role-, charge-, and symmetry-aware steps.",
    ),
    "cas_meta.json": (
        "cas_metadata",
        "Scalar CAS metadata loaded from APEX_CAS output. Array-valued data are stored separately in cas_arrays.npz.",
    ),
    "fcidump_ref.json": (
        "fcidump_reference",
        "Reference to the concrete FCIDUMP file used to initialize this filter session.",
    ),
    "settings.json": (
        "effective_settings",
        "Effective APEX_Filter/APEX_CAS settings used for this session after applying provenance and filter-side overrides.",
    ),
    "enumeration.json": (
        "enumeration_results",
        "Spin/oxidation/d-assignment enumeration results for Step 2. The actual config list is stored under the 'records' payload fields.",
    ),
    "enumeration_layers.json": (
        "enumeration_statistics",
        "Layer-by-layer counts for Step 2 enumeration. Use this file to inspect how many patterns survive each reduction stage.",
    ),
    "picked_configs.json": (
        "picked_labels_record",
        "Internal record of which labels were actually selected and sent into this step.",
    ),
    "uhf_summary.json": (
        "step3_summary",
        "Step 3 UHF summary. Each record corresponds to one UHF reference calculation and stores the final state signature when available.",
    ),
    "ccsd_summary.json": (
        "step4_summary",
        "Step 4 CCSD summary. Each record corresponds to one selected state and stores the best available CCSD energy and convergence flag.",
    ),
    "ccsd_t_summary.json": (
        "step5_summary",
        "Step 5 CCSD(T) summary. Each record corresponds to one selected state and stores the best available CCSD(T) energy and convergence flag.",
    ),
    "ccsdt_summary.json": (
        "step6_summary",
        "Step 6 CCSDT summary. Each record corresponds to one selected state and stores the best available CCSDT energy and convergence flag.",
    ),
    "dmrg_basis_summary.json": (
        "step7_summary",
        "Step 7 DMRG-basis summary. Each record tracks whether a DMRG basis was built successfully and which localization route was used.",
    ),
    "dmrg_basis_qc.json": (
        "step7_basis_qc",
        "Step 7 DMRG-basis quality-control metrics. Use these diagnostics to judge orthogonality, alpha/beta pairing quality, and ordering quality.",
    ),
    "dmrg_summary.json": (
        "step8_summary",
        "Step 8 DMRG summary. Each record stores the energy for one state/bond-dimension pair; unconverged runs retain the last available energy when possible.",
    ),
    "dmrg_extrapolation_summary.json": (
        "step9_summary",
        "Step 9 DMRG extrapolation summary. Each record stores an extrapolated infinite-bond-dimension energy for one selected state.",
    ),
    "final_summary.json": (
        "final_ranking_summary",
        "Final ranking summary combining the highest-level data available for each state.",
    ),
    "fno_summary.json": (
        "step11_summary",
        "Step 11 FNO-UCCSDTQ summary. Each record stores the frozen-natural-orbital composite results for one selected state.",
    ),
    "cc_composite_summary.json": (
        "step12_summary",
        "Step 12 CC composite summary. Each record stores the composite coupled-cluster estimate for one selected state.",
    ),
}


def _json_file_description(path: str) -> tuple[str, str]:
    name = os.path.basename(path)
    return _JSON_FILE_DESCRIPTIONS.get(
        name,
        (
            "json_artifact",
            "APEX_Filter JSON artifact. See the surrounding session step directory and filename for its role in the workflow.",
        ),
    )


def _wrap_json_payload(path: str, data):
    role, comment = _json_file_description(path)
    if isinstance(data, list):
        return {
            "_file_role": role,
            "_comment": comment,
            "records": data,
        }
    if isinstance(data, dict):
        payload = {
            "_file_role": role,
            "_comment": comment,
        }
        payload.update(data)
        return payload
    return data


def _unwrap_json_payload(data):
    if isinstance(data, dict):
        if "records" in data and ("_file_role" in data or "_comment" in data):
            return data["records"]
        if "_file_role" in data or "_comment" in data:
            return {k: v for k, v in data.items() if k not in {"_file_role", "_comment"}}
    return data


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return _unwrap_json_payload(json.load(f))


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_wrap_json_payload(path, data), f, indent=2, ensure_ascii=False)

def _metal_to_dict(m: MetalCenter) -> dict:
    return {
        "element": m.element,
        "index": m.index,
        "position": m.position.tolist() if m.position is not None else [],
        "neighbors": m.neighbors,
        "coordination": m.coordination,
        "label": m.label,
        "role": getattr(m, "role", "metal"),
        "charge": getattr(m, "charge", 0),
        "projection_role": getattr(m, "projection_role", "metal_df"),
    }


def _metal_from_dict(d: dict) -> MetalCenter:
    return MetalCenter(
        element=d["element"],
        index=d["index"],
        position=np.array(d["position"], dtype=float),
        neighbors=d.get("neighbors", []),
        coordination=d.get("coordination", 0),
        label=d.get("label", ""),
        role=d.get("role", "metal"),
        charge=d.get("charge", 0),
        projection_role=d.get("projection_role", "metal_df"),
    )


def _bridge_to_dict(b: BridgingAtom) -> dict:
    return {
        "element": b.element,
        "index": b.index,
        "position": b.position.tolist() if b.position is not None else [],
        "bridged_metals": b.bridged_metals,
        "role": b.role,
        "label": getattr(b, "label", ""),
        "charge": getattr(b, "charge", 0),
        "ligand_type": getattr(b, "ligand_type", ""),
        "projection_role": getattr(b, "projection_role", "bridging_p"),
    }


def _bridge_from_dict(d: dict) -> BridgingAtom:
    return BridgingAtom(
        element=d["element"],
        index=d["index"],
        position=np.array(d["position"], dtype=float),
        bridged_metals=d.get("bridged_metals", []),
        role=d.get("role", "bridging"),
        label=d.get("label", ""),
        charge=d.get("charge", 0),
        ligand_type=d.get("ligand_type", ""),
        projection_role=d.get("projection_role", "bridging_p"),
    )


def _ligand_to_dict(lg: TerminalLigand) -> dict:
    return {
        "name": lg.name,
        "atom_indices": lg.atom_indices,
        "donor_atom_index": lg.donor_atom_index,
        "charge": lg.charge,
        "metal_index": lg.metal_index,
        "label": getattr(lg, "label", ""),
        "role": getattr(lg, "role", "terminal"),
        "ligand_type": getattr(lg, "ligand_type", ""),
        "projection_role": getattr(lg, "projection_role", "exclude"),
    }


def _ligand_from_dict(d: dict) -> TerminalLigand:
    return TerminalLigand(
        name=d["name"],
        atom_indices=d.get("atom_indices", []),
        donor_atom_index=d.get("donor_atom_index", -1),
        charge=d.get("charge", 0),
        metal_index=d.get("metal_index", -1),
        label=d.get("label", ""),
        role=d.get("role", "terminal"),
        ligand_type=d.get("ligand_type", ""),
        projection_role=d.get("projection_role", "exclude"),
    )


def cluster_info_to_dict(ci: ClusterInfo) -> dict:
    return {
        "metals": [_metal_to_dict(m) for m in ci.metals],
        "bridging_atoms": [_bridge_to_dict(b) for b in ci.bridging_atoms],
        "terminal_ligands": [_ligand_to_dict(lg) for lg in ci.terminal_ligands],
        "all_elements": ci.all_elements,
        "all_positions": ci.all_positions.tolist() if ci.all_positions is not None else None,
        "formula": ci.formula,
        "total_charge": ci.total_charge,
        "target_spin": ci.target_spin,
        "symmetry_group": ci.symmetry_group,
        "metal_framework_symmetry": getattr(ci, "metal_framework_symmetry", "C1"),
        "reduction_symmetry": getattr(ci, "reduction_symmetry", ci.symmetry_group),
        "symmetry_axis_atoms": ci.symmetry_axis_atoms,
        "symmetry_source": getattr(ci, "symmetry_source", "auto"),
        "symmetry_confidence": getattr(ci, "symmetry_confidence", 0.0),
        "symmetry_candidates": getattr(ci, "symmetry_candidates", []),
        "family_scheme": getattr(ci, "family_scheme", ""),
        "benchmark_profile": getattr(ci, "benchmark_profile", ""),
        "config_reduction_mode": getattr(ci, "config_reduction_mode", "none"),
        "cluster_info_path": getattr(ci, "cluster_info_path", ""),
        "annotation_source": getattr(ci, "annotation_source", "auto"),
    }


def cluster_info_from_dict(d: dict) -> ClusterInfo:
    return ClusterInfo(
        metals=[_metal_from_dict(m) for m in d.get("metals", [])],
        bridging_atoms=[_bridge_from_dict(b) for b in d.get("bridging_atoms", [])],
        terminal_ligands=[_ligand_from_dict(lg) for lg in d.get("terminal_ligands", [])],
        all_elements=d.get("all_elements", []),
        all_positions=np.array(d["all_positions"], dtype=float) if d.get("all_positions") is not None else None,
        formula=d.get("formula", ""),
        total_charge=d.get("total_charge", 0),
        target_spin=d.get("target_spin", 0.0),
        symmetry_group=d.get("symmetry_group", "C1"),
        metal_framework_symmetry=d.get("metal_framework_symmetry", "C1"),
        reduction_symmetry=d.get("reduction_symmetry", d.get("symmetry_group", "C1")),
        symmetry_axis_atoms=d.get("symmetry_axis_atoms", []),
        symmetry_source=d.get("symmetry_source", "auto"),
        symmetry_confidence=d.get("symmetry_confidence", 0.0),
        symmetry_candidates=d.get("symmetry_candidates", []),
        family_scheme=d.get("family_scheme", ""),
        benchmark_profile=d.get("benchmark_profile", ""),
        config_reduction_mode=d.get("config_reduction_mode", "none"),
        cluster_info_path=d.get("cluster_info_path", ""),
        annotation_source=d.get("annotation_source", "auto"),
    )


# CAS scalar fields that go into cas_meta.json
_CAS_SCALAR_FIELDS = [
    "n_electrons", "n_orbitals", "n_qubits", "description",
    "cpt_cas_type", "source_method", "selection_method",
]

_CAS_LIST_FIELDS = [
    "orbital_labels", "orbital_labels_full", "active_indices",
]

# CAS numpy array fields that go into cas_arrays.npz
_CAS_ARRAY_FIELDS = [
    "mo_coeff_alpha", "mo_coeff_beta", "occupations", "orbital_ordering",
    "mo_coeff_full", "occupations_full",
    "projection_weights", "projection_weights_metal", "projection_weights_bridging",
]


def cas_to_json_and_npz(cas: CAS):
    """Split CAS into scalar dict (for JSON) and array dict (for NPZ)."""
    meta = {}
    # Scalar fields
    for f in _CAS_SCALAR_FIELDS:
        v = getattr(cas, f, None)
        if v is not None:
            meta[f] = v if not isinstance(v, type) else str(v)
    # List fields
    for f in _CAS_LIST_FIELDS:
        v = getattr(cas, f, None)
        if v is not None:
            meta[f] = v
    # Orbital groups
    if cas.orbital_groups:
        meta["orbital_groups"] = [dataclasses.asdict(og) for og in cas.orbital_groups]
    # quality
    if cas.quality is not None:
        q = cas.quality
        qd = {}
        for fld in dataclasses.fields(q):
            v = getattr(q, fld.name)
            if isinstance(v, np.ndarray):
                qd[fld.name] = v.tolist()
            else:
                qd[fld.name] = v
        meta["quality"] = qd

    arrays = {}
    for f in _CAS_ARRAY_FIELDS:
        v = getattr(cas, f, None)
        if v is not None:
            arrays[f] = np.asarray(v)

    return meta, arrays


def cas_from_json_and_npz(meta: dict, arrays: dict) -> CAS:
    """Reconstruct CAS from scalar dict + NPZ arrays dict."""
    from .models import ActiveSpaceLevel, ActiveSpaceQuality

    kwargs = {}
    for f in _CAS_SCALAR_FIELDS:
        if f in meta:
            kwargs[f] = meta[f]
    # list fields
    for f in _CAS_LIST_FIELDS:
        if f in meta:
            kwargs[f] = meta[f]
    # orbital groups
    if "orbital_groups" in meta:
        kwargs["orbital_groups"] = [OrbitalGroup(**og) for og in meta["orbital_groups"]]
    # quality
    if "quality" in meta:
        qd = meta["quality"]
        # Convert list fields back
        if "noon_values" in qd and isinstance(qd["noon_values"], list):
            qd["noon_values"] = np.array(qd["noon_values"])
        kwargs["quality"] = ActiveSpaceQuality(**qd)
    # arrays
    for f in _CAS_ARRAY_FIELDS:
        if f in arrays:
            kwargs[f] = arrays[f]

    return CAS(**kwargs)


# ──────────────────────────────────────────────────────────────────
# Spin/Electronic config serialization
# ──────────────────────────────────────────────────────────────────

def spin_isomer_to_dict(si: SpinIsomer) -> dict:
    return {
        "label": si.label,
        "spin_assignment": si.spin_assignment,
        "n_minority": si.n_minority,
        "family": si.family,
        "Sz": si.Sz,
        "symmetry_equivalent": si.symmetry_equivalent,
    }


def spin_isomer_from_dict(d: dict) -> SpinIsomer:
    return SpinIsomer(
        label=d["label"],
        spin_assignment={int(k): v for k, v in d.get("spin_assignment", {}).items()},
        n_minority=d.get("n_minority", 0),
        family=d.get("family", ""),
        Sz=d.get("Sz", 0.0),
        symmetry_equivalent=d.get("symmetry_equivalent", []),
    )


def oxidation_to_dict(oa: OxidationAssignment) -> dict:
    return {
        "assignments": oa.assignments,
        "description": oa.description,
    }


def oxidation_from_dict(d: dict) -> OxidationAssignment:
    return OxidationAssignment(
        assignments={int(k): v for k, v in d.get("assignments", {}).items()},
        description=d.get("description", ""),
    )


def electronic_config_to_dict(cfg: ElectronicConfig) -> dict:
    d = {
        "spin_isomer": spin_isomer_to_dict(cfg.spin_isomer) if cfg.spin_isomer else None,
        "oxidation": oxidation_to_dict(cfg.oxidation) if cfg.oxidation else None,
        "d_orbital_assignments": cfg.d_orbital_assignments,
        "minority_spin_sites": cfg.minority_spin_sites,
        "spin_assignment": cfg.spin_assignment,
        "config_id": cfg.config_id,
        "label": cfg.label,
    }
    return d


def electronic_config_from_dict(d: dict) -> ElectronicConfig:
    return ElectronicConfig(
        spin_isomer=spin_isomer_from_dict(d["spin_isomer"]) if d.get("spin_isomer") else None,
        oxidation=oxidation_from_dict(d["oxidation"]) if d.get("oxidation") else None,
        d_orbital_assignments={int(k): v for k, v in d.get("d_orbital_assignments", {}).items()},
        minority_spin_sites=[int(v) for v in d.get("minority_spin_sites", [])],
        spin_assignment={int(k): v for k, v in d.get("spin_assignment", {}).items()},
        config_id=d.get("config_id", 0),
        label=d.get("label", ""),
    )


def spin_isomer_family_to_dict(fam: SpinIsomerFamily) -> dict:
    return {
        "label": fam.label,
        "n_minority": fam.n_minority,
        "isomers": [spin_isomer_to_dict(iso) for iso in fam.isomers],
        "representative": spin_isomer_to_dict(fam.representative) if fam.representative else None,
    }


def spin_isomer_family_from_dict(d: dict) -> SpinIsomerFamily:
    return SpinIsomerFamily(
        label=d["label"],
        n_minority=d.get("n_minority", 0),
        isomers=[spin_isomer_from_dict(iso) for iso in d.get("isomers", [])],
        representative=spin_isomer_from_dict(d["representative"]) if d.get("representative") else None,
    )


def _sanitize_label(label: str) -> str:
    """Return a filesystem-safe label."""
    return label.replace("|", "_").replace(" ", "_")


def _npz_scalar(data, key, default=None):
    if hasattr(data, "files"):
        if key not in data.files:
            return default
        value = data[key]
    else:
        if key not in data:
            return default
        value = data[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.bool_):
        value = bool(value)
    if isinstance(value, np.ndarray) and value.dtype.kind == "S" and value.shape == ():
        value = value.astype(str).item()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


# ──────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────

_STEP_DIRS = {
    "step1_load": "step1_load",
    "step2_enumerate": "step2_enumerate",
    "step3_uhf": "step3_uhf",
    "step4_ccsd": "step4_ccsd",
    "step5_ccsd_t": "step5_ccsd_t",
    "step6_ccsdt": "step6_ccsdt",
    "step7_dmrg_basis": "step7_dmrg_basis",
    "step8_dmrg": "step8_dmrg",
    "step9_extrapolate": "step9_extrapolate",
    "step10_report": "step10_report",
    "step11_fno_uccsdtq": "step11_fno_uccsdtq",
    "step12_cc_composite": "step12_cc_composite",
}

_STEP_ORDER = [
    "step1_load",
    "step2_enumerate",
    "step3_uhf",
    "step4_ccsd",
    "step5_ccsd_t",
    "step6_ccsdt",
    "step7_dmrg_basis",
    "step8_dmrg",
    "step9_extrapolate",
    "step10_report",
    "step11_fno_uccsdtq",
    "step12_cc_composite",
]

_STEP_PREV = {
    "step2_enumerate": "step1_load",
    "step3_uhf": "step2_enumerate",
    "step4_ccsd": "step3_uhf",
    "step5_ccsd_t": "step4_ccsd",
    "step6_ccsdt": "step5_ccsd_t",
    "step7_dmrg_basis": "step6_ccsdt",
    "step8_dmrg": "step7_dmrg_basis",
    "step9_extrapolate": "step8_dmrg",
    "step10_report": "step9_extrapolate",
    "step11_fno_uccsdtq": "step6_ccsdt",
    "step12_cc_composite": "step11_fno_uccsdtq",
}

def _load_method_controls_template() -> dict:
    """Load the shared method-controls template used to seed new sessions."""
    template_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "shared",
            "config",
            "method_controls_template.yaml",
        )
    )
    with open(template_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid method controls template: {template_path}")
    return data


_DEFAULT_METHOD_CONTROLS = _load_method_controls_template()


class SessionManager:
    """Manages pipeline session state across independent CLI invocations."""

    def __init__(self, session_dir: str):
        self.session_dir = os.path.abspath(session_dir)
        self._session_json = os.path.join(self.session_dir, "session.json")
        self._method_controls_yaml = os.path.join(self.session_dir, "method_controls.yaml")
        self._migrate_legacy_step5_dir()

    def _migrate_legacy_step5_dir(self):
        """Rename legacy step5 directory ``step5_ccsdt`` to ``step5_ccsd_t``."""
        legacy = os.path.join(self.session_dir, "step5_ccsdt")
        current = os.path.join(self.session_dir, "step5_ccsd_t")
        if not os.path.exists(legacy):
            return
        if not os.path.exists(current):
            os.rename(legacy, current)
            return

        for root, dirs, files in os.walk(legacy):
            rel = os.path.relpath(root, legacy)
            dst_root = current if rel == "." else os.path.join(current, rel)
            os.makedirs(dst_root, exist_ok=True)
            for name in files:
                src = os.path.join(root, name)
                dst = os.path.join(dst_root, name)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
        shutil.rmtree(legacy)

    # ── Session lifecycle ──────────────────────────────────────────

    def create(self):
        """Create a new session directory structure."""
        os.makedirs(self.session_dir, exist_ok=True)
        for step_dir in _STEP_DIRS.values():
            os.makedirs(os.path.join(self.session_dir, step_dir), exist_ok=True)
        # results subdirs
        os.makedirs(os.path.join(self.session_dir, "step3_uhf", "results"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step4_ccsd", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step5_ccsd_t", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step6_ccsdt", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step7_dmrg_basis", "results"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step8_dmrg", "results"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step10_report"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step11_fno_uccsdtq", "results"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "step12_cc_composite"), exist_ok=True)

        if not os.path.exists(self._session_json):
            self._write_session({"completed_steps": [], "config_path": None})
        self.ensure_method_controls()

    def load(self) -> dict:
        """Load session metadata."""
        return _read_json(self._session_json)

    def _write_session(self, data: dict):
        _write_json(self._session_json, data)

    @property
    def method_controls_path(self) -> str:
        return self._method_controls_yaml

    def ensure_method_controls(self):
        """Create the session-local method control file if it does not exist."""
        if os.path.exists(self._method_controls_yaml):
            return
        with open(self._method_controls_yaml, "w") as f:
            yaml.safe_dump(_DEFAULT_METHOD_CONTROLS, f, sort_keys=False)

    def load_method_controls(self) -> dict:
        self.ensure_method_controls()
        with open(self._method_controls_yaml) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid method control file: {self._method_controls_yaml}")
        return data

    def resolve_method_controls(self, section: str, defaults: dict, cli_values: dict | None = None) -> dict:
        """
        Resolve method controls from built-in defaults, session YAML, and CLI values.

        Precedence:
        1. built-in defaults
        2. session-local method_controls.yaml
        3. CLI values that differ from the built-in defaults
        """
        controls = self.load_method_controls()
        resolved = dict(defaults)
        session_values = controls.get(section, {})
        if isinstance(session_values, dict):
            for key, value in session_values.items():
                if value is not None:
                    resolved[key] = value
        if cli_values:
            for key, value in cli_values.items():
                if key not in defaults or value != defaults[key]:
                    resolved[key] = value
        return resolved

    def mark_step_completed(self, step: str):
        data = self.load()
        if step not in data["completed_steps"]:
            data["completed_steps"].append(step)
        self._write_session(data)

    def require_previous(self, step: str, required: str):
        """Raise if *required* step has not been completed."""
        data = self.load()
        if required not in data.get("completed_steps", []):
            raise RuntimeError(
                f"Step '{step}' requires '{required}' to be completed first. "
                f"Completed: {data.get('completed_steps', [])}"
            )

    def _step_dir(self, step: str) -> str:
        return os.path.join(self.session_dir, _STEP_DIRS[step])

    # ── Step 1: Load ──────────────────────────────────────────────

    def save_load_state(
        self,
        cluster_info,
        cas,
        fcidump_path,
        settings,
        config_path,
        apex_cas_provenance: dict | None = None,
    ):
        d = self._step_dir("step1_load")
        # cluster_info
        _write_json(os.path.join(d, "cluster_info.json"), cluster_info_to_dict(cluster_info))
        # CAS: scalar → JSON, arrays → NPZ
        meta, arrays = cas_to_json_and_npz(cas)
        _write_json(os.path.join(d, "cas_meta.json"), meta)
        if arrays:
            np.savez(os.path.join(d, "cas_arrays.npz"), **arrays)
        # FCIDUMP path reference
        _write_json(os.path.join(d, "fcidump_ref.json"), {"fcidump_path": os.path.abspath(fcidump_path)})
        # ComputationSettings
        settings_payload = dataclasses.asdict(settings)
        if settings.solvation_model == "none":
            settings_payload.pop("solvation_epsilon", None)
        if apex_cas_provenance:
            settings_payload["apex_cas_provenance"] = apex_cas_provenance
        _write_json(os.path.join(d, "settings.json"), settings_payload)
        # session metadata
        data = self.load()
        data["config_path"] = os.path.abspath(config_path)
        self._write_session(data)
        self.mark_step_completed("step1_load")

    def load_load_state(self) -> dict:
        d = self._step_dir("step1_load")
        ci = cluster_info_from_dict(_read_json(os.path.join(d, "cluster_info.json")))
        cas_meta = _read_json(os.path.join(d, "cas_meta.json"))
        cas_arrays = {}
        npz_path = os.path.join(d, "cas_arrays.npz")
        if os.path.exists(npz_path):
            npz = np.load(npz_path, allow_pickle=True)
            cas_arrays = {k: npz[k] for k in npz.files}
        cas = cas_from_json_and_npz(cas_meta, cas_arrays)
        fcidump_path = _read_json(os.path.join(d, "fcidump_ref.json"))["fcidump_path"]
        from .CAS_loader import load_fcidump
        fcidump_data = load_fcidump(fcidump_path)
        raw_settings = _read_json(os.path.join(d, "settings.json"))
        raw_settings.pop("apex_cas_provenance", None)
        settings = ComputationSettings(**raw_settings)
        return {
            "cluster_info": ci,
            "cas": cas,
            "fcidump_data": fcidump_data,
            "settings": settings,
            "fcidump_path": fcidump_path,
            "config_path": self.load().get("config_path"),
        }

    # ── Step 2: Enumerate ──────────────────────────────────────────

    def save_enumeration(self, configs, spin_isomers, families, n_total, stats=None):
        d = self._step_dir("step2_enumerate")
        _write_json(
            os.path.join(d, "enumeration.json"),
            {
                "configs": [electronic_config_to_dict(c) for c in configs],
                "spin_isomers": [spin_isomer_to_dict(si) for si in spin_isomers],
                "families": [spin_isomer_family_to_dict(fam) for fam in families],
                "n_total": n_total,
                "stats": stats or {},
            },
        )
        self.mark_step_completed("step2_enumerate")

    def load_enumeration(self) -> dict:
        d = self._step_dir("step2_enumerate")
        data = _read_json(os.path.join(d, "enumeration.json"))
        return {
            "configs": [electronic_config_from_dict(c) for c in data["configs"]],
            "spin_isomers": [spin_isomer_from_dict(si) for si in data["spin_isomers"]],
            "families": [spin_isomer_family_from_dict(fam) for fam in data["families"]],
            "n_total": data["n_total"],
            "stats": data.get("stats", {}),
        }

    # ── Step 3: UHF ──────────────────────────────────────────────

    def save_uhf_picked(self, labels):
        d = self._step_dir("step3_uhf")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_uhf_picked(self) -> list[str]:
        d = self._step_dir("step3_uhf")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_uhf_result(self, label: str, result, *, family: str = "", state: dict | None = None):
        """Save a single UHF result as both legacy NPZ and structured HDF5."""
        d = os.path.join(self._step_dir("step3_uhf"), "results")
        safe = _sanitize_label(label)
        save_dict = {
            "energy": result.energy if result.energy is not None else 0.0,
            "converged": result.converged if result.converged is not None else False,
            "spin_sq": result.s_squared if result.s_squared is not None else 0.0,
        }
        if hasattr(result, "mo_coeff") and result.mo_coeff[0] is not None:
            save_dict["mo_coeff_a"] = result.mo_coeff[0]
            save_dict["mo_coeff_b"] = result.mo_coeff[1]
        if hasattr(result, "mo_occ") and result.mo_occ[0] is not None:
            save_dict["mo_occ_a"] = result.mo_occ[0]
            save_dict["mo_occ_b"] = result.mo_occ[1]
        if hasattr(result, "mo_energy") and result.mo_energy[0] is not None:
            save_dict["mo_energy_a"] = result.mo_energy[0]
            save_dict["mo_energy_b"] = result.mo_energy[1]
        if hasattr(result, "dm") and result.dm[0] is not None:
            save_dict["dm_a"] = result.dm[0]
            save_dict["dm_b"] = result.dm[1]
        diagnostics = getattr(result, "diagnostics", None) or {}
        if diagnostics.get("bs_stabilize_history"):
            save_dict["bs_stabilize_energy_history"] = np.array(
                [entry["energy"] for entry in diagnostics["bs_stabilize_history"]],
                dtype=float,
            )
            save_dict["bs_stabilize_delta_e_history"] = np.array(
                [entry["delta_e"] for entry in diagnostics["bs_stabilize_history"]],
                dtype=float,
            )
        if diagnostics.get("bs_tight_history"):
            save_dict["bs_tight_energy_history"] = np.array(
                [entry["energy"] for entry in diagnostics["bs_tight_history"]],
                dtype=float,
            )
            save_dict["bs_tight_delta_e_history"] = np.array(
                [entry["delta_e"] for entry in diagnostics["bs_tight_history"]],
                dtype=float,
            )
        if diagnostics.get("newton_history"):
            save_dict["newton_energy_history"] = np.array(
                [entry["energy"] for entry in diagnostics["newton_history"]],
                dtype=float,
            )
            save_dict["newton_delta_e_history"] = np.array(
                [entry["delta_e"] for entry in diagnostics["newton_history"]],
                dtype=float,
            )
        if diagnostics.get("final_delta_e") is not None:
            save_dict["final_delta_e"] = float(diagnostics["final_delta_e"])
        if diagnostics.get("final_state_signature") is not None:
            save_dict["final_state_signature"] = np.array(
                diagnostics["final_state_signature"], dtype=object
            )
        if diagnostics.get("final_d_basin") is not None:
            save_dict["final_d_basin_json"] = np.array(
                json.dumps(diagnostics["final_d_basin"]), dtype=object
            )
        if diagnostics.get("final_site_spin_proxy") is not None:
            save_dict["final_site_spin_proxy_json"] = np.array(
                json.dumps(diagnostics["final_site_spin_proxy"]), dtype=object
            )
        if state is None:
            try:
                state = self.load_load_state()
            except Exception:
                state = {}
        save_uhf_state_h5(
            os.path.join(d, f"{safe}_uhf.h5"),
            save_dict,
            label=label,
            family=family,
            settings=state.get("settings"),
            cluster_info=state.get("cluster_info"),
            fcidump_data=state.get("fcidump_data"),
            cas=state.get("cas"),
        )
        np.savez(os.path.join(d, f"{safe}_uhf.npz"), **save_dict)

    def save_uhf_summary(self, results: list[dict]):
        d = self._step_dir("step3_uhf")
        _write_json(os.path.join(d, "uhf_summary.json"), results)
        self.mark_step_completed("step3_uhf")

    def load_uhf_summary(self) -> list[dict]:
        d = self._step_dir("step3_uhf")
        return _read_json(os.path.join(d, "uhf_summary.json"))

    def rebuild_uhf_summary(self, configs: list[ElectronicConfig], current_results: list[dict] | None = None) -> list[dict]:
        """Rebuild the step3 summary from all saved UHF state artifacts.

        This keeps historical UHF results visible even if the latest `uhf`
        command only recomputed a subset of labels.
        """
        existing_summary = []
        summary_path = os.path.join(self._step_dir("step3_uhf"), "uhf_summary.json")
        if os.path.exists(summary_path):
            existing_summary = _read_json(summary_path)

        merged_by_label = {row["label"]: dict(row) for row in existing_summary if row.get("label")}
        for row in current_results or []:
            if row.get("label"):
                merged_by_label[row["label"]] = dict(row)

        config_by_safe = {_sanitize_label(cfg.label): cfg for cfg in configs}
        results_dir = os.path.join(self._step_dir("step3_uhf"), "results")
        state = None
        rebuilt = []
        preferred_artifacts = {}
        for name in sorted(os.listdir(results_dir)):
            if name.endswith("_uhf.h5"):
                preferred_artifacts[name[:-7]] = name
            elif name.endswith("_uhf.npz") and name[:-8] not in preferred_artifacts:
                preferred_artifacts[name[:-8]] = name

        for safe_label, name in sorted(preferred_artifacts.items(), key=lambda item: item[1]):
            cfg = config_by_safe.get(safe_label)
            if cfg is None:
                continue
            label = cfg.label
            prev = merged_by_label.get(label, {})
            artifact_path = os.path.join(results_dir, name)
            if name.endswith(".h5"):
                state_data = load_uhf_state_h5(artifact_path)
            else:
                npz = np.load(artifact_path, allow_pickle=True)
                state_data = {key: npz[key] for key in npz.files}

            energy = float(_npz_scalar(state_data, "energy", 0.0))
            converged = bool(_npz_scalar(state_data, "converged", False))
            s_squared = float(_npz_scalar(state_data, "spin_sq", 0.0))
            last_delta_e = _npz_scalar(state_data, "final_delta_e", None)
            if last_delta_e is not None:
                last_delta_e = float(last_delta_e)

            energy_tail = prev.get("energy_tail", [])
            for key in ("newton_energy_history", "bs_tight_energy_history", "bs_stabilize_energy_history"):
                if key in state_data and len(state_data[key]) > 0:
                    energy_tail = [float(v) for v in state_data[key][-5:]]
                    break

            final_state_signature = prev.get("final_state_signature")
            npz_final_sig = _npz_scalar(state_data, "final_state_signature", None)
            if npz_final_sig is not None:
                final_state_signature = str(npz_final_sig)

            final_d_basin = prev.get("final_d_basin", {})
            npz_final_d_basin = _npz_scalar(state_data, "final_d_basin_json", None)
            if npz_final_d_basin:
                final_d_basin = json.loads(str(npz_final_d_basin))

            final_site_spin_proxy = prev.get("final_site_spin_proxy", {})
            npz_final_site_spin_proxy = _npz_scalar(state_data, "final_site_spin_proxy_json", None)
            if npz_final_site_spin_proxy:
                final_site_spin_proxy = json.loads(str(npz_final_site_spin_proxy))

            if (
                final_state_signature is None
                and "dm_a" in state_data
                and "dm_b" in state_data
            ):
                if state is None:
                    state = self.load_load_state()
                from .reference_uhf import _summarize_final_state_from_dm
                inferred = _summarize_final_state_from_dm(
                    state["cas"],
                    cfg,
                    state["cluster_info"],
                    (np.asarray(state_data["dm_a"]), np.asarray(state_data["dm_b"])),
                )
                final_d_basin = inferred.get("final_d_basin", final_d_basin)
                final_site_spin_proxy = inferred.get("final_site_spin_proxy", final_site_spin_proxy)
                final_state_signature = inferred.get("final_state_signature", final_state_signature)

            rebuilt.append(
                {
                    "label": label,
                    "display_label": prev.get("display_label") or final_state_signature or label,
                    "energy": energy,
                    "converged": converged,
                    "s_squared": s_squared,
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "last_delta_e": last_delta_e,
                    "energy_tail": energy_tail,
                    "final_d_basin": final_d_basin,
                    "final_site_spin_proxy": final_site_spin_proxy,
                    "final_state_signature": final_state_signature,
                }
            )

        rebuilt.sort(key=lambda row: row["energy"] if row.get("energy") is not None else float("inf"))
        return rebuilt

    # ── Step 4: CCSD ──────────────────────────────────────────────

    def save_ccsd_picked(self, labels):
        d = self._step_dir("step4_ccsd")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_ccsd_picked(self) -> list[str]:
        d = self._step_dir("step4_ccsd")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_ccsd_summary(self, results: list[dict]):
        d = self._step_dir("step4_ccsd")
        _write_json(os.path.join(d, "ccsd_summary.json"), results)
        self.mark_step_completed("step4_ccsd")

    def load_ccsd_summary(self) -> list[dict]:
        d = self._step_dir("step4_ccsd")
        return _read_json(os.path.join(d, "ccsd_summary.json"))

    def rebuild_ccsd_summary(
        self,
        configs: list[ElectronicConfig],
        upstream_summary: list[dict] | None = None,
        current_results: list[dict] | None = None,
    ) -> list[dict]:
        """Rebuild the step4 CCSD summary from all saved NPZ results."""
        existing_summary = []
        summary_path = os.path.join(self._step_dir("step4_ccsd"), "ccsd_summary.json")
        if os.path.exists(summary_path):
            existing_summary = _read_json(summary_path)

        merged_by_label = {row["label"]: dict(row) for row in existing_summary if row.get("label")}
        for row in current_results or []:
            if row.get("label"):
                merged_by_label[row["label"]] = dict(row)

        upstream_by_label = {row["label"]: row for row in (upstream_summary or []) if row.get("label")}
        config_by_safe = {_sanitize_label(cfg.label): cfg for cfg in configs}
        results_dir = self.ccsd_scripts_dir
        rebuilt = []
        for name in sorted(os.listdir(results_dir)):
            if not name.endswith("_ccsd_results.npz"):
                continue
            safe_label = name[:-17]
            cfg = config_by_safe.get(safe_label)
            if cfg is None:
                continue
            label = cfg.label
            prev = merged_by_label.get(label, {})
            upstream = upstream_by_label.get(label, {})
            npz = np.load(os.path.join(results_dir, name), allow_pickle=True)

            rebuilt.append(
                {
                    "label": label,
                    "display_label": prev.get("display_label") or upstream.get("display_label") or label,
                    "method": prev.get("method", "UCCSD"),
                    "energy": float(_npz_scalar(npz, "ccsd_total", 0.0)),
                    "correlation_energy": float(_npz_scalar(npz, "ccsd_corr", 0.0)),
                    "converged": bool(_npz_scalar(npz, "ccsd_converged", False)),
                    "family": cfg.spin_isomer.family if cfg.spin_isomer else "",
                    "s_squared": float(_npz_scalar(npz, "spin_sq", 0.0)) if "spin_sq" in npz else None,
                    "two_s": float(_npz_scalar(npz, "two_s", 0.0)) if "two_s" in npz else None,
                    "two_sz_fe1": float(_npz_scalar(npz, "two_sz_fe1", 0.0)) if "two_sz_fe1" in npz else None,
                    "two_sz_fe2": float(_npz_scalar(npz, "two_sz_fe2", 0.0)) if "two_sz_fe2" in npz else None,
                }
            )

        rebuilt.sort(key=lambda row: row["energy"] if row.get("energy") is not None else float("inf"))
        return rebuilt

    @property
    def ccsd_scripts_dir(self) -> str:
        return os.path.join(self._step_dir("step4_ccsd"), "scripts")

    @property
    def ccsd_t_scripts_dir(self) -> str:
        return os.path.join(self._step_dir("step5_ccsd_t"), "scripts")

    @property
    def ccsdt_scripts_dir(self) -> str:
        return os.path.join(self._step_dir("step6_ccsdt"), "scripts")

    @property
    def dmrg_basis_results_dir(self) -> str:
        return os.path.join(self._step_dir("step7_dmrg_basis"), "results")

    @property
    def dmrg_results_dir(self) -> str:
        return os.path.join(self._step_dir("step8_dmrg"), "results")

    @property
    def fno_results_dir(self) -> str:
        return os.path.join(self._step_dir("step11_fno_uccsdtq"), "results")

    # ── Step 5: CCSD(T) ──────────────────────────────────────────

    def save_ccsd_t_picked(self, labels):
        d = self._step_dir("step5_ccsd_t")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_ccsd_t_picked(self) -> list[str]:
        d = self._step_dir("step5_ccsd_t")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_ccsd_t_summary(self, results: list[dict]):
        d = self._step_dir("step5_ccsd_t")
        _write_json(os.path.join(d, "ccsd_t_summary.json"), results)
        self.mark_step_completed("step5_ccsd_t")

    def load_ccsd_t_summary(self) -> list[dict]:
        d = self._step_dir("step5_ccsd_t")
        summary_path = os.path.join(d, "ccsd_t_summary.json")
        if not os.path.exists(summary_path):
            legacy_path = os.path.join(d, "ccsdt_summary.json")
            summary_path = legacy_path
        return _read_json(summary_path)

    # ── Step 6: CCSDT ───────────────────────────────────────────

    def save_ccsdt_picked(self, labels):
        d = self._step_dir("step6_ccsdt")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_ccsdt_picked(self) -> list[str]:
        d = self._step_dir("step6_ccsdt")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_ccsdt_summary(self, results: list[dict]):
        d = self._step_dir("step6_ccsdt")
        _write_json(os.path.join(d, "ccsdt_summary.json"), results)
        self.mark_step_completed("step6_ccsdt")

    def load_ccsdt_summary(self) -> list[dict]:
        d = self._step_dir("step6_ccsdt")
        return _read_json(os.path.join(d, "ccsdt_summary.json"))

    # ── Step 7: DMRG orbital-basis preparation ────────────────

    def save_dmrg_basis_picked(self, labels):
        d = self._step_dir("step7_dmrg_basis")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_dmrg_basis_picked(self) -> list[str]:
        d = self._step_dir("step7_dmrg_basis")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_dmrg_basis_summary(self, results: list[dict]):
        d = self._step_dir("step7_dmrg_basis")
        _write_json(os.path.join(d, "dmrg_basis_summary.json"), results)
        self.mark_step_completed("step7_dmrg_basis")

    def load_dmrg_basis_summary(self) -> list[dict]:
        d = self._step_dir("step7_dmrg_basis")
        return _read_json(os.path.join(d, "dmrg_basis_summary.json"))

    # ── Step 8: DMRG solve ─────────────────────────────────────

    def save_dmrg_picked(self, labels):
        d = self._step_dir("step8_dmrg")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_dmrg_picked(self) -> list[str]:
        d = self._step_dir("step8_dmrg")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_dmrg_summary(self, results: list[dict]):
        d = self._step_dir("step8_dmrg")
        _write_json(os.path.join(d, "dmrg_summary.json"), results)
        self.mark_step_completed("step8_dmrg")

    def load_dmrg_summary(self) -> list[dict]:
        d = self._step_dir("step8_dmrg")
        return _read_json(os.path.join(d, "dmrg_summary.json"))

    # ── Step 9: DMRG extrapolation ─────────────────────────────

    def save_dmrg_extrapolation_summary(self, results: list[dict]):
        d = self._step_dir("step9_extrapolate")
        _write_json(os.path.join(d, "dmrg_extrapolation_summary.json"), results)
        self.mark_step_completed("step9_extrapolate")

    def load_dmrg_extrapolation_summary(self) -> list[dict]:
        d = self._step_dir("step9_extrapolate")
        return _read_json(os.path.join(d, "dmrg_extrapolation_summary.json"))

    # ── Step 10: final reporting ────────────────────────────────

    def save_final_summary(self, summary: list[dict], markdown: Optional[str] = None):
        d = self._step_dir("step10_report")
        _write_json(os.path.join(d, "final_summary.json"), summary)
        if markdown is not None:
            with open(os.path.join(d, "final_report.md"), "w") as f:
                f.write(markdown)
        self.mark_step_completed("step10_report")

    def load_final_summary(self) -> list[dict]:
        d = self._step_dir("step10_report")
        return _read_json(os.path.join(d, "final_summary.json"))

    # ── Step 11: FNO-UCCSDTQ ────────────────────────────────────

    def save_fno_picked(self, labels):
        d = self._step_dir("step11_fno_uccsdtq")
        _write_json(os.path.join(d, "picked_configs.json"), {"labels": labels})

    def load_fno_picked(self) -> list[str]:
        d = self._step_dir("step11_fno_uccsdtq")
        return _read_json(os.path.join(d, "picked_configs.json"))["labels"]

    def save_fno_summary(self, results: list[dict]):
        d = self._step_dir("step11_fno_uccsdtq")
        _write_json(os.path.join(d, "fno_summary.json"), results)
        self.mark_step_completed("step11_fno_uccsdtq")

    def load_fno_summary(self) -> list[dict]:
        d = self._step_dir("step11_fno_uccsdtq")
        return _read_json(os.path.join(d, "fno_summary.json"))

    # ── Step 12: CC composite ───────────────────────────────────

    def save_cc_composite_summary(self, results: list[dict]):
        d = self._step_dir("step12_cc_composite")
        _write_json(os.path.join(d, "cc_composite_summary.json"), results)
        self.mark_step_completed("step12_cc_composite")

    def load_cc_composite_summary(self) -> list[dict]:
        d = self._step_dir("step12_cc_composite")
        return _read_json(os.path.join(d, "cc_composite_summary.json"))
